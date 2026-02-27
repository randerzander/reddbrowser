#!/usr/bin/env python3
"""Main application file for the Reddit Browser TUI."""

from textual.app import App, ComposeResult
from textual.containers import Grid, VerticalScroll, Horizontal, Vertical
from textual.binding import Binding
from textual.widgets import Static, Header, Footer, Button, Label, Input
from textual import events
from textual.message import Message
from textual.screen import ModalScreen
import os
from .api import get_first_two_pages, RedditAPI
from .hn_api import HackerNewsAPI
from .twitter_api import TwitterAPI, tweet_to_post
from .media import (
    OPENAI_AVAILABLE,
    generate_text_summary,
    generate_comments_summary,
    generate_ai_response,
    extract_article_text,
    download_image,
    download_image_sync,
    open_image_in_viewer,
)
from .comments import build_comment_tree, flatten_comments
from .http_headers import get_default_headers
from .text_utils import html_to_text
from typing import Dict, Optional
import html
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
from urllib.parse import urlparse
import subprocess
import logging
import re
import shutil
import sys
from rich.markup import escape as rich_escape
from rich.text import Text
from rich.table import Table
from rich.markup import render as render_markup

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Define as None if not available


LINK_PATTERN = re.compile(r"(https?://[^\s\]\)<>\"']+)", re.IGNORECASE)


def linkify(text: str) -> str:
    """Escape text and wrap plain URLs in Rich link markup."""
    if not text:
        return text

    text = rich_escape(text)

    def _wrap(match: re.Match) -> str:
        url = match.group(1)
        safe_url = url.replace('"', "%22")
        safe_text = rich_escape(url)
        return f"[link=\"{safe_url}\"]{safe_text}[/link]"

    return LINK_PATTERN.sub(_wrap, text)


def _disable_all_logging() -> None:
    """Disable all logging, including httpx/httpcore, to keep the TUI clean."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.addHandler(logging.NullHandler())
    root_logger.propagate = False
    logging.disable(logging.CRITICAL)

    for name in ("httpx", "httpcore", "httpcore.connection", "textual", "rich", "openai"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.disabled = True


_disable_all_logging()


def _copy_external(text: str) -> bool:
    """Copy text to clipboard using external utilities if available."""
    if shutil.which("wl-copy"):
        subprocess.run(["wl-copy"], input=text, text=True, check=False)
        return True
    if shutil.which("xclip"):
        subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=False)
        return True
    if shutil.which("xsel"):
        subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, check=False)
        return True
    if shutil.which("pbcopy"):
        subprocess.run(["pbcopy"], input=text, text=True, check=False)
        return True
    return False


def _copy_osc52(text: str) -> bool:
    """Copy to clipboard via OSC 52 terminal escape sequence."""
    try:
        data = base64.b64encode(text.encode("utf-8")).decode("ascii")
        sys.stdout.write(f"\x1b]52;c;{data}\x07")
        sys.stdout.flush()
        return True
    except Exception:
        return False


def copy_text_to_clipboard(text: str, app: Optional[App] = None) -> bool:
    """Try app clipboard, external tools, then OSC52."""
    if app is not None:
        copy_fn = getattr(app, "copy_to_clipboard", None)
        if callable(copy_fn):
            try:
                copy_fn(text)
                return True
            except Exception:
                pass
    if _copy_external(text):
        return True
    return _copy_osc52(text)

def _clipboard_warning() -> Optional[str]:
    if (
        shutil.which("wl-copy")
        or shutil.which("xclip")
        or shutil.which("xsel")
        or shutil.which("pbcopy")
    ):
        return None
    return "Clipboard tool not found. Install xclip (X11) or wl-clipboard (Wayland)."

class PostCard(Static, can_focus=True):
    """Widget to display a single Reddit post that can be focused."""

    def __init__(self, post_data: Dict, index: int, numbered_title: str = None):
        super().__init__()
        self.post_data = post_data
        self.index = index
        self.title = html.unescape(post_data["data"]["title"])
        self.numbered_title = numbered_title or self.title
        self.author = post_data["data"]["author"]
        self.score = post_data["data"]["score"]
        self.num_comments = post_data["data"]["num_comments"]
        self.url = post_data["data"]["url"]
        self.permalink = post_data["data"]["permalink"]
        self.selftext = html.unescape(post_data["data"].get("selftext", ""))
        self.selftext_html = post_data["data"].get("selftext_html", "") or ""
        if not self.selftext.strip() and self.selftext_html:
            self.selftext = self._html_to_text(self.selftext_html)

        # Truncate selftext if too long
        if len(self.selftext) > 100:
            self.selftext = self.selftext[:97] + "..."

        stats = f"{self.score} / {self.num_comments}"
        row = Table.grid(expand=True)
        row.add_column(ratio=1)
        row.add_column(justify="right", no_wrap=True)
        row.add_row(
            Text(self.numbered_title, style="green"),
            Text(stats, style="cyan"),
        )
        self.update(row)

    def on_click(self) -> None:
        """Handle click event."""
        self.focus()

    def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            # Send message to parent to handle post selection
            self.post_message(PostSelected(self.index))

    def on_focus(self) -> None:
        """Handle when the widget gets focus."""
        self.styles.background = "darkgray"
        self.styles.underline = True

    def on_blur(self) -> None:
        """Handle when the widget loses focus."""
        self.styles.background = "black"
        self.styles.underline = False


class PostSelected(Message):
    """Message sent when a post is selected."""

    def __init__(self, post_index: int):
        self.post_index = post_index
        super().__init__()


class CommentScreen(ModalScreen):
    """Screen to display post comments."""

    BINDINGS = [
        ("ctrl+c", "app.quit", "Quit"),
        ("ctrl+q", "ignore", "Disabled"),
        Binding("ctrl+a", "toggle_ai_column", "Toggle AI Column", priority=True),
        Binding("v", "view_image", "View Image/Gallery", priority=True),
    ]

    def __init__(self, post_data: Dict):
        super().__init__()
        self.post_data = post_data
        self.source = post_data.get("source", "reddit")
        self.title = html.unescape(post_data["data"]["title"])
        self.author = post_data["data"]["author"]
        self.score = post_data["data"]["score"]
        self.num_comments = post_data["data"]["num_comments"]
        self.url = post_data["data"]["url"]
        self.permalink = post_data["data"]["permalink"]
        self.selftext = html.unescape(post_data["data"].get("selftext", ""))
        self.hn_comments_url = post_data["data"].get("hn_comments_url")
        self.hn_id = post_data["data"].get("hn_id")
        self.label = Label("")
        self.caption_content_text = ""  # Source of truth for caption content
        self.all_comments = []  # Store all comments
        self.expanded_comments = set()  # Track expanded comments
        self.comments_per_page = 20  # Increased to show more comments
        self.current_comment_page = 0
        self.selected_comment_index = 0  # Track which comment is conceptually selected
        self.last_input_value = ""  # Track the last input value
        self._ai_column_visible = False
        self._config = None
        self.setup_logging()

    def setup_logging(self):
        """Setup file-based logging for debugging."""
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = True

    def _parse_config(self, path: str) -> Dict:
        if not path or not os.path.exists(path):
            return {}

        config: Dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key:
                        config[key] = value
        except Exception:
            return {}

        return config

    def _get_config(self) -> Dict:
        if self._config is None:
            config_path = os.getenv("REDD_BROWSER_CONFIG", "config.yaml")
            self._config = self._parse_config(config_path)
        return self._config

    def _get_vlm_model(self) -> str:
        env_model = os.getenv("VLM_MODEL")
        if env_model:
            return env_model
        config = self._get_config()
        config_model = config.get("vlm_model") if config else None
        if config_model:
            return config_model
        return "qwen/qwen-2.5-vl-7b-instruct:free"

    def action_ignore(self) -> None:
        """Ignore a keybinding (used to disable defaults like Ctrl+Q)."""
        return

    def action_toggle_ai_column(self) -> None:
        """Toggle visibility of the AI column."""
        self._ai_column_visible = not self._ai_column_visible
        self._apply_ai_column_visibility()

    def _apply_ai_column_visibility(self) -> None:
        """Apply current AI column visibility to the layout."""
        captions_col = self.query_one("#captions_column", Vertical)
        comments_col = self.query_one("#comments_column", VerticalScroll)
        prompt_input = self.query_one("#ai_prompt_input", Input)

        if self._ai_column_visible:
            captions_col.styles.display = "block"
            comments_col.styles.width = "1fr"
            captions_col.styles.width = "1fr"
            prompt_input.can_focus = True
        else:
            captions_col.styles.display = "none"
            comments_col.styles.width = "100%"
            prompt_input.can_focus = False

        self.refresh(layout=True)

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        yield Horizontal(
            VerticalScroll(self.label, id="comments_column"),
            Vertical(
                Static("[bold]AI Generated[/bold]", id="ai_header"),
                VerticalScroll(
                    Label("", id="caption_content", markup=True),
                    id="ai_content_area"
                ),
                Horizontal(
                    Input(placeholder="Ask about this post...", id="ai_prompt_input"),
                    Button("Submit", variant="primary", id="ai_submit_button"),
                ),
                id="captions_column"
            ),
            id="main_container"
        )
        yield Footer()

    def on_mount(self) -> None:
        """Set up styles and load content after mounting."""
        # Style the main container to divide space evenly
        self.query_one("#main_container", Horizontal).styles.height = "1fr"

        # Style the columns to have equal width
        comments_col = self.query_one("#comments_column", VerticalScroll)
        captions_col = self.query_one("#captions_column", Vertical)
        comments_col.styles.width = "1fr"
        captions_col.styles.width = "1fr"
        comments_col.styles.border = ("solid", "blue")
        captions_col.styles.border = ("solid", "green")
        self._apply_ai_column_visibility()
        self.label.styles.width = "100%"
        self.label.styles.text_wrap = "wrap"

        # Style the AI content area to take 80% of the column
        ai_content_area = self.query_one("#ai_content_area", VerticalScroll)
        ai_content_area.styles.height = "80%"

        # Style the caption content label to wrap text
        caption_content = self.query_one("#caption_content", Label)
        caption_content.styles.width = "100%"
        caption_content.styles.text_justify = "left"
        caption_content.can_focus = False

        warning = _clipboard_warning()
        if warning:
            self.notify(warning, severity="warning", timeout=8)

        # Style the prompt input and button
        prompt_input = self.query_one("#ai_prompt_input", Input)
        submit_button = self.query_one("#ai_submit_button", Button)

        # Style the input and button
        prompt_input.styles.width = "1fr"  # Take remaining space
        submit_button.styles.width = "12"  # Fixed width for button
        prompt_input.styles.height = "3"  # Fixed height for input
        submit_button.styles.height = "3"  # Fixed height for button

        prompt_input.can_focus = False
        submit_button.can_focus = True

        # Add some debugging to ensure widgets are properly configured
        self.logger.info(f"Input widget: {prompt_input}, ID: {prompt_input.id}")
        self.logger.info(f"Button widget: {submit_button}, ID: {submit_button.id}")

        # Load the post and comments without blocking the UI thread
        self.call_later(lambda: asyncio.create_task(self.load_comments()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle when any button is pressed."""
        if event.button.id == "ai_submit_button":
            self.handle_ai_submission()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle when any input is submitted (Enter pressed)."""
        if event.input.id == "ai_prompt_input":
            self.handle_ai_submission()

    def handle_ai_submission(self) -> None:
        """Centralized logic for submitting an AI prompt."""
        input_widget = self.query_one("#ai_prompt_input", Input)
        user_prompt = input_widget.value.strip()

        if not OPENAI_AVAILABLE:
            self.notify("OpenAI not available for AI interaction", severity="error", timeout=10)
            return

        if not user_prompt:
            self.notify("Please enter a prompt", severity="warning")
            return

        # Clear the input field
        input_widget.value = ""
        
        # Process the AI request
        asyncio.create_task(self.process_ai_request(user_prompt))

    def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "escape":
            self.dismiss()
        elif event.key == "j":
            self.next_comment_page()
        elif event.key == "k":
            self.prev_comment_page()
        elif event.key in ("+", "plus"):
            self.expand_selected_comment()
        elif event.key in ("-", "minus"):
            self.collapse_selected_comment()
        elif event.key == "down":
            self.select_next_comment()
        elif event.key == "up":
            self.select_previous_comment()
        elif event.key == "v":
            self.action_view_image()

    def action_view_image(self) -> None:
        """View image or gallery based on the current post."""
        if self._is_twitter():
            twitter_images = self._get_twitter_all_image_urls()
            if twitter_images:
                self._open_urls_with_xdg_open(twitter_images)
                return
        if self.is_gallery_post():
            self.open_gallery_first_image()
        elif self.is_image_post(self.url):
            self.view_image()
        elif self.url:
            self.open_url_in_browser()

    def _get_twitter_all_image_urls(self) -> list[str]:
        """Return all attached Twitter image URLs from viewed + context tweets."""
        urls = []
        seen = set()

        def _add(url: str) -> None:
            if not isinstance(url, str):
                return
            value = url.strip()
            if not value or value in seen:
                return
            seen.add(value)
            urls.append(value)

        data = self.post_data.get("data", {})
        for url in data.get("twitter_media_urls") or []:
            _add(url)
        for entry in data.get("twitter_context_entries") or []:
            for url in (entry.get("twitter_media_urls") or []):
                _add(url)
        return urls

    def _get_twitter_first_image_url(self) -> Optional[str]:
        """Return the first attached Twitter image URL if present."""
        media_urls = self.post_data.get("data", {}).get("twitter_media_urls") or []
        for url in media_urls:
            if isinstance(url, str) and url.strip():
                return url.strip()
        return None

    def _open_urls_with_xdg_open(self, urls: list[str]) -> None:
        """Open one or more URLs using xdg-open."""
        opened = 0
        for url in urls:
            try:
                subprocess.Popen(["xdg-open", url])
                opened += 1
            except FileNotFoundError:
                self.notify("xdg-open not found. Install xdg-utils.", severity="error", timeout=8)
                return
            except Exception:
                continue
        if opened:
            self.notify(f"Opened {opened} image URL{'s' if opened != 1 else ''} with xdg-open")
        else:
            self.notify("No Twitter image URLs could be opened.", severity="error", timeout=6)

    def open_url_in_browser(self) -> None:
        """Open the post URL in the default browser."""
        try:
            subprocess.Popen(["xdg-open", self.url])
            self.notify("Opened link in browser")
        except FileNotFoundError:
            self.notify("xdg-open not found. Install xdg-utils.", severity="error", timeout=8)

    def on_link_clicked(self, event) -> None:
        """Copy clicked links to the clipboard."""
        link = getattr(event, "link", None) or getattr(event, "href", None)
        if not link:
            return
        if copy_text_to_clipboard(link, app=self.app):
            self.notify("Copied link to clipboard", timeout=1.5)
        else:
            self.notify("Clipboard copy not available", severity="error")
        event.stop()

    async def process_ai_request(self, user_prompt: str):
        """Process the AI request with post text, VLM caption, top 5 comments, and user prompt."""
        try:
            self.logger.info("Starting AI request processing")

            # Notify that processing has started
            self.notify("Processing your request with AI...", timeout=3)

            # Show that we're processing in the AI column
            self._update_caption_column("[yellow]AI is thinking...[/yellow]", append=True)

            # Gather the required information
            post_text = self.selftext if self.selftext.strip() else "No post text provided."

            # Get the current caption/content in the AI column
            current_caption_content = self._get_caption_content()

            # Get top 5 comments
            top_comments = self.get_top_comments()

            # Prepare the full prompt for the LLM
            full_prompt = f"""
            Context about the Reddit post:
            - Post text: {post_text}

            AI-generated content about the post:
            - {current_caption_content}

            Top comments on the post:
            {top_comments}

            User's specific question:
            {user_prompt}

            Please provide a helpful response based on all this information.
            """

            self.logger.info("Calling generate_ai_response")

            # Generate the response
            response = await generate_ai_response(full_prompt)

            self.logger.info("Received response from AI")

            # Notify that the response has been received
            self.notify("LLM response received! Updating display...", timeout=2)

            # Append the response to the AI content
            user_query_response = f"[bold magenta]Your Question:[/bold magenta] {user_prompt}\n\n[bold cyan]AI Response:[/bold cyan] {response}"
            self._update_caption_column(user_query_response, append=True)

            # Scroll to the bottom to show the new content
            ai_content_area = self.query_one("#ai_content_area", VerticalScroll)
            ai_content_area.scroll_end(animate=False)

            self.logger.info("Successfully updated display with AI response")

        except Exception as e:
            self.logger.error(f"Error processing AI request: {str(e)}")

            # Notify about the error
            self.notify(f"Error processing AI request: {str(e)}", severity="error", timeout=10)

            error_msg = f"\n\n[red]Error processing AI request: {str(e)}[/red]"
            current_content = self._get_caption_content()
            self._set_caption_content(current_content + error_msg)

    def get_top_comments(self, limit: int = 10) -> str:
        """Extract the top comments from the post."""
        try:
            # Get top-level comments (already sorted by score in build_comment_tree)
            top_comments = []

            # Limit to top comments
            for i, comment in enumerate(self.all_comments[:limit]):
                comment_author = comment["data"].get("author", "[deleted]")
                comment_body = html.unescape(comment["data"].get("body", ""))
                comment_score = comment["data"].get("score", 0)

                author_label = self._format_author(comment_author)
                top_comments.append(f"{i+1}. Author: {author_label}, Score: {comment_score}\n   Comment: {comment_body}")

            if not top_comments:
                return "No comments available."

            return "\n".join(top_comments)
        except Exception:
            return "Could not retrieve comments."

    def _set_caption_content(self, content: str) -> None:
        """Update caption content in the UI and internal label."""
        self.caption_content_text = content or ""
        caption_scroll = self.query_one("#caption_content", Label)
        self._update_label_safe(caption_scroll, content)

    def _get_caption_content(self) -> str:
        """Get the current caption content tracked by the screen."""
        return self.caption_content_text or ""

    def _update_label_safe(self, label: Label, content: str) -> None:
        """Update a Rich/markup label, falling back to plain text on any error."""
        try:
            label.update(render_markup(content))
        except Exception:
            # Fall back to plain text renderable (no markup parsing).
            label.update(Text(content))

    def _html_to_text(self, content_html: str) -> str:
        """Best-effort conversion of HTML content to plain text."""
        return html_to_text(content_html)
    def _set_caption_for_generation(self, loading_message: str, start_fn, unavailable_message: str) -> None:
        """Set loading UI, then kick off generation if available."""
        self._set_caption_content(loading_message)
        if OPENAI_AVAILABLE:
            self.call_later(start_fn)
        else:
            self._set_caption_content(unavailable_message)

    def _is_hacker_news(self) -> bool:
        return self.source == "hn"

    def _is_twitter(self) -> bool:
        return self.source == "twitter"

    def _format_author(self, author: str) -> str:
        if self._is_hacker_news():
            return author
        if self._is_twitter():
            return f"@{author}"
        return f"u/{author}"

    def _format_title_with_media_badge(self) -> str:
        """Format the post title; for tweets with images, show [img] at right."""
        title = " ".join((self.title or "").split())
        if not (self._is_twitter() and self._get_twitter_first_image_url()):
            return rich_escape(title)

        badge = "[img]"
        # Approximate available width for the content area.
        width = max(30, int(getattr(self.size, "width", 80)) - 8)
        usable_title = max(8, width - len(badge) - 1)
        if len(title) > usable_title:
            title = title[: usable_title - 3] + "..."
        pad = max(1, width - len(title) - len(badge))
        return f"{rich_escape(title)}{' ' * pad}{rich_escape(badge)}"

    def _twitter_context_block(self) -> str:
        """Format quoted/replied-to tweet context for tweet detail view."""
        if not self._is_twitter():
            return ""
        entries = self.post_data.get("data", {}).get("twitter_context_entries") or []
        if not entries:
            return ""

        lines = ["[bold]Tweet Context:[/bold]"]
        for entry in entries:
            label = rich_escape(str(entry.get("label", "Context")))
            author = rich_escape(str(entry.get("author", "unknown")))
            text = linkify(str(entry.get("text", "") or ""))
            url = str(entry.get("url", "") or "")
            lines.append(f"{label}: [yellow]@{author}[/yellow]")
            if text:
                lines.append(f"[green]{text}[/green]")
            if url:
                lines.append(f"URL: [green]{linkify(url)}[/green]")
            lines.append("")
        return "\n".join(lines).strip()

    def _summary_source_text(self) -> str:
        """Build source text for AI summary generation."""
        base_text = (self.selftext or "").strip()
        if not self._is_twitter():
            return base_text

        parts = []
        author = self.post_data.get("data", {}).get("author", self.author)
        if base_text:
            parts.append(f"Viewed tweet by @{author}:\n{base_text}")
        else:
            parts.append(f"Viewed tweet by @{author}: [No text]")

        entries = self.post_data.get("data", {}).get("twitter_context_entries") or []
        linked_entries = [e for e in entries if "linked" in str(e.get("kind", "")) or "referenced" in str(e.get("kind", ""))]
        if not linked_entries and entries:
            linked_entries = entries[:1]

        for entry in linked_entries[:2]:
            label = str(entry.get("label", "Linked Tweet"))
            entry_author = str(entry.get("author", "unknown"))
            entry_text = str(entry.get("text", "") or "[No text]")
            parts.append(f"{label} by @{entry_author}:\n{entry_text}")

        return "\n\n".join(parts).strip()

    async def load_comments(self):
        """Load the post content and comments."""
        try:
            if self._is_hacker_news():
                await self.load_hn_comments()
            elif self._is_twitter():
                await self.load_twitter_comments()
            else:
                # Fetch comments from Reddit API
                reddit = RedditAPI()
                try:
                    data = await reddit.get_comments_async(self.permalink)
                finally:
                    await reddit.aclose()

                    # Extract comments data
                    comments_data = data[1]["data"]["children"] if len(data) > 1 else []

                    # Build a tree structure for nested comments
                self.all_comments = build_comment_tree(comments_data)

                # Initially expand all comments by adding all comment IDs with replies to expanded_comments
                self.expand_all_comments()

                # Determine what to show in the right column based on post content
                has_selftext = bool(self.selftext.strip())
                is_image = self.is_image_post(self.url)

                if is_image:
                    # Image post (with or without text) - generate image caption
                    self._set_caption_for_generation(
                        "[yellow]Generating image caption...[/yellow]",
                        self.start_image_description_generation,
                        "[red]OpenAI not available for caption generation[/red]",
                    )
                elif has_selftext:
                    # Text post only - generate text summary
                    self._set_caption_for_generation(
                        "[yellow]Generating text summary...[/yellow]",
                        self.start_text_summarization,
                        "[red]OpenAI not available for text summarization[/red]",
                    )
                else:
                    # Link post - attempt to fetch and summarize article content
                    if self.url and self.url.startswith("http"):
                        self._set_caption_for_generation(
                            "[yellow]Fetching article content...[/yellow]",
                            self.start_article_summarization,
                            "[red]OpenAI not available for article summarization[/red]",
                        )
                    else:
                        # Neither image nor text - show placeholder
                        self._set_caption_content("[blue]No content to summarize[/blue]")

                # Display the first page of comments
                self.display_comments()

        except Exception as e:
            author_label = rich_escape(self._format_author(self.author))
            error_content = (
                f"[bold][green]{self._format_title_with_media_badge()}[/green][/bold]\n\n"
                f"Author: [green]{author_label}[/green]\n"
                f"Score: [green]{self.score}[/green]\n"
                f"Comments: [green]{self.num_comments}[/green]\n"
                f"URL: [green]{linkify(self.url)}[/green]\n"
            )
            if self.hn_comments_url:
                error_content += f"HN Comments: [green]{linkify(self.hn_comments_url)}[/green]\n"
            error_content += "\n"
            context_block = self._twitter_context_block()
            if context_block:
                error_content += f"{context_block}\n\n"

            if self.selftext.strip():
                error_content += f"Content:\n[green]{linkify(self.selftext)}[/green]\n\n"

            error_content += f"[red]Error loading comments: {str(e)}[/red]\n\n"
            error_content += "[yellow]Press ESC to return[/yellow]"

            self._update_label_safe(self.label, error_content)

            # Update caption panel with error or placeholder
            caption_content = f"[red]Error loading AI content: {str(e)}[/red]"
            self._set_caption_content(caption_content)

    async def load_twitter_comments(self) -> None:
        """Load replies for a tweet."""
        tweet_id = self.post_data["data"].get("twitter_tweet_id")
        if not tweet_id:
            self.all_comments = []
            self.display_comments()
            return

        api = TwitterAPI(
            cookies_file=os.getenv("TWITTER_COOKIES_FILE", "cookies.json"),
            locale=os.getenv("TWITTER_LOCALE", "en-US"),
        )
        tweet, reply_tree = await api.get_tweet_and_reply_tree(str(tweet_id))
        self.all_comments = reply_tree
        try:
            self.post_data["data"]["twitter_context_entries"] = await api.get_tweet_context_entries(tweet)
        except Exception:
            self.post_data["data"]["twitter_context_entries"] = []
        latest_media_urls = []
        for media in list(getattr(tweet, "media", []) or []):
            media_type = str(getattr(media, "type", "") or "").lower()
            media_url = str(getattr(media, "media_url", "") or "").strip()
            if media_type == "photo" and media_url:
                latest_media_urls.append(media_url)
        if latest_media_urls:
            self.post_data["data"]["twitter_media_urls"] = latest_media_urls
        self.expand_all_comments()

        has_selftext = bool(self.selftext.strip())
        if has_selftext:
            self._set_caption_for_generation(
                "[yellow]Generating text summary...[/yellow]",
                self.start_text_summarization,
                "[red]OpenAI not available for text summarization[/red]",
            )
        elif self.url and self.url.startswith("http"):
            self._set_caption_for_generation(
                "[yellow]Fetching article content...[/yellow]",
                self.start_article_summarization,
                "[red]OpenAI not available for article summarization[/red]",
            )
        else:
            self._set_caption_content("[blue]No content to summarize[/blue]")

        self.display_comments()

    async def load_hn_comments(self) -> None:
        """Load Hacker News comments for a story."""
        if not self.hn_id:
            self.all_comments = []
            self.display_comments()
            return
        hn = HackerNewsAPI()
        try:
            self.all_comments = await hn.get_comments_tree_async(int(self.hn_id))
        finally:
            await hn.aclose()

        self.expand_all_comments()

        has_selftext = bool(self.selftext.strip())
        if has_selftext:
            self._set_caption_for_generation(
                "[yellow]Generating text summary...[/yellow]",
                self.start_text_summarization,
                "[red]OpenAI not available for text summarization[/red]",
            )
        elif self.url and self.url.startswith("http"):
            self._set_caption_for_generation(
                "[yellow]Fetching article content...[/yellow]",
                self.start_article_summarization,
                "[red]OpenAI not available for article summarization[/red]",
            )
        else:
            self._set_caption_content("[blue]No content to summarize[/blue]")

        self.display_comments()

    def expand_all_comments(self):
        """Initially expand all comments that have replies."""
        def traverse_comments(comments):
            for comment in comments:
                if len(comment.get("replies", [])) > 0:
                    self.expanded_comments.add(comment["data"]["id"])
                    traverse_comments(comment["replies"])

        traverse_comments(self.all_comments)

    def select_next_comment(self):
        """Select the next comment."""
        flattened_comments = flatten_comments(self.all_comments, self.expanded_comments)
        if flattened_comments and self.selected_comment_index < len(flattened_comments) - 1:
            self.selected_comment_index += 1
            self.display_comments()

    def select_previous_comment(self):
        """Select the previous comment."""
        if self.selected_comment_index > 0:
            self.selected_comment_index -= 1
            self.display_comments()

    def expand_selected_comment(self):
        """Expand the currently selected comment."""
        flattened_comments = flatten_comments(self.all_comments, self.expanded_comments)
        if 0 <= self.selected_comment_index < len(flattened_comments):
            comment = flattened_comments[self.selected_comment_index]
            if len(comment.get("replies", [])) > 0:
                comment_id = comment["data"]["id"]
                # Add to expanded regardless of current state
                if comment_id not in self.expanded_comments:
                    self.expanded_comments.add(comment_id)
                    self.display_comments()
                    self.notify(f"Expanded comment by {comment['data'].get('author', 'unknown')}")
                else:
                    self.notify("Comment is already expanded")

    def collapse_selected_comment(self):
        """Collapse the currently selected comment."""
        flattened_comments = flatten_comments(self.all_comments, self.expanded_comments)
        if 0 <= self.selected_comment_index < len(flattened_comments):
            comment = flattened_comments[self.selected_comment_index]
            # Check if the comment has replies that can be collapsed AND is currently expanded
            if comment.get("replies") and len(comment.get("replies", [])) > 0:
                comment_id = comment["data"]["id"]
                # Only collapse if it's currently expanded
                if comment_id in self.expanded_comments:
                    self.expanded_comments.remove(comment_id)
                    self.display_comments()
                    self.notify(f"Collapsed comment by {comment['data'].get('author', 'unknown')}")
                else:
                    self.notify("Comment is already collapsed")

    def next_comment_page(self):
        """Show next page of comments."""
        flattened_comments = flatten_comments(self.all_comments, self.expanded_comments)
        if flattened_comments and (self.current_comment_page + 1) * self.comments_per_page < len(flattened_comments):
            self.current_comment_page += 1
            self.display_comments()

    def prev_comment_page(self):
        """Show previous page of comments."""
        if self.current_comment_page > 0:
            self.current_comment_page -= 1
            self.display_comments()

    def display_comments(self):
        """Display the current page of comments with nesting."""
        # Flatten the comment tree for display
        flattened_comments = flatten_comments(self.all_comments, self.expanded_comments)

        start_idx = self.current_comment_page * self.comments_per_page
        end_idx = min(start_idx + self.comments_per_page, len(flattened_comments))

        # Check if this is an image post
        is_image_post = self.is_image_post(self.url)

        # Format the content
        author_label = rich_escape(self._format_author(self.author))
        content = (
            f"[bold][green]{self._format_title_with_media_badge()}[/green][/bold]\n\n"
            f"Author: [green]{author_label}[/green]\n"
            f"Score: [green]{self.score}[/green]\n"
            f"Comments: [green]{self.num_comments}[/green]\n"
            f"URL: [green]{linkify(self.url)}[/green]\n"
        )
        if self.hn_comments_url:
            content += f"HN Comments: [green]{linkify(self.hn_comments_url)}[/green]\n"
        content += "\n"
        context_block = self._twitter_context_block()
        if context_block:
            content += f"{context_block}\n\n"

        # If it's an image post and term-image is available, show a message about image display
        if is_image_post:
            content += f"[bold]IMAGE POST:[/bold]\n"
            content += f"[green]This is an image post: {linkify(self.url)}[/green]\n"
            content += f"[yellow]Press 'v' to open image in GUI viewer (feh, eog, etc.)[/yellow]\n\n"
        else:
            # Regular post content
            if self.selftext.strip():
                content += f"Content:\n[green]{linkify(self.selftext)}[/green]\n\n"
            else:
                content += "[yellow]Link post detected: article summary will appear in the AI panel.[/yellow]\n\n"

        content += "[bold]COMMENTS:[/bold]\n\n"

        # Add comments for current page
        for i in range(start_idx, end_idx):
            comment = flattened_comments[i]
            comment_data = comment["data"]
            author = comment_data.get("author", "[deleted]")
            safe_author = rich_escape(author)
            body = html.unescape(comment_data.get("body", ""))
            body = linkify(body)
            score = comment_data.get("score", 0)
            level = comment["level"]

            # Add indentation based on level
            indent = "  " * level

            # Check if this comment has replies and is expanded
            has_replies = len(comment.get("replies", [])) > 0
            is_expanded = comment_data["id"] in self.expanded_comments

            # Add expand/collapse indicator
            expand_indicator = "[+] " if has_replies and not is_expanded else "[-] " if has_replies and is_expanded else "    "

            # Highlight the selected comment
            is_selected = (i == self.selected_comment_index)
            author_prefix = "" if (self._is_hacker_news() or self._is_twitter()) else "u/"
            if is_selected:
                content += f"{indent}{expand_indicator}[red on white]Comment by {author_prefix}{safe_author} (Score: {score}):[/red on white]\n"
                content += f"{indent}[red on white]{body}[/red on white]\n\n"
            else:
                content += f"{indent}{expand_indicator}Comment by {author_prefix}[yellow]{safe_author}[/yellow] (Score: {score}):\n"
                content += f"{indent}[green]{body}[/green]\n\n"

        # Add pagination info
        total_pages = (len(flattened_comments) + self.comments_per_page - 1) // self.comments_per_page
        content += f"[yellow]Page {self.current_comment_page + 1} of {total_pages}[/yellow] | "
        content += f"[yellow]j/k: page up/down, ↑/↓: select comment, +/-: expand/collapse, v: view image in GUI, ESC: return[/yellow]"

        self._update_label_safe(self.label, content)

    def is_image_post(self, url: str) -> bool:
        """Check if the post URL points to an image."""
        from .media import is_image_url
        return is_image_url(url)

    def is_gallery_post(self) -> bool:
        """Check if the post is a Reddit gallery."""
        data = self.post_data.get("data", {})
        if data.get("is_gallery") and data.get("media_metadata"):
            return True
        url = (data.get("url") or "").lower()
        return "/gallery/" in url and bool(data.get("media_metadata") or data.get("gallery_data"))

    def _get_gallery_first_image_url(self) -> Optional[str]:
        """Get the first image URL from a Reddit gallery post."""
        data = self.post_data.get("data", {})
        if not data.get("is_gallery"):
            url = (data.get("url") or "").lower()
            if "/gallery/" not in url:
                return None

        media_metadata = data.get("media_metadata")
        gallery_data = data.get("gallery_data")

        if not media_metadata:
            try:
                permalink = data.get("permalink")
                api_url = None
                if permalink:
                    api_url = f"https://www.reddit.com{permalink}.json"
                else:
                    url = data.get("url") or ""
                    match = re.search(r"/gallery/([a-z0-9]+)", url, re.IGNORECASE)
                    if match:
                        post_id = match.group(1)
                        api_url = f"https://www.reddit.com/comments/{post_id}.json"

                if api_url:
                    reddit = RedditAPI()
                    try:
                        listing = reddit.get_json(api_url)
                    finally:
                        reddit.close()
                    if listing and isinstance(listing, list) and listing[0].get("data", {}).get("children"):
                        post_data = listing[0]["data"]["children"][0]["data"]
                        media_metadata = post_data.get("media_metadata")
                        gallery_data = post_data.get("gallery_data")
            except Exception:
                media_metadata = media_metadata or {}
                gallery_data = gallery_data or {}

        media_metadata = media_metadata or {}
        gallery_data = gallery_data or {}
        items = gallery_data.get("items") or []

        media_id = None
        if items:
            media_id = items[0].get("media_id")
        if not media_id and media_metadata:
            media_id = next(iter(media_metadata.keys()), None)
        if not media_id:
            return None

        meta = media_metadata.get(media_id) or {}
        url = None
        if isinstance(meta.get("s"), dict):
            url = meta["s"].get("u")
        if not url and isinstance(meta.get("p"), list) and meta["p"]:
            url = meta["p"][-1].get("u")
        if not url:
            return None

        return html.unescape(url)

    def open_gallery_first_image(self) -> None:
        """Open the first image in a Reddit gallery using an image viewer."""
        url = self._get_gallery_first_image_url()
        if not url:
            self.notify("Gallery image not available.", severity="error", timeout=5)
            return

        async def _open_async():
            image_path = await download_image(url)
            if not image_path:
                self.notify("Failed to download gallery image.", severity="error", timeout=6)
                return
            viewer_used = open_image_in_viewer(image_path)
            if not viewer_used:
                self.notify("No image viewer found. Install 'feh' or 'eog'", severity="error", timeout=10)
                return
            self.notify(f"Opened gallery image with {viewer_used}")

        asyncio.create_task(_open_async())

    def view_image(self, image_url: Optional[str] = None):
        """Display the image using feh (GUI image viewer) and generate description."""
        try:
            target_url = image_url or self.url
            if not target_url:
                self.notify("No image URL available.", severity="error", timeout=6)
                return

            temp_path = download_image_sync(target_url)
            if not temp_path:
                self.notify("Failed to download image.", severity="error", timeout=6)
                return

            viewer_used = open_image_in_viewer(temp_path)

            if not viewer_used:
                self.notify("No image viewer found. Install 'feh' or 'eog'", severity="error", timeout=10)
                # Clean up the temporary file if no viewer is found
                os.unlink(temp_path)
                return
            else:
                self.notify(f"Opened image with {viewer_used}")

            # Generate description if OpenAI is available
            if OPENAI_AVAILABLE and OpenAI is not None:
                # Run the description generation in the background using the threaded approach
                asyncio.create_task(self.run_vlm_for_file_in_thread(temp_path))
            else:
                self.notify("OpenAI not available. Install with: pip install openai", severity="warning")

        except Exception as e:
            self.notify(f"Error preparing image for viewer: {str(e)}", severity="error", timeout=10)

    def start_image_description_generation(self):
        """Start the image description generation after UI is displayed."""
        self.logger.info("start_image_description_generation called")
        if OPENAI_AVAILABLE:
            self.logger.info("OpenAI is available, starting image description")
            # Show notification that captioning is starting
            self.notify("Generating image description...")

            # Update the caption area with loading message
            caption_content = "[yellow]Generating image description...[/yellow]"
            self._set_caption_content(caption_content)
            self.logger.info("Updated caption area with image description loading message")

            # Run the description generation in a separate thread to prevent blocking
            asyncio.create_task(self.run_vlm_in_thread())
            self.logger.info("Started run_vlm_in_thread task")
        else:
            self.logger.error("OpenAI not available for image description")

    def start_text_summarization(self):
        """Start the text summarization after UI is displayed."""
        self.logger.info("start_text_summarization called")
        if OPENAI_AVAILABLE:
            self.logger.info("OpenAI is available, starting summarization")
            # Show notification that summarization is starting
            self.notify("Generating text summary...")

            # Update the caption area with loading message
            caption_content = "[yellow]Generating text summary...[/yellow]"
            self._set_caption_content(caption_content)
            self.logger.info("Updated caption area with loading message")

            # Run the summarization asynchronously
            asyncio.create_task(self.run_text_summarization_async())
            self.logger.info("Started run_text_summarization_async task")
        else:
            self.logger.error("OpenAI not available for text summarization")

    def start_article_summarization(self):
        """Start the article summarization after UI is displayed."""
        self.logger.info("start_article_summarization called")
        if OPENAI_AVAILABLE:
            self.logger.info("OpenAI is available, starting article summarization")
            self.notify("Fetching article content...")

            caption_content = "[yellow]Fetching article content...[/yellow]"
            self._set_caption_content(caption_content)
            self.logger.info("Updated caption area with article fetch loading message")

            asyncio.create_task(self.run_article_summarization_async())
            self.logger.info("Started run_article_summarization_async task")
        else:
            self.logger.error("OpenAI not available for article summarization")

    async def run_text_summarization_async(self):
        """Run the text summarization asynchronously."""
        self.logger.info("run_text_summarization_async started")
        try:
            # Generate summary using the media module function
            self.logger.info("Calling generate_text_summary")
            summary = await generate_text_summary(self._summary_source_text())
            self.logger.info(f"Received summary: {summary[:100]}...")  # Log first 100 chars

            if summary and not summary.startswith("Error"):
                self.logger.info("Summary received successfully, updating caption")
                # Update the caption column with the summary (replace initial content)
                self._schedule_caption_update(summary, "text", "Text summary generated!", append=False)
                self.logger.info("Caption update scheduled")
                await self._append_top_comments_summary()
            else:
                self.logger.info(f"Error in summary: {summary}")
                # Update with error message
                error_content = f"[red]{summary}[/red]"
                self._set_caption_content(error_content)
        except Exception as e:
            self.logger.error(f"Exception in run_text_summarization_async: {str(e)}")
            error_content = f"[red]Error in text summarization: {str(e)}[/red]"
            self._set_caption_content(error_content)

    async def run_article_summarization_async(self):
        """Fetch article content and generate a summary asynchronously."""
        self.logger.info("run_article_summarization_async started")
        try:
            self.logger.info("Calling extract_article_text")
            article_text = await extract_article_text(self.url)
            self.logger.info(f"Received article text: {article_text[:100]}...")

            if not article_text or article_text.startswith("Error"):
                error_content = f"[red]{article_text or 'Error fetching article content.'}[/red]"
                self._set_caption_content(error_content)
                return

            self._set_caption_content("[yellow]Summarizing article content...[/yellow]")

            self.logger.info("Calling generate_text_summary for article")
            summary = await generate_text_summary(article_text)
            self.logger.info(f"Received article summary: {summary[:100]}...")

            if summary and not summary.startswith("Error"):
                self._schedule_caption_update(summary, "text", "Article summary generated!", append=False)
                await self._append_top_comments_summary()
            else:
                error_content = f"[red]{summary}[/red]"
                caption_scroll = self.query_one("#caption_content", Label)
                caption_scroll.update(error_content)
        except Exception as e:
            self.logger.error(f"Exception in run_article_summarization_async: {str(e)}")
            error_content = f"[red]Error in article summarization: {str(e)}[/red]"
            caption_scroll = self.query_one("#caption_content", Label)
            caption_scroll.update(error_content)

    async def _append_top_comments_summary(self) -> None:
        """Summarize top comments and append the result to the AI panel."""
        top_comments_text = self.get_top_comments(limit=10)
        if top_comments_text.startswith("No comments"):
            self._schedule_caption_update(
                "No comments available to summarize.",
                "comments",
                "Top comments summary skipped.",
                append=True,
            )
            return

        self.logger.info("Calling generate_comments_summary")
        comments_summary = await generate_comments_summary(top_comments_text)
        if comments_summary and not comments_summary.startswith("Error"):
            self._schedule_caption_update(
                comments_summary,
                "comments",
                "Top comments summary generated!",
                append=True,
            )
            return

        self.logger.info(f"Error in comments summary: {comments_summary}")
        current_content = self._get_caption_content()
        error_content = f"{current_content}\n\n[red]{comments_summary}[/red]"
        self._set_caption_content(error_content)

    async def run_vlm_in_thread(self):
        """Run the VLM call in a separate thread."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, self.generate_image_description_sync)

    async def run_vlm_for_file_in_thread(self, image_path):
        """Run the VLM call for a file in a separate thread."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, self.generate_image_description_sync_from_path, image_path)

    def _get_mime_type(self, source, is_file=True):
        """Helper method to determine the MIME type based on file extension."""

        if is_file:
            file_ext = os.path.splitext(source)[1].lower()
        else:
            parsed_url = urlparse(source)
            file_ext = os.path.splitext(parsed_url.path)[1].lower()

        if file_ext in ['.png']:
            return 'image/png'
        elif file_ext in ['.gif']:
            return 'image/gif'
        else:
            return 'image/jpeg'  # default

    def _get_openai_client(self):
        """Helper method to initialize and return the OpenAI client."""

        # Check if OpenAI is available before proceeding
        if OpenAI is None:
            self.notify("OpenAI library not available. Install with: pip install openai", severity="error", timeout=10)
            return None

        # Get the API key from environment variable
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            self.notify("OPENROUTER_API_KEY not set in environment", severity="error", timeout=10)
            return None

        # Initialize the OpenAI client with OpenRouter
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def _generate_image_description(self, image_data, mime_type):
        """Helper method to generate image description using OpenAI."""
        client = self._get_openai_client()
        if not client:
            return None

        # Call the model to generate a description
        response = client.chat.completions.create(
            model=self._get_vlm_model(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Provide a comprehensive description of what you see in the image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        return self._extract_response_content(response)

    def _extract_response_content(self, response):
        """Extract text content from OpenAI-style responses with provider quirks."""
        try:
            message = response.choices[0].message
        except Exception:
            message = None

        content = None
        if message is not None:
            try:
                content = message.content
            except Exception:
                content = None

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        parts.append(text)
            content = "\n".join(parts) if parts else None

        if (not content or not str(content).strip()) and message is not None:
            for field in ("output_text", "text", "reasoning"):
                value = getattr(message, field, None)
                if value and str(value).strip():
                    content = value
                    break

        if not content or not str(content).strip():
            try:
                payload = response.model_dump()
            except Exception:
                payload = repr(response)
            self.logger.error("Empty response content. Raw response: %s", payload)
            return None

        return str(content).strip()

    def _update_caption_column(self, content, content_type="image", append=False):
        """Helper method to update the caption column with the AI-generated content."""
        self.logger.info(f"_update_caption_column called with content_type={content_type}, append={append}")

        # Determine the heading based on content type
        if content_type == "image":
            heading = "[bold blue]Image Caption:[/bold blue]\n"
        elif content_type == "text":
            heading = "[bold green]Text Summary:[/bold green]\n"
        elif content_type == "comments":
            heading = "[bold cyan]Top Comments Summary:[/bold cyan]\n"
        else:
            heading = ""

        # Get current content if we're appending
        if append:
            # Get current content from the actual DOM element
            current_content = self._get_caption_content()
            # Update the caption column by appending the new content
            caption_content = f"{current_content}\n\n{heading}[green]{content}[/green]"
        else:
            # Replace the entire content
            caption_content = f"{heading}[green]{content}[/green]"

        self.logger.info(f"Updating caption with content: {caption_content[:100]}...")  # First 100 chars

        # Update the DOM element directly
        if not self._ai_column_visible:
            self._ai_column_visible = True
            self._apply_ai_column_visibility()
        self._set_caption_content(caption_content)
        self.logger.info("Caption content updated")

        return caption_content

    def _schedule_caption_update(self, description, content_type="image", success_msg="AI content generated!", append=False):
        """Helper method to schedule caption updates on the main thread."""
        self.logger.info(f"_schedule_caption_update called with content_type={content_type}, append={append}")

        # Use Textual's worker system to schedule updates on the main thread
        async def update_ui():
            self.logger.info("Executing update_ui function")
            self._update_caption_column(description, content_type, append)
            self.logger.info("Caption column updated, sending notification")
            self.notify(success_msg)
            self.logger.info("Notification sent")

        # Schedule the update on the main thread
        self.logger.info("Scheduling update_ui function")
        self.call_later(update_ui)

    def _handle_image_description_error(self, error):
        """Helper method to handle image description errors."""
        async def show_error():
            self.notify(f"Error generating image description: {str(error)}", severity="error", timeout=10)

        # Schedule the error notification on the main thread
        self.call_later(show_error)

    def generate_image_description_sync_from_path(self, image_path):
        """Synchronous version of image description generation from a file path to run in a thread."""
        try:
            # Read the image file and encode it to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Determine the image format based on file extension
            mime_type = self._get_mime_type(image_path, is_file=True)

            # Generate the description
            description = self._generate_image_description(image_data, mime_type)
            if description is None:
                return  # Error already notified

            # Update the caption column with the description (replace initial content)
            self._schedule_caption_update(description, "image", "Image caption generated!", append=False)
        except Exception as e:
            self._handle_image_description_error(e)

    def generate_image_description_sync(self):
        """Synchronous version of image description generation to run in a thread."""
        try:
            # Download the image from the post URL
            response = requests.get(self.url, headers=get_default_headers())
            response.raise_for_status()

            # Encode the image to base64
            image_data = base64.b64encode(response.content).decode('utf-8')

            # Determine the image format based on URL
            mime_type = self._get_mime_type(self.url, is_file=False)

            # Generate the description
            description = self._generate_image_description(image_data, mime_type)
            if description is None:
                return  # Error already notified

            # Update the caption column with the description (replace initial content)
            self._schedule_caption_update(description, "image", "Image caption generated!", append=False)
        except Exception as e:
            self._handle_image_description_error(e)

class RedditBrowserApp(App):
    """A Textual app for browsing Reddit."""
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "ignore", "Disabled"),
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("j", "next_page", "Next 20 Posts"),
        ("k", "prev_page", "Previous 20 Posts"),
        ("n", "next_subreddit", "Next Subreddit"),
        ("p", "prev_subreddit", "Previous Subreddit"),
    ]
    
    def __init__(self, subreddit: str = "LocalLlama"):
        super().__init__()
        self.subreddit = subreddit
        self.posts = []
        self.current_page = 0
        self.posts_per_page = 20
        self._number_buffer = ""
        self._subreddits = self._load_subreddits()
        self._subreddit_index = self._resolve_subreddit_index()
        self._hn_subreddit = "news.ycombinator.com"
        self._twitter_subreddit = "twitter"

    def _get_subreddits_path(self) -> str:
        app_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(app_dir))
        return os.path.join(project_root, "subreddits.txt")

    def _load_subreddits(self) -> list:
        path = self._get_subreddits_path()
        if not os.path.exists(path):
            return []
        subreddits = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    subreddits.append(line)
        except Exception:
            return []
        return subreddits

    def _resolve_subreddit_index(self) -> int:
        if not self._subreddits:
            return -1
        lowered = [item.lower() for item in self._subreddits]
        try:
            return lowered.index(self.subreddit.lower())
        except ValueError:
            return -1

    def action_ignore(self) -> None:
        """Ignore a keybinding (used to disable defaults like Ctrl+Q)."""
        return

    def _subreddit_label(self) -> str:
        if self._is_hacker_news_subreddit() or self._is_twitter_subreddit():
            return self.subreddit
        return f"r/{self.subreddit}"

    def _update_app_title(self) -> None:
        self.title = self._subreddit_label()
        try:
            header = self.query_one("#custom_header", Static)
            row = Table.grid(expand=True)
            row.add_column(ratio=1, justify="center")
            row.add_column(justify="right", no_wrap=True)
            row.add_row(
                Text(self._subreddit_label(), style="bold white"),
                Text("Score / Comments", style="white"),
            )
            header.update(row)
        except Exception:
            pass
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Static("", id="custom_header")
        yield VerticalScroll(Grid(id="posts_grid"))
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        header = self.query_one("#custom_header", Static)
        header.styles.height = 1
        header.styles.background = "darkblue"
        header.styles.color = "white"
        header.styles.padding = (0, 1)

        self._update_app_title()
        self.load_posts()
    
    def load_posts(self) -> None:
        """Load posts from the subreddit."""
        self._update_app_title()
        try:
            if self._is_hacker_news_subreddit():
                self.load_hn_posts()
                return
            if self._is_twitter_subreddit():
                self.load_twitter_posts()
                return

            # Get first two pages of posts
            all_posts = get_first_two_pages(self.subreddit, user_agent=os.getenv("REDDIT_USER_AGENT"))
            self.posts = [post for post in all_posts if not post["data"].get("stickied", False)]
            
            # Reset to first page when loading new posts
            self.current_page = 0
            
            # Update the grid
            self.update_grid()
        except Exception as e:
            self.notify(f"Error loading posts: {str(e)}", severity="error", timeout=10)

    def _is_hacker_news_subreddit(self) -> bool:
        return self.subreddit.strip().lower() == self._hn_subreddit

    def _is_twitter_subreddit(self) -> bool:
        return self.subreddit.strip().lower() == self._twitter_subreddit

    def _hn_story_to_post(self, story: Dict) -> Dict:
        story_id = story.get("id")
        hn_comments_url = f"https://news.ycombinator.com/item?id={story_id}" if story_id else ""
        url = story.get("url") or hn_comments_url
        return {
            "source": "hn",
            "data": {
                "id": str(story_id) if story_id is not None else "",
                "title": story.get("title", ""),
                "author": story.get("by", "[deleted]"),
                "score": story.get("score", 0),
                "num_comments": story.get("descendants", 0),
                "url": url,
                "permalink": hn_comments_url,
                "selftext": html_to_text(story.get("text", "")),
                "hn_comments_url": hn_comments_url,
                "hn_id": story_id,
            },
        }

    def load_hn_posts(self) -> None:
        """Load top Hacker News stories."""
        hn = HackerNewsAPI()
        try:
            stories = hn.get_top_stories(limit=50)
        finally:
            hn.close()

        self.posts = [self._hn_story_to_post(story) for story in stories]
        self.current_page = 0
        self.update_grid()

    def load_twitter_posts(self) -> None:
        """Load latest tweets from the authenticated home timeline."""
        api = TwitterAPI(
            cookies_file=os.getenv("TWITTER_COOKIES_FILE", "cookies.json"),
            locale=os.getenv("TWITTER_LOCALE", "en-US"),
        )
        tweets = api.get_latest_timeline_sync(limit=50)
        self.posts = [tweet_to_post(tweet) for tweet in tweets]
        self.current_page = 0
        self.update_grid()
    
    def update_grid(self) -> None:
        """Update the grid with current posts."""
        grid = self.query_one("#posts_grid", Grid)

        # Clear existing posts
        grid.remove_children()

        # Calculate the start and end indices for the current page
        start_idx = self.current_page * self.posts_per_page
        end_idx = min(start_idx + self.posts_per_page, len(self.posts))

        # Configure grid layout - single column
        grid.styles.grid_size_columns = 1
        grid.styles.grid_gutter = "0"
        grid.styles.overflow = "auto"

        # Add post cards to the grid with numbering for the current page
        for i in range(start_idx, end_idx):
            # Add numbering to the title (relative to current page, not global)
            page_number = i - start_idx + 1
            numbered_title = f"{page_number}. {html.unescape(self.posts[i]['data']['title'])}"
            post_card = PostCard(self.posts[i], i, numbered_title=numbered_title)
            post_card.styles.height = "1"  # Single line per post
            post_card.styles.background = "black"
            post_card.styles.color = "white"
            post_card.can_focus = True
            grid.mount(post_card)

    def on_post_selected(self, message: PostSelected) -> None:
        """Handle when a post is selected."""
        # Convert the page-relative index to global index
        page_relative_index = message.post_index
        global_index = self.current_page * self.posts_per_page + page_relative_index

        if 0 <= global_index < len(self.posts):
            post_data = self.posts[global_index]
            # Push the comment screen
            self.push_screen(CommentScreen(post_data))
    
    def action_refresh(self) -> None:
        """Refresh the posts."""
        self.load_posts()
        self.notify("Posts refreshed!")

    def action_next_page(self) -> None:
        """Go to next page of posts."""
        if len(self.posts) > (self.current_page + 1) * self.posts_per_page:
            self.current_page += 1
            self.update_grid()
            self.notify(f"Showing posts {(self.current_page * self.posts_per_page) + 1}-{min((self.current_page + 1) * self.posts_per_page, len(self.posts))}")

    def action_prev_page(self) -> None:
        """Go to previous page of posts."""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_grid()
            self.notify(f"Showing posts {(self.current_page * self.posts_per_page) + 1}-{min((self.current_page + 1) * self.posts_per_page, len(self.posts))}")

    def action_next_subreddit(self) -> None:
        """Switch to the next subreddit in subreddits.txt."""
        if not self._subreddits:
            self.notify("No subreddits.txt list found.", severity="error", timeout=4)
            return
        if self._subreddit_index < 0:
            next_index = 0
        else:
            next_index = (self._subreddit_index + 1) % len(self._subreddits)
        next_subreddit = self._subreddits[next_index]
        if next_subreddit.lower() == self.subreddit.lower():
            return
        self.subreddit = next_subreddit
        self._subreddit_index = next_index
        self._update_app_title()
        self.load_posts()
        self.notify(f"Switched to {self._subreddit_label()}")

    def action_prev_subreddit(self) -> None:
        """Switch to the previous subreddit in subreddits.txt."""
        if not self._subreddits:
            self.notify("No subreddits.txt list found.", severity="error", timeout=4)
            return
        if self._subreddit_index < 0:
            prev_index = len(self._subreddits) - 1
        else:
            prev_index = (self._subreddit_index - 1) % len(self._subreddits)
        prev_subreddit = self._subreddits[prev_index]
        if prev_subreddit.lower() == self.subreddit.lower():
            return
        self.subreddit = prev_subreddit
        self._subreddit_index = prev_index
        self._update_app_title()
        self.load_posts()
        self.notify(f"Switched to {self._subreddit_label()}")

    def on_key(self, event: events.Key) -> None:
        """Handle key press events, including number input for direct post selection."""
        # Check if the key is a digit
        if event.key.isdigit():
            # Store the digit for later processing
            self._number_buffer += event.key

            # Show what number has been entered so far
            self.notify(f"Entered: {self._number_buffer}", timeout=1.0)

            # Process the number after a short delay to allow multi-digit input
            self.set_timer(0.3, self.process_entered_number)  # Reduced from 1.0 to 0.3 seconds
            event.prevent_default()  # Prevent default handling

    def process_entered_number(self) -> None:
        """Process the number entered by the user."""
        if self._number_buffer:
            if self._number_buffer.isdigit():
                post_num = int(self._number_buffer)
                if 1 <= post_num <= len(self.posts):
                    # Calculate which page this post is on
                    post_index = post_num - 1  # Convert to 0-based index
                    target_page = post_index // self.posts_per_page

                    # Change to the target page
                    self.current_page = target_page
                    self.update_grid()

                    # Notify user about the navigation
                    self.notify(f"Loading post {post_num}...")

                    # Clear the buffer
                    self._number_buffer = ""

                    # Open the comments for this post immediately
                    self.open_post_comments(post_index)
                else:
                    self.notify(f"Invalid post number. Please enter a number between 1 and {len(self.posts)}.")
                    self._number_buffer = ""
            else:
                self._number_buffer = ""

    def open_post_comments(self, post_index: int) -> None:
        """Open the comments for the specified post."""
        if 0 <= post_index < len(self.posts):
            post_data = self.posts[post_index]
            # Push the comment screen directly
            self.push_screen(CommentScreen(post_data))

def main():
    """Main entry point."""
    import sys
    subreddit = sys.argv[1] if len(sys.argv) > 1 else "LocalLlama"
    app = RedditBrowserApp(subreddit=subreddit)
    app.run()


if __name__ == "__main__":
    main()
