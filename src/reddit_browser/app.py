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
from .api import get_first_two_pages
from .media import (
    OPENAI_AVAILABLE,
    generate_text_summary,
    generate_comments_summary,
    generate_ai_response,
    extract_article_text,
    download_image,
    open_image_in_viewer,
)
from .http_headers import get_default_headers
from typing import Dict, Optional
import html
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
from urllib.parse import urlparse
import tempfile
import subprocess
import httpx
import logging
import re
import shutil
import sys
from rich.markup import escape as rich_escape
from rich.text import Text
from rich.markup import render as render_markup

try:
    import term_image.image as _term_image
    TERM_IMAGE_AVAILABLE = True
except ImportError:
    TERM_IMAGE_AVAILABLE = False

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

        # Simple content display - just the title in green
        content = f"[green]{rich_escape(self.numbered_title)}[/green]"
        self.update(content)

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
        self.title = html.unescape(post_data["data"]["title"])
        self.author = post_data["data"]["author"]
        self.score = post_data["data"]["score"]
        self.num_comments = post_data["data"]["num_comments"]
        self.url = post_data["data"]["url"]
        self.permalink = post_data["data"]["permalink"]
        self.selftext = html.unescape(post_data["data"].get("selftext", ""))
        self.label = Label("")
        self.caption_label = Label("")  # For VLM captions
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

        if self._ai_column_visible:
            captions_col.styles.display = "block"
            comments_col.styles.width = "1fr"
            captions_col.styles.width = "1fr"
        else:
            captions_col.styles.display = "none"
            comments_col.styles.width = "100%"

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

        prompt_input.can_focus = True
        submit_button.can_focus = True

        # Add some debugging to ensure widgets are properly configured
        self.logger.info(f"Input widget: {prompt_input}, ID: {prompt_input.id}")
        self.logger.info(f"Button widget: {submit_button}, ID: {submit_button.id}")

        # Add event listener for the prompt input and ensure it gets focus
        self.call_later(lambda: self.ensure_input_focus())

        # Load the post and comments
        self.call_later(self.load_comments)

    def ensure_input_focus(self):
        """Ensure the prompt input gets focus after a delay."""
        try:
            prompt_input = self.query_one("#ai_prompt_input", Input)
            if prompt_input:
                prompt_input.focus()
                # Add a notification to confirm the input has focus
                self.notify("Prompt input is ready for input", timeout=1)

                # Ensure the input widget is properly configured
                prompt_input.can_focus = True
                prompt_input.focus()
        except Exception as e:
            self.logger.error(f"Error focusing input: {e}")



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
        if self.is_gallery_post():
            self.open_gallery_first_image()
        elif self.is_image_post(self.url) and TERM_IMAGE_AVAILABLE:
            self.view_image()

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
                comment_body = html.unescape(comment["data"].get("body", "")[:200])  # Limit length
                comment_score = comment["data"].get("score", 0)

                top_comments.append(f"{i+1}. Author: u/{comment_author}, Score: {comment_score}\n   Comment: {comment_body}")

            if not top_comments:
                return "No comments available."

            return "\n".join(top_comments)
        except Exception:
            return "Could not retrieve comments."

    def _set_caption_content(self, content: str) -> None:
        """Update caption content in the UI and internal label."""
        self.caption_content_text = content or ""
        self._update_label_safe(self.caption_label, content)
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
        if not content_html:
            return ""
        # Strip tags with a simple regex to avoid extra deps here.
        text = re.sub(r"<[^>]+>", " ", content_html)
        return html.unescape(" ".join(text.split()))
    def _set_caption_for_generation(self, loading_message: str, start_fn, unavailable_message: str) -> None:
        """Set loading UI, then kick off generation if available."""
        self._set_caption_content(loading_message)
        if OPENAI_AVAILABLE:
            self.call_later(start_fn)
        else:
            self._set_caption_content(unavailable_message)

    async def load_comments(self):
        """Load the post content and comments."""
        try:

            # Fetch comments from Reddit API
            url = f"https://www.reddit.com{self.permalink}.json"
            headers = get_default_headers()

            async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                # Extract comments data
                comments_data = data[1]["data"]["children"] if len(data) > 1 else []

                # Build a tree structure for nested comments
                self.all_comments = self.build_comment_tree(comments_data)

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
            error_content = (
                f"[bold][green]{rich_escape(self.title)}[/green][/bold]\n\n"
                f"Author: u/[green]{rich_escape(self.author)}[/green]\n"
                f"Score: [green]{self.score}[/green]\n"
                f"Comments: [green]{self.num_comments}[/green]\n"
                f"URL: [green]{linkify(self.url)}[/green]\n\n"
            )

            if self.selftext.strip():
                error_content += f"Content:\n[green]{linkify(self.selftext)}[/green]\n\n"

            error_content += f"[red]Error loading comments: {str(e)}[/red]\n\n"
            error_content += "[yellow]Press ESC to return[/yellow]"

            self._update_label_safe(self.label, error_content)

            # Update caption panel with error or placeholder
            caption_content = f"[red]Error loading AI content: {str(e)}[/red]"
            self._set_caption_content(caption_content)

    def expand_all_comments(self):
        """Initially expand all comments that have replies."""
        def traverse_comments(comments):
            for comment in comments:
                if len(comment.get("replies", [])) > 0:
                    self.expanded_comments.add(comment["data"]["id"])
                    traverse_comments(comment["replies"])

        traverse_comments(self.all_comments)

    def build_comment_tree(self, comments_data):
        """Build a tree structure from flat comments data."""
        def process_replies(replies_list):
            """Recursively process replies to build the tree."""
            result = []
            if not replies_list or replies_list == []:
                return result

            for item in replies_list:
                if item["kind"] == "t1":  # It's a comment
                    comment_data = item["data"]
                    comment_obj = {
                        "data": comment_data,
                        "replies": [],
                        "level": 0  # Will be set correctly during flattening
                    }

                    # Process nested replies if they exist
                    if "replies" in comment_data and comment_data["replies"]:
                        if isinstance(comment_data["replies"], dict) and "data" in comment_data["replies"]:
                            nested_replies = comment_data["replies"]["data"].get("children", [])
                            comment_obj["replies"] = process_replies(nested_replies)

                    result.append(comment_obj)

            # Sort by score descending
            result.sort(key=lambda x: x["data"].get("score", 0), reverse=True)
            return result

        # Process the top-level comments
        root_comments = []
        for item in comments_data:
            if item["kind"] == "t1":  # It's a comment
                comment_data = item["data"]
                comment_obj = {
                    "data": comment_data,
                    "replies": [],
                    "level": 0
                }

                # Process nested replies if they exist
                if "replies" in comment_data and comment_data["replies"]:
                    if isinstance(comment_data["replies"], dict) and "data" in comment_data["replies"]:
                        nested_replies = comment_data["replies"]["data"].get("children", [])
                        comment_obj["replies"] = process_replies(nested_replies)

                root_comments.append(comment_obj)

        # Sort root comments by score descending
        root_comments.sort(key=lambda x: x["data"].get("score", 0), reverse=True)
        return root_comments

    def select_next_comment(self):
        """Select the next comment."""
        flattened_comments = self.flatten_comments(self.all_comments)
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
        flattened_comments = self.flatten_comments(self.all_comments)
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
        flattened_comments = self.flatten_comments(self.all_comments)
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

    def toggle_current_comment_expansion(self):
        """Toggle expansion of the currently viewed comment."""
        # For now, we'll toggle the first comment that has replies
        # In a more advanced implementation, we'd track which comment is "focused"
        flattened_comments = self.flatten_comments(self.all_comments)

        # For simplicity, we'll just toggle the first comment with replies
        for comment in flattened_comments:
            if len(comment.get("replies", [])) > 0:
                comment_id = comment["data"]["id"]
                if comment_id in self.expanded_comments:
                    self.expanded_comments.remove(comment_id)
                else:
                    self.expanded_comments.add(comment_id)
                break

        # Redraw the comments
        self.display_comments()

    def next_comment_page(self):
        """Show next page of comments."""
        flattened_comments = self.flatten_comments(self.all_comments)
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
        flattened_comments = self.flatten_comments(self.all_comments)

        start_idx = self.current_comment_page * self.comments_per_page
        end_idx = min(start_idx + self.comments_per_page, len(flattened_comments))

        # Check if this is an image post
        is_image_post = self.is_image_post(self.url)

        # Format the content
        content = (
            f"[bold][green]{rich_escape(self.title)}[/green][/bold]\n\n"
            f"Author: u/[green]{rich_escape(self.author)}[/green]\n"
            f"Score: [green]{self.score}[/green]\n"
            f"Comments: [green]{self.num_comments}[/green]\n"
            f"URL: [green]{linkify(self.url)}[/green]\n\n"
        )

        # If it's an image post and term-image is available, show a message about image display
        if is_image_post and TERM_IMAGE_AVAILABLE:
            content += f"[bold]IMAGE POST:[/bold]\n"
            content += f"[green]This is an image post: {linkify(self.url)}[/green]\n"
            content += f"[yellow]Press 'v' to open image in GUI viewer (feh, eog, etc.)[/yellow]\n\n"
        elif is_image_post and not TERM_IMAGE_AVAILABLE:
            content += f"[green]This is an image post: {linkify(self.url)}[/green]\n"
            content += "[yellow]Install term-image to view images in terminal[/yellow]\n\n"
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
            body = html.unescape(comment_data.get("body", "")[:200])  # Limit length
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
            if is_selected:
                content += f"{indent}{expand_indicator}[red on white]Comment by u/{safe_author} (Score: {score}):[/red on white]\n"
                content += f"{indent}[red on white]{body}[/red on white]\n\n"
            else:
                content += f"{indent}{expand_indicator}Comment by u/[yellow]{safe_author}[/yellow] (Score: {score}):\n"
                content += f"{indent}[green]{body}[/green]\n\n"

        # Add pagination info
        total_pages = (len(flattened_comments) + self.comments_per_page - 1) // self.comments_per_page
        content += f"[yellow]Page {self.current_comment_page + 1} of {total_pages}[/yellow] | "
        content += f"[yellow]j/k: page up/down, ↑/↓: select comment, +/-: expand/collapse, v: view image in GUI, ESC: return[/yellow]"

        self._update_label_safe(self.label, content)

    def is_image_post(self, url: str) -> bool:
        """Check if the post URL points to an image."""
        if not url:
            return False

        # Common image extensions
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg']

        # Check if URL ends with an image extension or comes from image hosting services
        url_lower = url.lower()

        # Check for common image hosting domains
        image_domains = [
            'i.redd.it', 'i.imgur.com', 'imgur.com', 'flickr.com',
            'instagram.com', 'twitter.com', 'facebook.com',
            'cdn.discordapp.com', 'media.discordapp.net'
        ]

        # Check if it's from a known image domain
        for domain in image_domains:
            if domain in url_lower:
                # Special handling for imgur - check if it's not an album/gallery
                if 'imgur.com' in domain and any(x in url_lower for x in ['/a/', '/gallery/', 'album']):
                    return False  # Imgur albums/galleries are not single images
                return True

        # Check if URL ends with an image extension
        for ext in image_extensions:
            if url_lower.endswith(ext):
                return True

        return False

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
                    response = requests.get(api_url, headers=get_default_headers(), timeout=10)
                    response.raise_for_status()
                    listing = response.json()
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

    def view_image(self):
        """Display the image using feh (GUI image viewer) and generate description."""
        try:
            # Download the image to a temporary file
            response = requests.get(self.url, headers=get_default_headers())
            response.raise_for_status()

            # Get file extension from URL
            parsed_url = urlparse(self.url)
            file_ext = os.path.splitext(parsed_url.path)[1]
            if not file_ext:
                # Guess from content type if not in URL
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    file_ext = '.jpg'
                elif 'png' in content_type:
                    file_ext = '.png'
                elif 'gif' in content_type:
                    file_ext = '.gif'
                else:
                    file_ext = '.png'  # Default

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name

            # Try to open with feh or other image viewers
            viewers = [
                ['feh', temp_path],           # Lightweight image viewer
                ['xdg-open', temp_path],      # Generic opener
                ['eog', temp_path],           # Eye of GNOME
                ['gpicview', temp_path],      # Lightweight GTK viewer
                ['gthumb', temp_path],        # GNOME image viewer
                ['ristretto', temp_path],     # XFCE image viewer
                ['shotwell', temp_path],      # Photo manager
            ]

            viewer_used = None
            for viewer_cmd in viewers:
                try:
                    # Run the image viewer in the background so the app continues
                    subprocess.Popen(viewer_cmd)
                    viewer_used = viewer_cmd[0]
                    break
                except FileNotFoundError:
                    continue  # Viewer not installed, try next one

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
            summary = await generate_text_summary(self.selftext)
            self.logger.info(f"Received summary: {summary[:100]}...")  # Log first 100 chars

            if summary and not summary.startswith("Error"):
                self.logger.info("Summary received successfully, updating caption")
                # Update the caption column with the summary (replace initial content)
                self._schedule_caption_update(summary, "text", "Text summary generated!", append=False)
                self.logger.info("Caption update scheduled")

                # Summarize the top comments and append to the AI panel
                top_comments_text = self.get_top_comments(limit=10)
                if top_comments_text.startswith("No comments"):
                    self._schedule_caption_update(
                        "No comments available to summarize.",
                        "comments",
                        "Top comments summary skipped.",
                        append=True,
                    )
                else:
                    self.logger.info("Calling generate_comments_summary")
                    comments_summary = await generate_comments_summary(top_comments_text)
                    if comments_summary and not comments_summary.startswith("Error"):
                        self._schedule_caption_update(
                            comments_summary,
                            "comments",
                            "Top comments summary generated!",
                            append=True,
                        )
                    else:
                        self.logger.info(f"Error in comments summary: {comments_summary}")
                        current_content = self._get_caption_content()
                        error_content = f"{current_content}\n\n[red]{comments_summary}[/red]"
                        self._set_caption_content(error_content)
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
            else:
                error_content = f"[red]{summary}[/red]"
                caption_scroll = self.query_one("#caption_content", Label)
                caption_scroll.update(error_content)
        except Exception as e:
            self.logger.error(f"Exception in run_article_summarization_async: {str(e)}")
            error_content = f"[red]Error in article summarization: {str(e)}[/red]"
            caption_scroll = self.query_one("#caption_content", Label)
            caption_scroll.update(error_content)

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

    async def generate_image_description_for_post(self):
        """Generate a description of the image using OpenRouter API for the current post."""
        # Run the description generation in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, self.generate_image_description_sync)

    def flatten_comments(self, comments, level=0):
        """Flatten the comment tree for display."""
        result = []
        for comment in comments:
            # Add the current comment
            comment_copy = dict(comment)
            comment_copy["level"] = level
            result.append(comment_copy)

            # If the comment is expanded, add its replies
            comment_id = comment["data"]["id"]
            if comment_id in self.expanded_comments:
                result.extend(self.flatten_comments(comment["replies"], level + 1))

        return result


class RedditBrowserApp(App):
    """A Textual app for browsing Reddit."""
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "ignore", "Disabled"),
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("j", "next_page", "Next 20 Posts"),
        ("k", "prev_page", "Previous 20 Posts"),
    ]
    
    def __init__(self, subreddit: str = "LocalLlama"):
        super().__init__()
        self.subreddit = subreddit
        self.posts = []
        self.current_page = 0
        self.posts_per_page = 20
        self._number_buffer = ""

    def action_ignore(self) -> None:
        """Ignore a keybinding (used to disable defaults like Ctrl+Q)."""
        return
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield VerticalScroll(Grid(id="posts_grid"))
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.load_posts()
    
    def load_posts(self) -> None:
        """Load posts from the subreddit."""
        try:
            # Get first two pages of posts
            all_posts = get_first_two_pages(self.subreddit)
            self.posts = [post for post in all_posts if not post["data"].get("stickied", False)]
            
            # Reset to first page when loading new posts
            self.current_page = 0
            
            # Update the grid
            self.update_grid()
        except Exception as e:
            self.notify(f"Error loading posts: {str(e)}", severity="error", timeout=10)
    
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

    def focus_post(self, pos_in_page: int) -> None:
        """Focus the post at the given position within the current page."""
        # Get the grid containing the posts
        grid = self.query_one("#posts_grid", Grid)

        # Get all the PostCard widgets
        post_cards = grid.children

        # Make sure we have a valid index
        if 0 <= pos_in_page < len(post_cards):
            # Focus the correct post card
            post_card = post_cards[pos_in_page]
            post_card.focus()
            # Scroll to make sure it's visible
            grid.scroll_to_widget(post_card)



def main():
    """Main entry point."""
    import sys
    subreddit = sys.argv[1] if len(sys.argv) > 1 else "LocalLlama"
    app = RedditBrowserApp(subreddit=subreddit)
    app.run()


if __name__ == "__main__":
    main()
