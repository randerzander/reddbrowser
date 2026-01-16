#!/usr/bin/env python3
"""Main application file for the Reddit Browser TUI."""

from textual.app import App, ComposeResult
from textual.containers import Grid, VerticalScroll, Horizontal, Center, Middle, ScrollableContainer
from textual.widgets import Static, Header, Footer, Button, Label, Input, DataTable
from textual import events
from textual.message import Message
from textual.screen import ModalScreen
import sys
import os
from .api import RedditAPI, get_first_two_pages, get_comments_tree
from .media import is_image_url, open_image_in_viewer, generate_image_description, download_image, OPENAI_AVAILABLE
from typing import List, Dict, Set
import html
import asyncio
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from textual import work
from textual.worker import Worker, get_current_worker

try:
    from term_image.image import from_url, from_file
    TERM_IMAGE_AVAILABLE = True
except ImportError:
    TERM_IMAGE_AVAILABLE = False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Define as None if not available


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

        # Truncate selftext if too long
        if len(self.selftext) > 100:
            self.selftext = self.selftext[:97] + "..."

        # Simple content display - just the title in green
        content = f"[green]{self.numbered_title}[/green]"
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
        self.all_comments = []  # Store all comments
        self.expanded_comments = set()  # Track expanded comments
        self.comments_per_page = 20  # Increased to show more comments
        self.current_comment_page = 0
        self.selected_comment_index = 0  # Track which comment is conceptually selected

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        yield VerticalScroll(self.label)
        yield Footer()

    def on_mount(self) -> None:
        """Load the post and comments when mounted."""
        self.call_later(self.load_comments)

    async def load_comments(self):
        """Load the post content and comments."""
        try:
            import httpx

            # Fetch comments from Reddit API
            url = f"https://www.reddit.com{self.permalink}.json"
            headers = {"User-Agent": "RedditBrowser/0.1.0"}

            async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                # Extract post and comments data
                post_info = data[0]["data"]["children"][0]["data"]
                comments_data = data[1]["data"]["children"] if len(data) > 1 else []

                # Build a tree structure for nested comments
                self.all_comments = self.build_comment_tree(comments_data)

                # Initially expand all comments by adding all comment IDs with replies to expanded_comments
                self.expand_all_comments()

                # Display the first page of comments
                self.display_comments()

                # Check if this is an image post and schedule description generation if OpenAI is available
                if self.is_image_post(self.url) and OPENAI_AVAILABLE:
                    # Schedule the description generation to happen after the content is displayed
                    self.call_later(self.start_image_description_generation)

        except Exception as e:
            error_content = (
                f"[bold][green]{self.title}[/green][/bold]\n\n"
                f"Author: u/[green]{self.author}[/green]\n"
                f"Score: [green]{self.score}[/green]\n"
                f"Comments: [green]{self.num_comments}[/green]\n"
                f"URL: [green]{self.url}[/green]\n\n"
            )

            if self.selftext.strip():
                error_content += f"Content:\n[green]{self.selftext}[/green]\n\n"

            error_content += f"[red]Error loading comments: {str(e)}[/red]\n\n"
            error_content += "[yellow]Press ESC to return[/yellow]"

            self.label.update(error_content)

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

    def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "escape":
            self.dismiss()  # Use dismiss() for screens instead of pop_screen()
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
            # View image if this is an image post
            if self.is_image_post(self.url) and TERM_IMAGE_AVAILABLE:
                self.view_image()

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
            f"[bold][green]{self.title}[/green][/bold]\n\n"
            f"Author: u/[green]{self.author}[/green]\n"
            f"Score: [green]{self.score}[/green]\n"
            f"Comments: [green]{self.num_comments}[/green]\n"
            f"URL: [green]{self.url}[/green]\n\n"
        )

        # If it's an image post and term-image is available, show a message about image display
        if is_image_post and TERM_IMAGE_AVAILABLE:
            content += f"[bold]IMAGE POST:[/bold]\n"
            content += f"[green]This is an image post: {self.url}[/green]\n"
            content += f"[yellow]Press 'v' to open image in GUI viewer (feh, eog, etc.)[/yellow]\n\n"
        elif is_image_post and not TERM_IMAGE_AVAILABLE:
            content += f"[green]This is an image post: {self.url}[/green]\n"
            content += "[yellow]Install term-image to view images in terminal[/yellow]\n\n"
        else:
            # Regular post content
            if self.selftext.strip():
                content += f"Content:\n[green]{self.selftext}[/green]\n\n"

        content += "[bold]COMMENTS:[/bold]\n\n"

        # Add comments for current page
        for i in range(start_idx, end_idx):
            comment = flattened_comments[i]
            comment_data = comment["data"]
            author = comment_data.get("author", "[deleted]")
            body = html.unescape(comment_data.get("body", "")[:200])  # Limit length
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
                content += f"{indent}{expand_indicator}[red on white]Comment by u/{author} (Score: {score}):[/red on white]\n"
                content += f"{indent}[red on white]{body}[/red on white]\n\n"
            else:
                content += f"{indent}{expand_indicator}Comment by u/[yellow]{author}[/yellow] (Score: {score}):\n"
                content += f"{indent}[green]{body}[/green]\n\n"

        # Add pagination info
        total_pages = (len(flattened_comments) + self.comments_per_page - 1) // self.comments_per_page
        content += f"[yellow]Page {self.current_comment_page + 1} of {total_pages}[/yellow] | "
        content += f"[yellow]j/k: page up/down, ↑/↓: select comment, +/-: expand/collapse, v: view image in GUI, ESC: return[/yellow]"

        self.label.update(content)

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

    def view_image(self):
        """Display the image using feh (GUI image viewer) and generate description."""
        try:
            import requests
            from urllib.parse import urlparse
            import tempfile
            import subprocess

            # Download the image to a temporary file
            response = requests.get(self.url, headers={"User-Agent": "RedditBrowser/0.1.0"})
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
                self.notify("No image viewer found. Install 'feh' or 'eog'", severity="error")
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
            self.notify(f"Error preparing image for viewer: {str(e)}", severity="error")

    async def generate_image_description(self, image_path):
        """Generate a description of the image using OpenRouter API."""
        try:
            import os
            import base64

            # Read the image file and encode it to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Get the API key from environment variable
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                self.notify("OPENROUTER_API_KEY not set in environment", severity="error")
                return

            # Initialize the OpenAI client with OpenRouter
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )

            # Call the model to generate a description
            response = client.chat.completions.create(
                model="qwen/qwen-2.5-vl-7b-instruct:free",
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

            # Get the description
            description = response.choices[0].message.content

            # Update the label to include the description right after the image information
            current_content = self.label.renderable.plain if hasattr(self.label.renderable, 'plain') else str(self.label.renderable)

            # Find the position to insert the description - after the image info but before comments
            if "Press 'v' to open image in GUI viewer" in current_content:
                # Insert right after the image viewer instruction
                insertion_point = current_content.find("Press 'v' to open image in GUI viewer") + len("Press 'v' to open image in GUI viewer")
                updated_content = current_content[:insertion_point] + f"\n\n[yellow]IMAGE DESCRIPTION:[/yellow]\n[green]{description}[/green]" + current_content[insertion_point:]
            else:
                # If not found, append at the end
                updated_content = current_content + f"\n\n[yellow]IMAGE DESCRIPTION:[/yellow]\n[green]{description}[/green]"

            # Update the label - this is already on the main thread since it's called from the UI context
            self.label.update(updated_content)

            self.notify("Image description generated and displayed")

        except Exception as e:
            self.notify(f"Error generating image description: {str(e)}", severity="error")

    def start_image_description_generation(self):
        """Start the image description generation after UI is displayed."""
        if OPENAI_AVAILABLE:
            # Show notification that captioning is starting
            self.notify("Generating image description...")
            # Run the description generation in a separate thread to prevent blocking
            asyncio.create_task(self.run_vlm_in_thread())

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

    def generate_image_description_sync_from_path(self, image_path):
        """Synchronous version of image description generation from a file path to run in a thread."""
        try:
            import os
            import base64
            from urllib.parse import urlparse

            # Check if OpenAI is available before proceeding
            if OpenAI is None:
                self.notify("OpenAI library not available. Install with: pip install openai", severity="error")
                return

            # Read the image file and encode it to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Get the API key from environment variable
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                self.notify("OPENROUTER_API_KEY not set in environment", severity="error")
                return

            # Determine the image format based on file extension
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext in ['.png']:
                mime_type = 'image/png'
            elif file_ext in ['.gif']:
                mime_type = 'image/gif'
            else:
                mime_type = 'image/jpeg'  # default

            # Initialize the OpenAI client with OpenRouter
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )

            # Call the model to generate a description
            response = client.chat.completions.create(
                model="qwen/qwen-2.5-vl-7b-instruct:free",
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

            # Get the description
            description = response.choices[0].message.content

            # Update the label to include the description right after the image information
            current_content = self.label.renderable.plain if hasattr(self.label.renderable, 'plain') else str(self.label.renderable)

            # Find the position to insert the description - after the image info but before comments
            if "Press 'v' to open image in GUI viewer" in current_content:
                # Insert right after the image viewer instruction
                insertion_point = current_content.find("Press 'v' to open image in GUI viewer") + len("Press 'v' to open image in GUI viewer")
                updated_content = current_content[:insertion_point] + f"\n\n[yellow]IMAGE DESCRIPTION:[/yellow]\n[green]{description}[/green]" + current_content[insertion_point:]
            else:
                # If not found, append at the end
                updated_content = current_content + f"\n\n[yellow]IMAGE DESCRIPTION:[/yellow]\n[green]{description}[/green]"

            # Schedule UI updates on the main thread using Textual's proper mechanism
            from textual.worker import Worker, get_current_worker
            import asyncio

            # Use Textual's worker system to schedule updates on the main thread
            async def update_ui():
                self.label.update(updated_content)
                self.notify("Image description added to post!")

            # Schedule the update on the main thread
            self.call_later(update_ui)

        except Exception as e:
            from textual.worker import Worker, get_current_worker
            import asyncio

            async def show_error():
                self.notify(f"Error generating image description: {str(e)}", severity="error")

            # Schedule the error notification on the main thread
            self.call_later(show_error)

    def generate_image_description_sync(self):
        """Synchronous version of image description generation to run in a thread."""
        try:
            import os
            import base64
            import requests
            from urllib.parse import urlparse

            # Check if OpenAI is available before proceeding
            if OpenAI is None:
                self.notify("OpenAI library not available. Install with: pip install openai", severity="error")
                return

            # Download the image from the post URL
            response = requests.get(self.url, headers={"User-Agent": "RedditBrowser/0.1.0"})
            response.raise_for_status()

            # Encode the image to base64
            image_data = base64.b64encode(response.content).decode('utf-8')

            # Get the API key from environment variable
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                self.notify("OPENROUTER_API_KEY not set in environment", severity="error")
                return

            # Determine the image format based on URL
            parsed_url = urlparse(self.url)
            file_ext = os.path.splitext(parsed_url.path)[1].lower()
            if file_ext in ['.png']:
                mime_type = 'image/png'
            elif file_ext in ['.gif']:
                mime_type = 'image/gif'
            else:
                mime_type = 'image/jpeg'  # default

            # Initialize the OpenAI client with OpenRouter
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )

            # Call the model to generate a description
            response = client.chat.completions.create(
                model="qwen/qwen-2.5-vl-7b-instruct:free",
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

            # Get the description
            description = response.choices[0].message.content

            # Update the label to include the description right after the image information
            current_content = self.label.renderable.plain if hasattr(self.label.renderable, 'plain') else str(self.label.renderable)

            # Find the position to insert the description - after the image info but before comments
            if "Press 'v' to open image in GUI viewer" in current_content:
                # Insert right after the image viewer instruction
                insertion_point = current_content.find("Press 'v' to open image in GUI viewer") + len("Press 'v' to open image in GUI viewer")
                updated_content = current_content[:insertion_point] + f"\n\n[yellow]IMAGE DESCRIPTION:[/yellow]\n[green]{description}[/green]" + current_content[insertion_point:]
            else:
                # If not found, append at the end
                updated_content = current_content + f"\n\n[yellow]IMAGE DESCRIPTION:[/yellow]\n[green]{description}[/green]"

            # Schedule UI updates on the main thread using Textual's proper mechanism
            from textual.worker import Worker, get_current_worker
            import asyncio

            # Use Textual's worker system to schedule updates on the main thread
            async def update_ui():
                self.label.update(updated_content)
                self.notify("Image description added to post!")

            # Schedule the update on the main thread
            self.call_later(update_ui)

        except Exception as e:
            from textual.worker import Worker, get_current_worker
            import asyncio

            async def show_error():
                self.notify(f"Error generating image description: {str(e)}", severity="error")

            # Schedule the error notification on the main thread
            self.call_later(show_error)

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
            self.notify(f"Error loading posts: {str(e)}", severity="error")
    
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