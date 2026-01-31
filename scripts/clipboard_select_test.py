#!/usr/bin/env python3
"""Standalone Textual test: copy selected text to clipboard."""

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, TextArea, Static
import shutil
import subprocess


SAMPLE_TEXT = """\
This is a selection-to-clipboard test.

Select any text with the mouse and it will be copied to your clipboard.

Some links to try:
https://www.reddit.com/r/LocalLLaMA/
https://example.com/path?query=hello&x=1

You can select across lines too.
"""


class ReadOnlyTextArea(TextArea):
    """A read-only TextArea that still supports mouse selection."""

    def on_key(self, event) -> None:
        # Prevent edits while still allowing selection/copy.
        event.prevent_default()


class SelectionCopyApp(App):
    """App that copies any selected text to the system clipboard."""

    CSS = """
    Screen {
        background: black;
    }
    #container {
        padding: 1 2;
    }
    #title {
        margin-bottom: 1;
        color: white;
    }
    #text {
        height: 1fr;
        border: round $accent;
    }
    """

    def __init__(self):
        super().__init__()
        self._last_copied = ""

    def _copy_external(self, text: str) -> bool:
        """Fallback copy via external clipboard utilities."""
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

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Select text to copy it automatically.", id="title"),
            ReadOnlyTextArea(SAMPLE_TEXT, id="text"),
            id="container",
        )
        yield Footer()

    def on_text_area_selection_changed(self, event) -> None:
        text_area = event.text_area
        if hasattr(text_area, "get_selected_text"):
            selection = text_area.get_selected_text()
        else:
            selection = getattr(text_area, "selected_text", "")
        if not selection:
            return
        if selection == self._last_copied:
            return
        copy_fn = getattr(self, "copy_to_clipboard", None)
        if callable(copy_fn):
            copy_fn(selection)
        external_ok = self._copy_external(selection)
        self._last_copied = selection
        if external_ok or callable(copy_fn):
            self.notify("Selection copied to clipboard", timeout=1.2)
        else:
            self.notify("Clipboard copy not available", severity="error")


if __name__ == "__main__":
    SelectionCopyApp().run()
