# Reddit Browser

A terminal-based TUI for browsing Reddit built with Textual. This was entirely vibe coded to scratch an itch - no support, no maintenance, just vibes.

## Usage

```bash
python app.py [subreddit]
```

Defaults to r/LocalLlama if no subreddit specified.

## Prerequisites

Make sure you're running this in a real terminal environment (not in IDE terminals that don't properly handle input). The TUI requires a proper terminal to handle keyboard input.

## Installation

Install dependencies with:
```bash
pip install -r requirements.txt
```

For enhanced image viewing in terminal (optional):
```bash
pip install term-image
```

For AI features (optional):
```bash
pip install openai
export OPENROUTER_API_KEY="your-api-key-here"
```

## Controls

- `j`/`k`: Navigate posts
- `r`: Refresh
- `q`: Quit
- `Enter`: Open post
- Numbers: Jump to specific post
- `j`/`k` in comments: Navigate comment pages
- `↑`/`↓` in comments: Select comments
- `+`/`-` in comments: Expand/collapse comments
- `v` in comments: View image in GUI viewer
- `ESC` in comments: Return to post list

## Notes

This was vibe coded to satisfy a specific need. I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve or maintain it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.