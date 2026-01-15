#!/usr/bin/env python3
"""Entry point for the Reddit Browser application."""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from reddit_browser.app import main

if __name__ == "__main__":
    main()
