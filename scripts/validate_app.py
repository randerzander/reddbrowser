"""Validation script to ensure the Reddit browser application is properly structured."""

import sys
import traceback

def validate_application():
    """Validate that the application modules can be imported and run."""
    print("Validating Reddit Browser Application...")
    
    try:
        # Test importing the API module
        from reddit_browser.api import RedditAPI, get_first_two_pages
        print("✓ API module imported successfully")
        
        # Test importing the main application
        from reddit_browser.app import RedditBrowserApp, PostCard, main
        print("✓ Main application module imported successfully")

        # Test that we can instantiate the app
        app = RedditBrowserApp("LocalLlama")
        print("✓ Application instance created successfully")

        # Test that we can access the key methods
        assert hasattr(app, 'compose')
        assert hasattr(app, 'on_mount')
        assert hasattr(app, 'load_posts')
        assert hasattr(app, 'update_grid')
        assert hasattr(app, 'action_refresh')
        print("✓ All required methods are available")
        
        # Test the API functionality again
        posts = get_first_two_pages("LocalLlama")
        assert len(posts) > 0, "Should have retrieved at least one post"
        print(f"✓ API test successful - retrieved {len(posts)} posts")
        
        print("\n✓ All validations passed! The Reddit Browser application is properly structured.")
        print("\nNote: The TUI may not run in all environments, but the core functionality is working.")
        print("To run the application, use: poetry run python -m reddit_browser.main [subreddit]")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False
    except AssertionError as e:
        print(f"✗ Assertion error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = validate_application()
    sys.exit(0 if success else 1)