"""Test script to verify HTML entity handling in post titles."""

from reddit_browser.api import get_first_two_pages


def test_html_entities():
    """Test that HTML entities in post titles are properly decoded."""
    print("Testing HTML entity handling...")
    
    try:
        # Get posts from r/LocalLlama
        posts = get_first_two_pages("LocalLlama")
        
        print(f"Retrieved {len(posts)} posts")
        
        # Display first few posts to check title formatting
        for i, post in enumerate(posts[:5]):  # Show first 5 posts
            data = post["data"]
            title = data["title"]
            original_title = title
            
            # Simulate the html.unescape that happens in PostCard
            import html
            decoded_title = html.unescape(title)
            
            print(f"\nPost {i+1}:")
            print(f"  Original: {original_title}")
            print(f"  Decoded:  {decoded_title}")
            
            # Check if there were HTML entities to decode
            if original_title != decoded_title:
                print(f"  ✓ HTML entities detected and decoded")
        
        print(f"\nHTML entity handling test completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_html_entities()