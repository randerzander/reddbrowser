"""Test script to verify the Reddit API functionality."""

from reddit_browser.api import get_first_two_pages


def test_api():
    """Test the Reddit API functionality."""
    print("Testing Reddit API connection...")
    
    try:
        # Get posts from r/LocalLlama
        posts = get_first_two_pages("LocalLlama")
        
        print(f"Successfully retrieved {len(posts)} posts from r/LocalLlama")
        
        # Display first few posts as a sample
        for i, post in enumerate(posts[:5]):  # Show first 5 posts
            data = post["data"]
            title = data["title"]
            author = data["author"]
            score = data["score"]
            num_comments = data["num_comments"]
            
            print(f"\nPost {i+1}:")
            print(f"  Title: {title}")
            print(f"  Author: {author}")
            print(f"  Score: {score}")
            print(f"  Comments: {num_comments}")
            
        print(f"\nTotal posts retrieved: {len(posts)}")
        print("API test completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_api()