#!/usr/bin/env python3
"""Test script to specifically test image captioning for the 'Latest upgrade...A100 40 GB' post."""

import os
import base64
import requests
import time
from openai import OpenAI

def find_a100_image_in_subreddit(subreddit="LocalLlama"):
    """Find the A100 image post in the subreddit."""
    print(f"Searching for 'A100' image post in r/{subreddit}...")
    
    # Use the Reddit API to get posts
    import httpx
    
    url = f"https://www.reddit.com/r/{subreddit}/.json"
    headers = {"User-Agent": "RedditBrowser/0.1.0"}
    
    try:
        with httpx.Client(headers=headers, timeout=10.0) as client:
            response = client.get(url, params={"limit": 50})
            response.raise_for_status()
            data = response.json()
            
            posts = data["data"]["children"]
            
            for post in posts:
                post_data = post["data"]
                title = post_data["title"]
                url = post_data["url"]
                
                # Look for posts with "A100" in the title
                if "A100" in title and "upgrade" in title.lower():
                    print(f"Found post: {title}")
                    print(f"URL: {url}")
                    
                    # Check if it's an image post
                    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg']
                    url_lower = url.lower()
                    
                    # Check for common image hosting domains
                    image_domains = [
                        'i.redd.it', 'i.imgur.com', 'imgur.com', 'flickr.com', 
                        'instagram.com', 'twitter.com', 'facebook.com',
                        'cdn.discordapp.com', 'media.discordapp.net'
                    ]
                    
                    is_image = False
                    for domain in image_domains:
                        if domain in url_lower:
                            # Special handling for imgur - check if it's not an album/gallery
                            if 'imgur.com' in domain and any(x in url_lower for x in ['/a/', '/gallery/', 'album']):
                                continue  # Skip albums/galleries
                            is_image = True
                            break
                    
                    # Check if URL ends with an image extension
                    for ext in image_extensions:
                        if url_lower.endswith(ext):
                            is_image = True
                            break
                    
                    if is_image:
                        print(f"This appears to be an image post!")
                        return url, title
                    else:
                        print(f"This is not an image post.")
            
            print("No A100 image post found.")
            # Show some sample titles to help with debugging
            print("\nSample titles from recent posts:")
            for i, post in enumerate(posts[:5]):
                print(f"  {i+1}. {post['data']['title']}")
            
            return None, None
            
    except Exception as e:
        print(f"Error searching for post: {e}")
        return None, None

def test_a100_image_captioning():
    """Test captioning specifically for the A100 image."""
    
    # Get the API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        print("Please set it with: export OPENROUTER_API_KEY='your-api-key'")
        return False
    
    # Find the A100 image post
    image_url, title = find_a100_image_in_subreddit()
    
    if not image_url:
        print("Could not find A100 image post, trying with a generic test image...")
        # Use a test image if we can't find the specific one
        image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
        title = "Test Image"
    
    try:
        print(f"\nDownloading image: {image_url}")
        
        # Add headers to avoid 403 errors
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        
        print(f"Downloaded {len(response.content)} bytes")
        
        # Encode the image to base64
        image_data = base64.b64encode(response.content).decode('utf-8')
        
        # Determine the image format based on URL
        from urllib.parse import urlparse
        parsed_url = urlparse(image_url)
        file_ext = os.path.splitext(parsed_url.path)[1].lower()
        if file_ext in ['.png']:
            mime_type = 'image/png'
        elif file_ext in ['.gif']:
            mime_type = 'image/gif'
        else:
            mime_type = 'image/jpeg'  # default
        
        print(f"Using MIME type: {mime_type}")
        
        # Initialize the OpenAI client with OpenRouter
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        print(f"\nCalling Qwen 2.5 VL model for image captioning...")
        print(f"Model: qwen/qwen-2.5-vl-7b-instruct:free")
        
        start_time = time.time()
        
        # Call the model to generate a description
        response = client.chat.completions.create(
            model="qwen/qwen-2.5-vl-7b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Describe this image in detail. Provide a comprehensive description of what you see in the image, focusing on technical aspects if it's hardware-related."
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
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get the description
        description = response.choices[0].message.content
        
        print("\n" + "="*60)
        print("IMAGE CAPTIONING RESULT")
        print("="*60)
        print(f"Post Title: {title}")
        print(f"Image URL: {image_url}")
        print(f"Processing Time: {duration:.2f} seconds")
        print(f"Description Length: {len(description)} characters")
        print("\nDESCRIPTION:")
        print(description)
        print("="*60)
        
        if duration > 10:  # If it took more than 10 seconds, it might be slow
            print(f"\n⚠️  WARNING: Processing took {duration:.2f} seconds, which is quite long.")
            print("This might indicate issues with the model or API endpoint.")
        else:
            print(f"\n✅ SUCCESS: Image captioning completed in {duration:.2f} seconds.")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to generate image description: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("A100 Image Captioning Test")
    print("==========================")
    print("Testing image captioning for 'Latest upgrade...A100 40 GB' post")
    print()
    
    success = test_a100_image_captioning()
    
    if success:
        print("\n✓ A100 image captioning test completed!")
    else:
        print("\n✗ A100 image captioning test failed!")
        print("The issue might be with the API key, network, or the specific image.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())