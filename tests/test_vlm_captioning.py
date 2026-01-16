#!/usr/bin/env python3
"""Test script to verify OpenRouter VLM captioning functionality."""

import os
import base64
import requests
import tempfile
from openai import OpenAI

def test_vlm_captioning():
    """Test the VLM captioning functionality with a sample image."""
    
    # Get the API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        print("Please set it with: export OPENROUTER_API_KEY='your-api-key'")
        return False
    
    try:
        # Download a test image
        print("Downloading test image...")
        image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"

        # Add headers to avoid 403 errors
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(response.content)
            temp_path = tmp_file.name
        
        print(f"Downloaded test image to: {temp_path}")
        
        # Read the image file and encode it to base64
        with open(temp_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        print("Initializing OpenAI client with OpenRouter...")
        
        # Initialize the OpenAI client with OpenRouter
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        print("Calling the VLM model for image description...")
        
        # Call the model to generate a description
        response = client.chat.completions.create(
            model="nvidia/nemotron-nano-12b-v2-vl:free",
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
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Get the description
        description = response.choices[0].message.content
        
        print("\n" + "="*50)
        print("IMAGE DESCRIPTION GENERATED SUCCESSFULLY!")
        print("="*50)
        print(description)
        print("="*50)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to generate image description: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return False

def main():
    print("OpenRouter VLM Captioning Test")
    print("==============================")
    
    success = test_vlm_captioning()
    
    if success:
        print("\n✓ VLM captioning test PASSED!")
        print("The image captioning functionality is working correctly.")
    else:
        print("\n✗ VLM captioning test FAILED!")
        print("Please check your API key and internet connection.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())