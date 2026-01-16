#!/usr/bin/env python3
"""Test script to compare OpenRouter VLM models: Nemotron vs Qwen."""

import os
import base64
import requests
import tempfile
import time
from openai import OpenAI

def test_model_performance(model_name, image_data, image_format="png"):
    """Test a specific model and measure its performance."""
    
    # Get the API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        print("Please set it with: export OPENROUTER_API_KEY='your-api-key'")
        return None, 0
    
    try:
        # Initialize the OpenAI client with OpenRouter
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        print(f"Calling model: {model_name}")
        
        start_time = time.time()
        
        # Call the model to generate a description
        response = client.chat.completions.create(
            model=model_name,
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
                                "url": f"data:image/{image_format};base64,{image_data}"
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
        
        return description, duration
        
    except Exception as e:
        print(f"ERROR with model {model_name}: {e}")
        return None, 0

def compare_models():
    """Compare the performance of different VLM models."""
    
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
        
        print("Starting model comparison...\n")
        
        # Define models to test
        models = [
            ("nvidia/nemotron-nano-12b-v2-vl:free", "Nemotron Nano"),
            ("qwen/qwen-2.5-vl-7b-instruct:free", "Qwen 2.5 VL")
        ]
        
        results = {}
        
        for model_id, model_name in models:
            print(f"Testing {model_name} ({model_id})...")
            description, duration = test_model_performance(model_id, image_data, "png")
            
            if description:
                results[model_name] = {
                    'description': description,
                    'duration': duration
                }
                print(f"  ✓ Completed in {duration:.2f} seconds")
                print(f"  Description preview: {description[:100]}...\n")
            else:
                print(f"  ✗ Failed\n")
        
        # Print comparison summary
        print("="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"  Time: {result['duration']:.2f} seconds")
            print(f"  Description: {result['description'][:150]}...")
            print()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"ERROR in comparison: {e}")
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
    print("OpenRouter VLM Model Comparison Test")
    print("===================================")
    print("Comparing: nvidia/nemotron-nano-12b-v2-vl:free vs qwen/qwen-2.5-vl-7b-instruct:free")
    print()
    
    success = compare_models()
    
    if success:
        print("✓ Model comparison completed successfully!")
        print("Check the summary above to see which model performs better.")
    else:
        print("✗ Model comparison failed!")
        print("Please check your API key and internet connection.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())