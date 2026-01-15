"""Quick test to list available Gemini models."""
import os
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available models:")
try:
    for model in client.models.list():
        print(f"  - {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"    Methods: {model.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")

# Try to generate content
print("\nTesting gemini-1.5-flash:")
try:
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Say hello'
    )
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
