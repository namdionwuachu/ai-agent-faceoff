# test_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Please set GOOGLE_API_KEY in your .env file")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

print("🔍 Checking available models...")
print("=" * 50)

available_models = []
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        available_models.append(model.name)
        print(f"✅ {model.name}")

print(f"\n📊 Total available models: {len(available_models)}")

if available_models:
    print("\n🧪 Testing the first available model...")
    try:
        model = genai.GenerativeModel(available_models[0])
        response = model.generate_content("Say 'Hello World' in a creative way")
        print(f"✅ Model works! Response: {response.text}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
else:
    print("❌ No available models found. Check your API key and billing.")
