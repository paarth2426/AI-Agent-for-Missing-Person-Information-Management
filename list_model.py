import google.generativeai as genai

def get_api_key():
    try:
        with open("key.txt", "r") as f:
            content = f.read().strip()
            if "=" in content:
                return content.split("=")[1].strip()
            return content
    except FileNotFoundError:
        return ""

genai.configure(api_key=get_api_key())
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
