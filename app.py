import gradio as gr
import json
import google.generativeai as genai

# Securely load API key from external file to keep it hidden from the source code
def get_api_key():
    try:
        with open("key.txt", "r") as f:
            content = f.read().strip()
            if "=" in content:
                return content.split("=")[1].strip()
            return content
    except FileNotFoundError:
        return ""

api_key = get_api_key()
if api_key:
    genai.configure(api_key=api_key)

# Initialize Gemini Model
model = genai.GenerativeModel('gemini-3.5-flash-lite')

def load_data():
    try:
        with open("data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def query_agent(user_query):
    if not api_key:
        return "⚠️ Error: API key not configured properly. Make sure key.txt exists."
    
    data = load_data()
    
    if not data:
        return "⚠️ Error: No missing persons data found in data.json."
    
    context = f"""
    You are an AI assistant helping to track and locate missing persons based on a database. 
    Here is the current database in JSON format:
    {json.dumps(data, indent=2)}
    
    Answer the user's query based ONLY on the data provided above. 
    If the answer cannot be found in the data, state that clearly.
    
    User Query: {user_query}
    """
    
    try:
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

# Gradio Frontend UI
with gr.Blocks() as demo:
    gr.Markdown("# 🕵️‍♂️ Missing Persons AI Agent")
    gr.Markdown("Search and query the missing persons database using natural language.")
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Ask a Question", 
                placeholder="e.g. List all missing persons from Mumbai.",
                lines=3
            )
            submit_btn = gr.Button("Search", variant="primary")
            
        with gr.Column(scale=2):
            output_display = gr.Textbox(label="AI Response", lines=10, interactive=False)
            
    submit_btn.click(fn=query_agent, inputs=query_input, outputs=output_display)
    query_input.submit(fn=query_agent, inputs=query_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
