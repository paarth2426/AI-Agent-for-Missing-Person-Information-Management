import ollama

response = ollama.chat(
    model="qwen3-14b-64k:latest",
    messages=[
        {
            "role": "user",
            "content": "Explain what an AI agent is in two sentences."
        }
    ]
)

print(response["message"]["content"])