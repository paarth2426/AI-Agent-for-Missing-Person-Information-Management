import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

print("API key loaded:", bool(api_key))

client = TavilyClient(api_key=api_key)

response = client.search(
    query="artificial intelligence agents",
    max_results=3
)

for result in response["results"]:
    print("\nTitle:", result["title"])
    print("URL:", result["url"])