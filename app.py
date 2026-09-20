import json
import os
import ollama
import gradio as gr
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

MODEL = "qwen3-14b-64k:latest"

with open("data.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

activity_log = []


def log(message):
    activity_log.append(message)


def search_cases(location=None, min_age=None, max_age=None, gender=None):
    log("🔎 Searching synthetic case database...")

    results = []

    for person in cases:
        if location and location.lower() not in person["last_seen"].lower():
            continue

        if min_age is not None and person["age"] < min_age:
            continue

        if max_age is not None and person["age"] > max_age:
            continue

        if gender and gender.lower() != person["gender"].lower():
            continue

        results.append(person)

    log(f"✓ Database search returned {len(results)} record(s).")

    return results


def get_case(case_id):
    log(f"📋 Retrieving case {case_id}...")

    for person in cases:
        if person["id"].lower() == case_id.lower():
            log(f"✓ Case {case_id} retrieved.")
            return person

    log(f"⚠️ Case {case_id} was not found.")
    return None


def web_search(query):
    log(f"🌐 Searching the web: {query}")

    try:
        response = tavily.search(
            query=query,
            max_results=5
        )

        results = []

        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", "")
            })

        log(f"✓ Tavily returned {len(results)} web result(s).")

        return results

    except Exception as e:
        log(f"❌ Tavily error: {str(e)}")
        return []


def investigate(query):
    global activity_log

    activity_log = []

    log("🤖 Agent started.")
    log(f"📝 Query: {query}")

    system_prompt = """
You are an AI investigation assistant operating on a synthetic
missing-person demonstration database.

Your job is to investigate user queries using the available tools.

You have three tools:

1. search_cases
   Search the synthetic missing-person database using location,
   age range, and gender.

2. get_case
   Retrieve the complete details of a specific case ID.

3. web_search
   Search the public web using Tavily when external information
   would be useful.

You should decide which tools are necessary to answer the user's query.

Important rules:

- The database contains completely fictional demonstration records.
- Never claim that a person has been positively identified.
- Never claim that two people are the same person.
- Treat database records as synthetic demonstration data.
- Clearly distinguish database information from web information.
- Do not invent facts.
- If web results are unrelated or insufficient, say so.
- Give the user a concise investigation report.
"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_cases",
                "description": "Search the synthetic missing-person database by location, age range, and gender.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location where the person was last seen."
                        },
                        "min_age": {
                            "type": "integer",
                            "description": "Minimum age."
                        },
                        "max_age": {
                            "type": "integer",
                            "description": "Maximum age."
                        },
                        "gender": {
                            "type": "string",
                            "description": "Gender filter."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_case",
                "description": "Retrieve complete details for a case using its case ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "case_id": {
                            "type": "string",
                            "description": "Case ID such as MP001."
                        }
                    },
                    "required": ["case_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web for information relevant to an investigation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Web search query."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]

    for _ in range(6):

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        message = response["message"]

        messages.append(message)

        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            log("🤖 Agent completed its investigation.")

            report = message.get("content", "")

            activity = "\n".join(activity_log)

            return activity, report

        for call in tool_calls:

            function_name = call["function"]["name"]
            arguments = call["function"].get("arguments", {})

            if function_name == "search_cases":
                result = search_cases(
                    location=arguments.get("location"),
                    min_age=arguments.get("min_age"),
                    max_age=arguments.get("max_age"),
                    gender=arguments.get("gender")
                )

            elif function_name == "get_case":
                result = get_case(
                    arguments.get("case_id")
                )

            elif function_name == "web_search":
                result = web_search(
                    arguments.get("query")
                )

            else:
                result = {"error": "Unknown tool"}

            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False)
            })

    log("⚠️ Agent reached the maximum investigation steps.")

    return "\n".join(activity_log), "The investigation could not be completed within the allowed tool steps."


def run_investigation(query):
    if not query.strip():
        return "Please enter an investigation query.", ""

    activity, report = investigate(query)

    return activity, report


with gr.Blocks(
    title="Missing Person Investigation AI",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🔎 Missing Person Investigation AI

        ### AI-powered investigation assistant

        Search a synthetic case database and optionally investigate
        related information using web search.

        **⚠️ Demonstration only — all case records are fictional.**
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            query = gr.Textbox(
                label="Investigation Query",
                placeholder="Example: Find missing people aged 18-25 last seen in Mumbai and investigate related information.",
                lines=4
            )

            investigate_button = gr.Button(
                "🔎 Investigate",
                variant="primary"
            )

        with gr.Column(scale=1):

            gr.Markdown(
                """
                ### Example Queries

                **Database search**

                Find missing people aged 18-25 last seen in Mumbai.

                **Case investigation**

                Investigate case MP001.

                **Combined investigation**

                Find young people missing from Mumbai and search the web for potentially relevant information.
                """
            )

    gr.Markdown("## 🤖 Agent Activity")

    activity_output = gr.Markdown(
        value="Waiting for investigation..."
    )

    gr.Markdown("## 📋 Investigation Report")

    report_output = gr.Markdown(
        value="Your investigation report will appear here."
    )

    investigate_button.click(
        fn=run_investigation,
        inputs=query,
        outputs=[activity_output, report_output]
    )

    gr.Markdown(
        """
        ---
        **Demo Notice:** This application uses synthetic missing-person
        records. Web search results are external information and should
        not be interpreted as verified identification or evidence.
        """
    )


if __name__ == "__main__":
    demo.launch()