import os
import json

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI(
    title="AI Study Agent",
    description="Education-focused AI Agent",
    version="1.0.0"
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================
# REQUEST MODEL
# =========================

class StudentRequest(BaseModel):
    message: str


# =========================
# STUDY PLAN TOOL
# =========================

def study_plan_tool(topic):
    return f"Create a 7-day study plan for {topic}."


# =========================
# TOOL DEFINITION
# =========================

tools = [
    {
        "type": "function",
        "function": {
            "name": "study_plan_tool",
            "description": "Creates a 7-day study plan for an educational topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The educational topic the student wants to study."
                    }
                },
                "required": ["topic"]
            }
        }
    }
]


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "AI Study Agent is running"
    }


# =========================
# AGENT ENDPOINT
# =========================

@app.post("/agent")
def run_agent(request: StudentRequest):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an education AI agent. "
                "Use the study plan tool when appropriate."
            )
        },
        {
            "role": "user",
            "content": request.message
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # -------------------------
    # TOOL CALL
    # -------------------------

    if message.tool_calls:

        tool_call = message.tool_calls[0]

        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "study_plan_tool":

            topic = arguments["topic"]

            tool_result = study_plan_tool(topic)

            messages.append(message)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                }
            )

            final_response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages
            )

            return {
                "agent_response": final_response.choices[0].message.content,
                "tool_used": function_name,
                "topic": topic
            }

    # -------------------------
    # NORMAL RESPONSE
    # -------------------------

    return {
        "agent_response": message.content,
        "tool_used": None
    }
