import json
import textwrap
from typing import Iterable

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    create_function_call_output_item,
)

# ---- Configuration ----
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"  # LLM endpoint name

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a tiny analytics assistant.

    You can answer two kinds of questions:
    1. Pure reasoning questions that you can answer directly.
    2. Small data questions by calling the Python code interpreter tool.

    When you call the Python tool:
    - Write short, clean Python code.
    - Prefer base Python or pandas.
    - Print only the final answer, not explanations.

    For "store visits" questions:
    - Create a tiny in-memory table with columns:
      city (str), day (str), visits (int), revenue (float)
    - Invent 5-10 realistic rows and base your answer on that table.
    - After the tool call, briefly explain in natural language what you did.
    """
)  # System prompt to steer the agent

mlflow.openai.autolog()  # Enable tracing in the deployed agent

openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()  # OpenAI-compatible client
uc_function_client = DatabricksFunctionClient()  # UC function client

builtin_tools = UCFunctionToolkit(
    function_names=["system.ai.python_exec"],
    client=uc_function_client,
).tools  # Load built-in Python code interpreter tool

for tool in builtin_tools:
    tool["function"].pop("strict", None)


def call_tool(tool_name: str, parameters: dict) -> str:
    """
    Execute a Unity Catalog function tool and return its textual result.
    Only supports the built-in Python executor.
    """
    if tool_name != "system__ai__python_exec":
        msg = f"Unknown tool: {tool_name}"
        raise ValueError(msg)

    result = uc_function_client.execute_function(
        "system.ai.python_exec",
        parameters=parameters,
    )
    return result.value  # Return the text printed by the Python code


def call_llm(prompt: str) -> Iterable[dict]:
    """
    Call the Databricks foundation model with streaming enabled.
    Yields raw OpenAI-style chunks as dicts.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for chunk in openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=messages,
        tools=builtin_tools,
        stream=True,
    ):
        yield chunk.to_dict()  # Convert SDK object to plain dict

def _build_text_events(text: str, item_id: str):
    """Yield minimal Responses events for a plain text answer."""
    yield {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "delta": text,
    }
    yield {
        "type": "response.output_item.done",
        "item": {
            "id": item_id,
            "content": [{"text": text, "type": "output_text"}],
            "role": "assistant",
            "type": "message",
        },
    }


EXPLANATION_SYSTEM_PROMPT = (
    "You are a tiny analytics assistant. "
    "Answer in English. "
    "First sentence: direct answer (for example: "
    '"The city with the highest total revenue is X"). '
    "Optionally, add one short sentence explaining what you did. "
    "Do not print the full table unless the user explicitly asks for it."
)

def run_agent(user_prompt: str):
    """
    Simple agent:

    1) Call the LLM with tools enabled.
    2) If the LLM decides to call system.ai.python_exec, execute it.
    3) Call the LLM again to turn the tool output into a short answer.
    4) Yield Responses-style events for the final text.
    """

    # ---- Step 1: first LLM call with tools ----
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    first = openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=messages,
        tools=builtin_tools,
    )
    assistant_msg = first.choices[0].message
    completion_id = first.id
    tool_calls = getattr(assistant_msg, "tool_calls", None)

    # Case A: no tool used -> just return the model's answer
    if not tool_calls:
        final_text = assistant_msg.content or ""
        yield from _build_text_events(final_text, completion_id)
        return

    # Case B: the model decided to use system.ai.python_exec
    call = tool_calls[0]
    tool_output = call_tool(
        call.function.name,
        json.loads(call.function.arguments),
    )

    # ---- Step 2: second LLM call to explain the tool result ----
    followup_messages = [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        assistant_msg.to_dict(),  # contains the tool_call metadata
        {
            "role": "tool",
            "name": call.function.name,
            "tool_call_id": call.id,
            "content": tool_output,
        },
    ]
    second = openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=followup_messages,
    )
    final_msg = second.choices[0].message
    final_text = final_msg.content or ""
    completion2_id  = second.id

    yield from _build_text_events(final_text, completion2_id )


class TinyAnalyticsAgent(ResponsesAgent):
    """
    Minimal ResponsesAgent wrapper around `run_agent`.
    Implements predict_stream and predict for MLflow serving.
    """
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Iterable[ResponsesAgentStreamEvent]:
        prompt = request.input[-1].content  # Use last message as user prompt
        for chunk in run_agent(prompt):
            yield ResponsesAgentStreamEvent(**chunk)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs)


AGENT = TinyAnalyticsAgent()  # Instantiate the agent
mlflow.models.set_model(AGENT)  # Register the agent for MLflow logging/serving
