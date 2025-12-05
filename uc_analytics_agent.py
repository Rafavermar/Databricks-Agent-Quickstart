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
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

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
)

# Enable tracing in the deployed agent as well
mlflow.openai.autolog()

openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
uc_function_client = DatabricksFunctionClient()

builtin_tools = UCFunctionToolkit(
    function_names=["system.ai.python_exec"],
    client=uc_function_client,
).tools

for tool in builtin_tools:
    if "strict" in tool["function"]:
        del tool["function"]["strict"]


def call_tool(tool_name: str, parameters: dict) -> str:
    if tool_name != "system__ai__python_exec":
        msg = f"Unknown tool: {tool_name}"
        raise ValueError(msg)

    result = uc_function_client.execute_function(
        "system.ai.python_exec",
        parameters=parameters,
    )
    return result.value


def call_llm(prompt: str) -> Iterable[dict]:
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
        yield chunk.to_dict()


def run_agent(prompt: str) -> Iterable[dict]:
    last_chunk = None
    for chunk in output_to_responses_items_stream(call_llm(prompt)):
        last_chunk = chunk
        yield chunk.model_dump(exclude_none=True)

    if last_chunk is not None and last_chunk.item.get("type") == "function_call":
        tool_name = last_chunk.item["name"]
        tool_args = json.loads(last_chunk.item["arguments"])
        tool_result = call_tool(tool_name, tool_args)
        yield {
            "type": "response.output_item.done",
            "item": create_function_call_output_item(
                call_id=last_chunk.item["call_id"],
                output=tool_result,
            ),
        }


class TinyAnalyticsAgent(ResponsesAgent):
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Iterable[ResponsesAgentStreamEvent]:
        prompt = request.input[-1].content
        for chunk in run_agent(prompt):
            yield ResponsesAgentStreamEvent(**chunk)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs)


AGENT = TinyAnalyticsAgent()
mlflow.models.set_model(AGENT)
