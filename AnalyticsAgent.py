# Databricks notebook source
# MAGIC %md
# MAGIC # Tiny Analytics Agent – end-to-end quickstart
# MAGIC
# MAGIC This notebook does four things:
# MAGIC
# MAGIC 1. Configures a simple **tool-calling agent** using Databricks foundation models.
# MAGIC 2. Uses the built-in Unity Catalog tool `system.ai.python_exec` so the agent can run Python code.
# MAGIC 3. Wraps the agent as an **MLflow ResponsesAgent** and logs it to Unity Catalog.
# MAGIC 4. Deploys the agent to a **Model Serving endpoint** you can connect to a Databricks App.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Optional: install packages
# MAGIC
# MAGIC On many recent Databricks runtimes these libraries are already available.
# MAGIC If the imports in the next cell fail, run this cell once and then restart the
# MAGIC Python process (`Detach & re-attach` the notebook or `dbutils.library.restartPython()`).

# COMMAND ----------

# Uncomment only if you actually need it
%pip install -U -qqqq mlflow databricks-openai databricks-agents
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and basic configuration

# COMMAND ----------

# MAGIC %sql
# MAGIC --USE CATALOG workspace;
# MAGIC --CREATE SCHEMA agent;

# COMMAND ----------

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
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksFunction
from pkg_resources import get_distribution

# ---- Configuration you will most likely touch ----

# 1. Foundation model endpoint to use as the LLM.
#    Change this to any chat-capable Databricks foundation model endpoint
#    available in your workspace.
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

# 2. Where to register the agent in Unity Catalog.
#    Use an existing catalog + schema that you own.
REGISTERED_MODEL_NAME = "workspace.agent.uc_analytics_agent"

# 3. Simple system prompt to steer the agent.
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

# Enable MLflow OpenAI autologging for traces inside the notebook and in serving
mlflow.openai.autolog()

# OpenAI-compatible client that talks to Databricks model serving endpoints
openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

# Unity Catalog function client to call built-in tools
uc_function_client = DatabricksFunctionClient()

# Load built-in Python code interpreter tool: system.ai.python_exec
builtin_tools = UCFunctionToolkit(
    function_names=["system.ai.python_exec"], client=uc_function_client
).tools

# Make the tool schema a bit more forgiving (remove "strict" if present)
for tool in builtin_tools:
    if "strict" in tool["function"]:
        del tool["function"]["strict"]


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Low-level helpers: call tool + call LLM
# MAGIC
# MAGIC These helpers are intentionally tiny so the control flow is easy to follow.

# COMMAND ----------

def call_tool(tool_name: str, parameters: dict) -> str:
    """
    Execute a Unity Catalog function tool and return its textual result.

    For this demo we only support the built-in Python executor:
    `system.ai.python_exec`.
    """
    if tool_name != "system__ai__python_exec":
        msg = f"Unknown tool: {tool_name}"
        raise ValueError(msg)

    # This calls the UC function system.ai.python_exec on your workspace.
    # `parameters` is a dict that comes directly from the LLM tool call.
    result = uc_function_client.execute_function(
        "system.ai.python_exec",
        parameters=parameters,
    )
    # .value contains the text printed by the Python code
    return result.value


def call_llm(prompt: str) -> Iterable[dict]:
    """
    Call the Databricks foundation model with streaming enabled and
    yield the raw OpenAI-style chunks as dicts.
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
        # Convert SDK object to plain dict so MLflow helpers can consume it
        yield chunk.to_dict()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `run_agent`: single-prompt agent loop
# MAGIC
# MAGIC * Calls the LLM with tool support.
# MAGIC * Converts LLM output into **Responses API** events.
# MAGIC * If the model decides to call a tool, we execute it and emit one more event.

# COMMAND ----------

def run_agent(prompt: str) -> Iterable[dict]:
    """
    Execute a single agent turn and yield Responses-compatible events (as dicts).

    This is the core "agent brain" we will later wrap into a ResponsesAgent
    so it can be logged and served with MLflow.
    """
    last_chunk = None

    # Stream model output and convert it to Responses events
    for chunk in output_to_responses_items_stream(call_llm(prompt)):
        last_chunk = chunk
        yield chunk.model_dump(exclude_none=True)

    # If the last item was a tool call, execute the tool
    if last_chunk is not None and last_chunk.item.get("type") == "function_call":
        tool_name = last_chunk.item["name"]
        tool_args = json.loads(last_chunk.item["arguments"])
        tool_result = call_tool(tool_name, tool_args)

        # Wrap the tool result in a Responses output item
        yield {
            "type": "response.output_item.done",
            "item": create_function_call_output_item(
                call_id=last_chunk.item["call_id"],
                output=tool_result,
            ),
        }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Quick notebook-only test of `run_agent`
# MAGIC
# MAGIC This is *not* using MLflow yet, just the raw helper.

# COMMAND ----------

for event in run_agent(
    "Create a tiny store visits table and tell me which city has "
    "the highest total revenue."
):
    print(event)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Wrap the logic as an MLflow `ResponsesAgent`
# MAGIC
# MAGIC This wrapper:
# MAGIC
# MAGIC * Implements `predict_stream()` → yields `ResponsesAgentStreamEvent`.
# MAGIC * Implements `predict()` → returns `ResponsesAgentResponse`.
# MAGIC * Will be the actual **model object** that we log with MLflow.

# COMMAND ----------

class TinyAnalyticsAgent(ResponsesAgent):
    """Minimal ResponsesAgent wrapper around `run_agent`."""

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Iterable[ResponsesAgentStreamEvent]:
        # For this demo we treat the last message as the user prompt.
        # (Full chat history support is easy to add later.)
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


# Quick sanity check of the wrapper
from mlflow.types.responses import ResponsesAgentRequest

AGENT = TinyAnalyticsAgent()

test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is 12 * 7, step by step?"}]
)

for e in AGENT.predict_stream(test_request):
    print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save the agent to a Python file for logging
# MAGIC
# MAGIC We now put **all** of the agent code into a single file and call
# MAGIC `mlflow.models.set_model(AGENT)` inside it.
# MAGIC
# MAGIC This file will be referenced when we log the model with MLflow.

# COMMAND ----------

# MAGIC %%writefile uc_analytics_agent.py
# MAGIC import json
# MAGIC import textwrap
# MAGIC from typing import Iterable
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
# MAGIC
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     create_function_call_output_item,
# MAGIC )
# MAGIC
# MAGIC # ---- Configuration ----
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC
# MAGIC SYSTEM_PROMPT = textwrap.dedent(
# MAGIC     """
# MAGIC     You are a tiny analytics assistant.
# MAGIC
# MAGIC     You can answer two kinds of questions:
# MAGIC     1. Pure reasoning questions that you can answer directly.
# MAGIC     2. Small data questions by calling the Python code interpreter tool.
# MAGIC
# MAGIC     When you call the Python tool:
# MAGIC     - Write short, clean Python code.
# MAGIC     - Prefer base Python or pandas.
# MAGIC     - Print only the final answer, not explanations.
# MAGIC
# MAGIC     For "store visits" questions:
# MAGIC     - Create a tiny in-memory table with columns:
# MAGIC       city (str), day (str), visits (int), revenue (float)
# MAGIC     - Invent 5-10 realistic rows and base your answer on that table.
# MAGIC     - After the tool call, briefly explain in natural language what you did.
# MAGIC     """
# MAGIC )
# MAGIC
# MAGIC # Enable tracing in the deployed agent as well
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC uc_function_client = DatabricksFunctionClient()
# MAGIC
# MAGIC builtin_tools = UCFunctionToolkit(
# MAGIC     function_names=["system.ai.python_exec"],
# MAGIC     client=uc_function_client,
# MAGIC ).tools
# MAGIC
# MAGIC for tool in builtin_tools:
# MAGIC     if "strict" in tool["function"]:
# MAGIC         del tool["function"]["strict"]
# MAGIC
# MAGIC
# MAGIC def call_tool(tool_name: str, parameters: dict) -> str:
# MAGIC     if tool_name != "system__ai__python_exec":
# MAGIC         msg = f"Unknown tool: {tool_name}"
# MAGIC         raise ValueError(msg)
# MAGIC
# MAGIC     result = uc_function_client.execute_function(
# MAGIC         "system.ai.python_exec",
# MAGIC         parameters=parameters,
# MAGIC     )
# MAGIC     return result.value
# MAGIC
# MAGIC
# MAGIC def call_llm(prompt: str) -> Iterable[dict]:
# MAGIC     messages = [
# MAGIC         {"role": "system", "content": SYSTEM_PROMPT},
# MAGIC         {"role": "user", "content": prompt},
# MAGIC     ]
# MAGIC     for chunk in openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=messages,
# MAGIC         tools=builtin_tools,
# MAGIC         stream=True,
# MAGIC     ):
# MAGIC         yield chunk.to_dict()
# MAGIC
# MAGIC
# MAGIC def run_agent(prompt: str) -> Iterable[dict]:
# MAGIC     last_chunk = None
# MAGIC     for chunk in output_to_responses_items_stream(call_llm(prompt)):
# MAGIC         last_chunk = chunk
# MAGIC         yield chunk.model_dump(exclude_none=True)
# MAGIC
# MAGIC     if last_chunk is not None and last_chunk.item.get("type") == "function_call":
# MAGIC         tool_name = last_chunk.item["name"]
# MAGIC         tool_args = json.loads(last_chunk.item["arguments"])
# MAGIC         tool_result = call_tool(tool_name, tool_args)
# MAGIC         yield {
# MAGIC             "type": "response.output_item.done",
# MAGIC             "item": create_function_call_output_item(
# MAGIC                 call_id=last_chunk.item["call_id"],
# MAGIC                 output=tool_result,
# MAGIC             ),
# MAGIC         }
# MAGIC
# MAGIC
# MAGIC class TinyAnalyticsAgent(ResponsesAgent):
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Iterable[ResponsesAgentStreamEvent]:
# MAGIC         prompt = request.input[-1].content
# MAGIC         for chunk in run_agent(prompt):
# MAGIC             yield ResponsesAgentStreamEvent(**chunk)
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs)
# MAGIC
# MAGIC
# MAGIC AGENT = TinyAnalyticsAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log and register the agent with MLflow (Unity Catalog)
# MAGIC
# MAGIC * `python_model="uc_analytics_agent.py"` points to the file we just wrote.
# MAGIC * `resources` tell Databricks which internal resources the agent needs.
# MAGIC * We register the model directly into Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksFunction(function_name="system.ai.python_exec"),
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name ="agent",
        python_model="uc_analytics_agent.py",
        extra_pip_requirements=[
            f"databricks-connect=="
            f"{get_distribution('databricks-connect').version}"
        ],
        resources=resources,
        registered_model_name=REGISTERED_MODEL_NAME,
    )

logged_agent_info

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Deploy the agent to a Model Serving endpoint
# MAGIC
# MAGIC We now create a **Mosaic AI Agent endpoint** using `databricks.agents.deploy`.
# MAGIC Once it's `READY`, you can:
# MAGIC
# MAGIC * Open it in **AI Playground** and chat with it.
# MAGIC * Add it as a **Model Serving resource** in a Databricks App.

# COMMAND ----------

from databricks import agents

deployment_info = agents.deploy(
    model_name=REGISTERED_MODEL_NAME,
    model_version=logged_agent_info.registered_model_version,
    scale_to_zero=True,
)

print("Deployment info:")
print(deployment_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Use this agent from a Databricks App (high-level steps)
# MAGIC
# MAGIC 1. Wait until the serving endpoint created above is in `READY` state  
# MAGIC    (Serving UI → Endpoints).
# MAGIC 2. In **Apps** → create or open an app.
# MAGIC 3. Add a **Model Serving** resource and select this endpoint.
# MAGIC 4. In your app code, call the endpoint using the standard chat/Responses
# MAGIC    interface (the App templates already handle this for you).
# MAGIC
# MAGIC For the PoC, keeping the agent this simple is enough to understand:
# MAGIC * how the agent calls tools (`system.ai.python_exec`),
# MAGIC * how MLflow wraps it as a `ResponsesAgent`,
# MAGIC * and how deployment → App wiring works end-to-end.
