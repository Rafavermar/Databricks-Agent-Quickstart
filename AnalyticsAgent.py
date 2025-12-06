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
# MAGIC --Run this cell once at the begining
# MAGIC
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
    tool["function"].pop("strict", None)


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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `run_agent`: single-prompt agent loop
# MAGIC
# MAGIC * Calls the LLM with tool support.
# MAGIC * Converts LLM output into **Responses API** events.
# MAGIC * If the model decides to call a tool, we execute it and emit one more event.

# COMMAND ----------

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

    # Step 1: first LLM call with tools
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

    # Step 2: second LLM call to explain the tool result
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
# MAGIC * Implements `predict_stream()` -> yields `ResponsesAgentStreamEvent`.
# MAGIC * Implements `predict()` -> returns `ResponsesAgentResponse`.
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
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"  # LLM endpoint name
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
# MAGIC )  # System prompt to steer the agent
# MAGIC
# MAGIC mlflow.openai.autolog()  # Enable tracing in the deployed agent
# MAGIC
# MAGIC openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()  # OpenAI-compatible client
# MAGIC uc_function_client = DatabricksFunctionClient()  # UC function client
# MAGIC
# MAGIC builtin_tools = UCFunctionToolkit(
# MAGIC     function_names=["system.ai.python_exec"],
# MAGIC     client=uc_function_client,
# MAGIC ).tools  # Load built-in Python code interpreter tool
# MAGIC
# MAGIC for tool in builtin_tools:
# MAGIC     tool["function"].pop("strict", None)
# MAGIC
# MAGIC
# MAGIC def call_tool(tool_name: str, parameters: dict) -> str:
# MAGIC     """
# MAGIC     Execute a Unity Catalog function tool and return its textual result.
# MAGIC     Only supports the built-in Python executor.
# MAGIC     """
# MAGIC     if tool_name != "system__ai__python_exec":
# MAGIC         msg = f"Unknown tool: {tool_name}"
# MAGIC         raise ValueError(msg)
# MAGIC
# MAGIC     result = uc_function_client.execute_function(
# MAGIC         "system.ai.python_exec",
# MAGIC         parameters=parameters,
# MAGIC     )
# MAGIC     return result.value  # Return the text printed by the Python code
# MAGIC
# MAGIC
# MAGIC def call_llm(prompt: str) -> Iterable[dict]:
# MAGIC     """
# MAGIC     Call the Databricks foundation model with streaming enabled.
# MAGIC     Yields raw OpenAI-style chunks as dicts.
# MAGIC     """
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
# MAGIC         yield chunk.to_dict()  # Convert SDK object to plain dict
# MAGIC
# MAGIC def _build_text_events(text: str, item_id: str):
# MAGIC     """Yield minimal Responses events for a plain text answer."""
# MAGIC     yield {
# MAGIC         "type": "response.output_text.delta",
# MAGIC         "item_id": item_id,
# MAGIC         "delta": text,
# MAGIC     }
# MAGIC     yield {
# MAGIC         "type": "response.output_item.done",
# MAGIC         "item": {
# MAGIC             "id": item_id,
# MAGIC             "content": [{"text": text, "type": "output_text"}],
# MAGIC             "role": "assistant",
# MAGIC             "type": "message",
# MAGIC         },
# MAGIC     }
# MAGIC
# MAGIC
# MAGIC EXPLANATION_SYSTEM_PROMPT = (
# MAGIC     "You are a tiny analytics assistant. "
# MAGIC     "Answer in English. "
# MAGIC     "First sentence: direct answer (for example: "
# MAGIC     '"The city with the highest total revenue is X"). '
# MAGIC     "Optionally, add one short sentence explaining what you did. "
# MAGIC     "Do not print the full table unless the user explicitly asks for it."
# MAGIC )
# MAGIC
# MAGIC def run_agent(user_prompt: str):
# MAGIC     """
# MAGIC     Simple agent:
# MAGIC
# MAGIC     1) Call the LLM with tools enabled.
# MAGIC     2) If the LLM decides to call system.ai.python_exec, execute it.
# MAGIC     3) Call the LLM again to turn the tool output into a short answer.
# MAGIC     4) Yield Responses-style events for the final text.
# MAGIC     """
# MAGIC
# MAGIC     # Step 1: first LLM call with tools
# MAGIC     messages = [
# MAGIC         {"role": "user", "content": user_prompt},
# MAGIC     ]
# MAGIC     first = openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=messages,
# MAGIC         tools=builtin_tools,
# MAGIC     )
# MAGIC     assistant_msg = first.choices[0].message
# MAGIC     completion_id = first.id
# MAGIC     tool_calls = getattr(assistant_msg, "tool_calls", None)
# MAGIC
# MAGIC     # Case A: no tool used -> just return the model's answer
# MAGIC     if not tool_calls:
# MAGIC         final_text = assistant_msg.content or ""
# MAGIC         yield from _build_text_events(final_text, completion_id)
# MAGIC         return
# MAGIC
# MAGIC     # Case B: the model decided to use system.ai.python_exec
# MAGIC     call = tool_calls[0]
# MAGIC     tool_output = call_tool(
# MAGIC         call.function.name,
# MAGIC         json.loads(call.function.arguments),
# MAGIC     )
# MAGIC
# MAGIC     # Step 2: second LLM call to explain the tool result
# MAGIC     followup_messages = [
# MAGIC         {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
# MAGIC         {"role": "user", "content": user_prompt},
# MAGIC         assistant_msg.to_dict(),  # contains the tool_call metadata
# MAGIC         {
# MAGIC             "role": "tool",
# MAGIC             "name": call.function.name,
# MAGIC             "tool_call_id": call.id,
# MAGIC             "content": tool_output,
# MAGIC         },
# MAGIC     ]
# MAGIC     second = openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=followup_messages,
# MAGIC     )
# MAGIC     final_msg = second.choices[0].message
# MAGIC     final_text = final_msg.content or ""
# MAGIC     completion2_id  = second.id
# MAGIC
# MAGIC     yield from _build_text_events(final_text, completion2_id )
# MAGIC
# MAGIC
# MAGIC class TinyAnalyticsAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Minimal ResponsesAgent wrapper around `run_agent`.
# MAGIC     Implements predict_stream and predict for MLflow serving.
# MAGIC     """
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Iterable[ResponsesAgentStreamEvent]:
# MAGIC         prompt = request.input[-1].content  # Use last message as user prompt
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
# MAGIC AGENT = TinyAnalyticsAgent()  # Instantiate the agent
# MAGIC mlflow.models.set_model(AGENT)  # Register the agent for MLflow logging/serving

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log and register the agent with MLflow (Unity Catalog)
# MAGIC
# MAGIC * `python_model="uc_analytics_agent.py"` points to the file we just wrote.
# MAGIC * `resources` tell Databricks which internal resources the agent needs.
# MAGIC * We register the model directly into Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")  # Set registry to Unity Catalog

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),  # Reference LLM endpoint
    DatabricksFunction(function_name="system.ai.python_exec"),   # Reference Python exec tool
]

with mlflow.start_run():  # Start MLflow run for logging
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",  # Name for the logged model
        python_model="uc_analytics_agent.py",  # Python file containing agent code
        extra_pip_requirements=[
            f"databricks-connect=="
            f"{get_distribution('databricks-connect').version}"
        ],  # Ensure correct databricks-connect version
        resources=resources,  # Attach serving endpoint and function as resources
        registered_model_name=REGISTERED_MODEL_NAME,  # Register model in UC
    )

logged_agent_info  # Display logged model info

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
