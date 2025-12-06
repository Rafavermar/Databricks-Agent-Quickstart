# Tiny Analytics Agent on Databricks (Free Edition)

A very small **analytics agent** built on Databricks:

- Uses a **foundation model endpoint** (LLM).
- Can call one **tool**: `system.ai.python_exec` (Unity Catalog function).
- Answers simple analytics questions by:
  - asking the LLM to write Python,
  - running that Python,
  - and turning the result into a short natural-language answer.
- Logged with **MLflow**, deployed as an **Agent serving endpoint**, and tested via **Review App**.

This was my **first real immersion in Databricks agents**, following the official docs and a community walkthrough, then adapting everything to **Databricks Free Edition**.

---

## 0. Repository contents

- A notebook that:
  - defines a tiny `run_agent()` loop,
  - wraps it in a `TinyAnalyticsAgent` (a `ResponsesAgent`),
  - logs it with MLflow and registers it in Unity Catalog,
  - deploys it as an **Agent endpoint**.
- A small text overview: `tiny_analytics_agent_explanation.txt`  
  (mental model of the whole flow).

---

## 1. Scenario: tiny store visits analytics

We keep the use case intentionally simple.

### 1.1 Analytics prompt

```text
Create a tiny store visits table and tell me which city has the highest total revenue.
```

The agent should:

- Let the LLM invent a small dataset.
- Use `system.ai.python_exec` to:
  - build a Pandas DataFrame,
  - group and aggregate.
- Answer which city has the highest revenue, and optionally show a small table.

The agent is stateless: every request is handled independently.

### 1.2 Table prompt

```text
Create a tiny store visits table with columns city, day, visits, revenue.
Add the table visualization to your answer.
```

---

## 2. Architecture (high-level)

**User → Agent endpoint**

The client (Review App, future apps, curl, etc.) sends a prompt to the Agent endpoint using the Requests/Responses API.

**Agent endpoint → `TinyAnalyticsAgent`**

The Agent endpoint loads the MLflow model `TinyAnalyticsAgent` and forwards the request.

**`TinyAnalyticsAgent` → LLM + tool**

1. First call: ask the LLM (with tools enabled) how to solve the prompt.
2. If the model decides to call the tool, execute `system.ai.python_exec` with the generated Python code.
3. Optional second call: ask the LLM to turn the tool output into a short, user-friendly answer.
4. Convert the final text into Responses events and return.

---

## 3. Components

### 3.1 Foundation model endpoint

Any Databricks Model Serving endpoint that is enabled in your workspace, for example:

```text
databricks-meta-llama-3-3-70b-instruct
```

In Free Edition, some models may be visible but return a rate limit of 0.  
If you see `PERMISSION_DENIED: endpoint is temporarily disabled due to a rate limit of 0`, switch to another model.

The notebook uses the OpenAI-compatible client from `WorkspaceClient` to call this endpoint.

### 3.2 Tool: `system.ai.python_exec`

Unity Catalog function that executes Python in a controlled environment.

Expose it as a single tool to the LLM. When the LLM wants to run Python, it emits a tool call with:

- `name = "system__ai__python_exec"`
- JSON arguments containing `"code": "...python code..."`.

This is forwarded to:

```python
DatabricksFunctionClient.execute_function("system.ai.python_exec", parameters)
```

and capture the textual output.

### 3.3 Helper functions

**`call_tool(tool_name, parameters)`**

- Validates the tool name.
- Calls the UC function.
- Returns the text output.

**`_build_text_events(text, item_id)`**

Wraps a plain text string into the minimal Responses event schema:

- one `response.output_text.delta`,
- one `response.output_item.done`.

---

## 4. `run_agent()` - the core loop

Single-prompt, stateless design:

1. **First LLM call (with tools)**  
   - Messages: `[{ "role": "user", "content": prompt }]`  
   - Tools: the definition of `system.ai.python_exec`.

   - If the model does **not** call any tool:
     - Take its message content and return it as Responses events.

   - If there **is** a tool call:
     - Parse the JSON arguments to extract the Python code.
     - Call `call_tool()` to execute `system.ai.python_exec`.
     - Get `tool_output` (plain text).

2. **Optional second LLM call (without tools)**  
   Build messages with:

   - a small system prompt:  
     *"short answer, first sentence gives the result, optional one-line explanation"*
   - the original user message,
   - the tool result (as if the assistant had run the code).

   Then:

   - Call the LLM again, without tools.
   - Extract the final answer text.

3. **Return Responses events**  
   Use `_build_text_events()` to convert the final text into:

   - a delta event,
   - a final output item.

This function is the “brain” of the agent inside the notebook.

---

## 5. `TinyAnalyticsAgent` - wrapping for MLflow

We wrap `run_agent()` in a `ResponsesAgent` subclass:

```python
class TinyAnalyticsAgent(ResponsesAgent):
    def predict_stream(self, request: ResponsesAgentRequest, **_):
        # Stateless: only use the latest user message
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
```

Then, write it to a Python file and register it with MLflow:

```python
AGENT = TinyAnalyticsAgent()
mlflow.models.set_model(AGENT)
```

This is what MLflow logs and what the Agent endpoint uses at runtime.

---

## 6. Logging and deployment

### 6.1 Log and register with MLflow

Set the registry URI to Unity Catalog:

```python
mlflow.set_registry_uri("databricks-uc")
```

Log the model:

```python
logged_agent_info = mlflow.pyfunc.log_model(
    artifact_path="agent",
    python_model="uc_analytics_agent.py",
    resources=[
        DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
        DatabricksFunction(function_name="system.ai.python_exec"),
    ],
    registered_model_name=REGISTERED_MODEL_NAME,  # e.g. "catalog.schema.uc_analytics_agent"
)
```

Result: a registered model version for the agent.

### 6.2 Deploy as an Agent serving endpoint

Use the Agents API:

```python
from databricks import agents

deployment_info = agents.deploy(
    model_name=REGISTERED_MODEL_NAME,
    model_version=logged_agent_info.registered_model_version,
    scale_to_zero=True,  # optional
)
```

Databricks creates an Agent endpoint that speaks the Requests/Responses protocol.

---

## 7. Testing with Review App

1. Open **Review App** in the Databricks UI.
2. Choose the Agent endpoint (e.g. `agents_workspace-agent-uc_analytics_agent`).
3. Try the demo prompts:

**Analytics**

```text
Create a tiny store visits table and tell me which city has the highest total revenue.
```

**Table**

```text
Create a tiny store visits table with columns city, day, visits, revenue.
Add the table visualization to your answer.
```

In **Details & Timeline** you should see:

- LLM tool call to `system.ai.python_exec`,
- Python code with Pandas,
- tool output,
- final assistant answer.

---

## 8. Known limitations & TODOs

Summary of what this PoC does **not** do (yet).

### 8.1 Stateless agent

- Only the last user message is used.
- Follow-ups like *“Show me the table you used for that calculation”* will not behave as humans expect.

### 8.2 Prompt trade-offs

- Combining *“show the table”* and *“tell me the winning city”* in a single prompt is not perfectly reliable.
- In practice, it is safer to use separate prompts:
  - one for the calculation,
  - one for the table.

### 8.3 Chatbot App template incompatibility

- The basic Databricks Chatbot App template supports only chat-completions endpoints.
- Our Agent endpoint uses the Responses protocol, so that template shows **Unsupported Endpoint Type**.
- A richer app (e.g. the official Next.js sample) or a small custom app is needed to build a UI directly on top of this agent.

### 8.4 Free Edition constraints

- Some foundation models are visible but have a rate limit of 0 and cannot be used.
- You may need to try a few endpoints and pick one that returns normal completions.

These are good candidates for future iterations.
