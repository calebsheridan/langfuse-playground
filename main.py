import os
import json
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context

import litellm
from litellm import completion

from tools import get_current_weather, calculate

load_dotenv()

# set callbacks for langfuse
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# Here we check for API keys but do not fully validate providers.
# This immediately throw and is non-blocking where fully validation blocks.

openai_apy_key = os.getenv("OPENAI_API_KEY")
if not openai_apy_key:
    raise Exception("OPENAI_API_KEY environment variable is not set.")
# print(litellm.validate_environment("openai/gpt-3.5-turbo"))

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise Exception("GROQ_API_KEY environment variable is not set.")
# print(litellm.validate_environment("anthropic/claude-3-haiku-20240307"))

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise Exception("ANTHROPIC_API_KEY environment variable is not set.")
# print(litellm.validate_environment("groq/llama3-8b-8192"))

@observe()
def main():
  langfuse_context.update_current_trace(name="tool testing", tags=["dev"])
  models = [
    # "gpt-3.5-turbo", 
    # "claude-3-haiku-20240307", 
    "groq/llama3-groq-8b-8192-tool-use-preview",
    # "groq/llama-3.1-8b-instant",
    # "groq/llama3-8b-8192",
    # "groq/llama-3.1-70b-versatile",
  ]

  print(f"[Loading prompts from storage...]")
  system_prompt = load_system_prompt_from_storage()
  user_prompt = load_user_prompt_from_storage()

  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]

  tools = load_tools_from_storage()

  for model in models:
    trace_id = langfuse_context.get_current_trace_id()
    try:
      response = generation(trace_id=trace_id, model=model, messages=messages, tools=tools)
      message = response.choices[0].message.content
      print(f"{model}> {message}")
    except Exception as e:
      print(f"Error with generation for trace_id {trace_id}: {e}")

  # Waits for all generations to complete
  langfuse_context.flush()

@observe()
def load_system_prompt_from_storage():
  with open("./prompts/system.txt", "r") as file:
    system_prompt = file.read()

  return system_prompt

@observe()
def load_user_prompt_from_storage():
  with open("./prompts/user.txt", "r") as file:
    user_prompt = file.read()

  return user_prompt

@observe()
def load_tools_from_storage():
  with open("./tools/tools.json", "r") as file:
    tools = file.read()
  
  return json.loads(tools)

@observe()
def handle_tool_calls(response_message):
  tool_responses = []
  tool_calls = response_message.tool_calls
  print(f"[Handling tool calls ({len(tool_calls)})...]")

  available_functions = {
      "get_current_weather": get_current_weather.get_current_weather,
      "calculate": calculate.calculate
  }

  for tool_call in tool_calls:
    langfuse_context.update_current_trace(tags=["tool-call"])
    
    function_name = tool_call.function.name
    print(f"[Tool call]\n{function_name}")
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)

    print(f"Function args: {function_args}")
    function_response = function_to_call(**function_args)

    print(f"Function response: {function_response}")
    tool_responses.append(
      {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "name": tool_call.function.name,
        "content": function_response,
      }
    )
  return tool_responses

@observe()
def generation(trace_id, model, messages, tools=None, tool_choice="auto"):
  print(f"\n\n[Generating completion for {model} with trace_id {trace_id}]")
  
  toolArgs = {}
  if (tools != None):
    print(f"[Tools provided: {len(tools)} | Tool choice: {tool_choice}]")
    langfuse_context.update_current_trace(tags=["tool-use"])
    toolArgs = {
      "tools": tools,
      "tool_choice": tool_choice
    }

  print(f"[Fetching completion...]")
  response = completion(
    model=model,
    temperature=1,
    max_tokens=100,
    messages=messages,
    metadata={
        "generation_name": model,
        "trace_id": trace_id,
    },
    **toolArgs,
  )
  response_message = response.choices[0].message
  tool_calls = response_message.tool_calls

  if tool_calls:
    tool_responses = handle_tool_calls(response_message)

    # modify response message for certain models
    if "groq/" in model:
      response_message.function_call = tool_calls[0].function
      response_message.tool_calls = []

    messages.append(response_message)  # extend conversation with assistant's reply
    messages.extend(tool_responses) # extend conversation with tool responses
    
    # get a new response from the model where it can see the tool responses
    second_response = generation(trace_id=trace_id, model=model, messages=messages, tools=tools, tool_choice="none") 
    return second_response

  print(f"[Returning response...]")
  return response

if __name__ == "__main__":
    main()