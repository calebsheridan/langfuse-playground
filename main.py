import os
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context

import litellm
from litellm import completion

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
  langfuse_context.update_current_trace(name="COOLNAME", tags=["dev", "litellm"])
  models = ["gpt-3.5-turbo", "claude-3-haiku-20240307", "groq/llama3-8b-8192", "groq/llama-3.1-8b-instant", "groq/llama-3.1-70b-versatile"]

  print(f"[Loading prompts from storage...]")
  system_prompt = load_system_prompt_from_storage()
  user_prompt = load_user_prompt_from_storage()

  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]

  for model in models:
    trace_id = langfuse_context.get_current_trace_id()
    try:
      response = generation(trace_id=trace_id, model=model, messages=messages)
      message = response.choices[0].message.content
      print(message)
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
def generation(trace_id, model, messages):
  print(f"[Generating completion for {model} with trace_id {trace_id}]")

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
  )

  print(f"[Returning response...]")
  return response

if __name__ == "__main__":
    main()