import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import time
import json

GPT_MODEL = "gpt-4-0613"

# function
def move_to_object(name: str):
    print("move to object: " + name)

def set_speed(speed: float):
    print("set_speed: " + speed)


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages: dict,
    functions: list = None,
    function_call: dict = None,
    model: str = GPT_MODEL
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def execute_function_call(assistant_message: dict):
    try:
        parsed_output = json.loads(
            assistant_message["function_call"]["arguments"]
        )
        arguments = ", ".join(str(arg) for arg in parsed_output.values())
        method = f"{assistant_message['function_call']['name']}('{arguments}')"
        eval(method)
    except Exception as e:
        print(e)

def execute_function_call_list(function_requests: list):
    for assistant_message in function_requests:
        execute_function_call(assistant_message)


def main():
    functions = [
        {
            "name": "move_to_object",
            "description": "Move to target position",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "target object name"
                    }
                },
                "required": ["name"]
            },
        },
        {
            "name": "set_speed",
            "description": "Set move speed",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "integer",
                        "description": "The number of speed",
                    }
                },
                "required": ["speed"]
            },
        },
    ]

    messages = []
    messages.append(
        {
            "role": "system",
            "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "Move to the kitchen via the table."
        }
    )
    chat_response = chat_completion_request(
        messages, functions=functions
    )
    assistant_message = chat_response.json()["choices"][0]["message"]
    messages.append(assistant_message)
    print(assistant_message)

    while True:
        function_requests = []
        user_content = input("User> ")
        if user_content in ["!quit", "!exit"]: break

        messages.append(
            {
                "role": "user",
                "content": user_content
            }
        )
        # execute chat completion
        while True:
            response = chat_completion_request(
                messages, functions=functions
            )
            full_message = response.json()["choices"][0]
            assistant_message = full_message["message"]
            messages.append(assistant_message)

            if full_message["finish_reason"] == "function_call":
                print(f"Function generation requested, calling function")
                function_requests.append(assistant_message)
            else:
                # 関数リストがある場合、まず実行
                if len(function_requests) > 0:
                    execute_function_call_list(function_requests)
                print(f"Function not required, responding to user.")
                print(assistant_message)
                break
            time.sleep(.5)
        time.sleep(.5)


if __name__ == "__main__":
    main()
