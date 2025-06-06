from google import genai
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Any
from mistralai.client import MistralClient
import json
import re
import asyncio
from datetime import datetime
import pandas as pd


load_dotenv('../.env')
print("[INFO] Environment variables loaded.")


class GeminiResponse(BaseModel):
    filename: str
    code: str


class AIRequests:

    def __init__(self):
        # GPT 4.1 mini
        self.gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.grok3_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
        self.deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"), endpoint="https://api.mistral.ai")

    def chatgpt_request(self, save_path: str, problem_id: str, problem_description: Any, programming_language: str):
        print(f"[CHATGPT-{programming_language}] problem_id: {problem_id}")
        response = self.gpt_client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": f"You are a coding assistant. filename must be like {problem_id}_gpt.{programming_language}"},
                {"role": "user",
                 "content": f"Solve this problem_id {problem_id} in {programming_language}: {problem_description}"}
            ],
            functions=[
                {
                    "name": "return_code_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "code": {"type": "string"}
                        },
                        "required": ["filename", "code"]
                    }
                }
            ]
        )
        function_call = response.choices[0].message.function_call
        if function_call and function_call.arguments:
            json_format = json.loads(function_call.arguments)
            with open(f"{save_path}/{json_format.get('filename')}", "w") as result_file:
                result_file.write(json_format.get("code"))

        print(f"[CHATGPT-{programming_language}] {problem_id} generation completed at {datetime.now()}")

    def gemini_request(self, save_path: str, problem_id: str, problem_description: Any, programming_language: str):
        print(f"[GEMINI-{programming_language}] problem_id: {problem_id}")
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"You are a coding assistant. filename must be like {problem_id}_gemini.{programming_language}. Solve this problem_id {problem_id} in {programming_language}: {problem_description}",
            config={
                "response_mime_type": "application/json",
                "response_schema": GeminiResponse,
            }
        )
        gemini_response = response.parsed
        with open(f"{save_path}/{gemini_response.filename}", "w") as f:
            f.write(gemini_response.code)

        print(f"[GEMINI-{programming_language}] {problem_id} generation completed at {datetime.now()}")

    def grok3_request(self, save_path: str, problem_id: str, problem_description: Any, programming_language: str):
        print(f"[GROK3-{programming_language}] problem_id: {problem_id}")
        if programming_language == "py":
            response = self.grok3_client.chat.completions.create(
                model="grok-3-mini-fast-beta",
                messages=[
                    {"role": "system",
                     "content": f"You are a coding assistant. without any explanation. just return the code, like ```python\n(.*?)\n```"},
                    {"role": "user",
                     "content": f"Solve this problem_id {problem_id} in {programming_language}: {problem_description}"}
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"```python\n(.*?)\n```", response_content, re.DOTALL)
            if not match:
                print("[GROK3-ERROR] No valid Python code block found in response.")
                return
        elif programming_language == "cpp":
            response = self.grok3_client.chat.completions.create(
                model="grok-3-mini-fast-beta",
                messages=[
                    {"role": "system",
                     "content": f"You are a coding assistant. without any explanation. just return the code, like ```cpp\n(.*?)\n```"},
                    {"role": "user",
                     "content": f"Solve this problem_id {problem_id} in {programming_language}: {problem_description}"}
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"```cpp\n(.*?)\n```", response_content, re.DOTALL)
            if not match:
                print("[GROK3-ERROR] No valid Cpp code block found in response.")
                return
        else:
            raise Exception("[GROK3-ERROR] Unknown programming language.")

        code = match.group(1).strip()
        with open(f"{save_path}/{problem_id}_grok3.{programming_language}", "w", encoding="utf-8") as result_file:
            result_file.write(code)

        print(f"[GROK3-{programming_language}] {problem_id} generation completed at {datetime.now()}")


    def deepseek_request(self, save_path: str, problem_id: str, problem_description: Any, programming_language: str):
        print(f"[DEEPSEEK-{programming_language}] problem_id: {problem_id}")
        if programming_language == "py":
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": f"You are a coding assistant, without any explanation. just return the code, like ```python\n(.*?)\n```"},
                    {"role": "user",
                     "content": f"Solve this problem_id {problem_id} in {programming_language}: {problem_description}"}
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"```python\n(.*?)\n```", response_content, re.DOTALL)
            if not match:
                print("[DEEPSEEK-ERROR] No valid Python code block found in response.")
                return
        elif programming_language == "cpp":
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": f"You are a coding assistant, without any explanation. just return the code, like ```cpp\n(.*?)\n```"},
                    {"role": "user",
                     "content": f"Solve this problem_id {problem_id} in {programming_language}: {problem_description}"}
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"```cpp\n(.*?)\n```", response_content, re.DOTALL)
            if not match:
                print("[DEEPSEEK-ERROR] No valid Cpp code block found in response.")
                return
        else:
            raise Exception("[DEEPSEEK-ERROR] Unknown programming language.")

        code = match.group(1).strip()
        with open(f"{save_path}/{problem_id}_deepseek.{programming_language}", "w", encoding="utf-8") as result_file:
            result_file.write(code)

        print(f"[DEEPSEEK-{programming_language}] {problem_id} generation completed at {datetime.now()}")


    def mistral_request(self, save_path: str, problem_id: str, problem_description: Any, programming_language: str):
        print(f"[MISTRAL-{programming_language}] problem_id: {problem_id}")
        if programming_language == "py":
            response = self.mistral_client.chat(
                model="mistral-small-latest",
                messages=[
                    {"role": "system",
                     "content": f"You are a coding assistant, without any explanation. just return the code, like ```python\n(.*?)\n```"},
                    {"role": "user",
                     "content": f"Solve this problem_id {problem_id} in {programming_language}: {problem_description}"}
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"```python\n(.*?)\n```", response_content, re.DOTALL)
            if not match:
                print("[MISTRAL-ERROR] No valid Python code block found in response.")
                return
        elif programming_language == "cpp":
            response = self.mistral_client.chat(
                model="mistral-small-latest",
                messages=[
                    {"role": "system",
                     "content": f"You are a coding assistant, without any explanation. just return the code, like ```cpp\n(.*?)\n```"},
                    {"role": "user",
                     "content": f"Solve this problem_id {problem_id} in {programming_language}: {problem_description}"}
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"```cpp\n(.*?)\n```", response_content, re.DOTALL)
            if not match:
                print("[MISTRAL-ERROR] No valid Cpp code block found in response.")
                return
        else:
            raise Exception("[MISTRAL-ERROR] Unknown programming language.")

        code = match.group(1).strip()
        with open(f"{save_path}/{problem_id}_mistral.{programming_language}", "w", encoding="utf-8") as result_file:
            result_file.write(code)

        print(f"[MISTRAL-{programming_language}] {problem_id} generation completed at {datetime.now()}")


if __name__ == "__main__":
    request_instance: AIRequests = AIRequests()

    print("#######################")
    print("## AI Code Generator ##")
    print("#######################")

    df = pd.read_csv('../csv/sample250_problem_list.csv')
    problem_ids = df["problem_id"]

    async def run_requests() -> None:
        for language in ["cpp", "py"]:
            for problem_id in problem_ids:
                os.makedirs(f"../ai_codes/{problem_id}", exist_ok=True)
                save_path: str = f"../ai_codes/{problem_id}"
                with open(f"../problem_descriptions/{problem_id}.html", "r") as problem:
                    description: str = problem.read()
                    request_tasks: list[str] = [
                        "chatgpt_request",
                        "gemini_request",
                        "grok3_request",
                        "deepseek_request",
                        "mistral_request"
                    ]

                    tasks: list[Any] = [
                        asyncio.to_thread(
                            getattr(request_instance, task_name),
                            save_path, problem_id, description, language
                        )
                        for task_name in request_tasks
                    ]

                    await asyncio.gather(*tasks)

                print("#" * 100)

    asyncio.run(run_requests())
