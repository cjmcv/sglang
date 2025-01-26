import openai
from pydantic import BaseModel
import json
from typing import List

# client = openai.Client(
#     base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
client = openai.Client(
    base_url="http://127.0.0.1:8000/v1/", api_key="EMPTY")


# # Text completion
# response = client.completions.create(
# 	model="default",
# 	prompt="The capital of France is",
# 	temperature=0,
# 	max_tokens=32,
# )
# print(response)

# # Chat completion
# response = client.chat.completions.create(
#     model="default",
#     messages=[
#         {"role": "system", "content": "You are a helpful expert math tutor"},
#         {"role": "user", "content": "Solve 8x + 31 = 2."},
#     ],
#     temperature=0,
#     max_tokens=200
# )
# print(response)

#####################################################################################
# class Step(BaseModel):
#     explanation: str
#     output: str

# class MathResponse(BaseModel):
#     steps: List[Step]
#     final_answer: str

# json_schema = MathResponse.model_json_schema()

# response = client.beta.chat.completions.parse(
#     model="default",
#     messages=[
#         {"role": "system", "content": "You are a helpful expert math tutor"},
#         {"role": "user", "content": "Solve 8x + 31 = 2."},
#     ],
#     # response_format={
#     #     "type": "json_schema",
#     #     "json_schema": {"name": "foo", "schema": json.dumps(json_schema)},
#     # },
#     response_format=MathResponse,
#     extra_body={
#         'guided_decoding_bachend': "xgrammar",
#         'guided_json': json_schema
#     },
#     temperature=0,
#     seed=0
# )
# # print(response)
# message = response.choices[0].message
# print(message)

# # for i,step in enumerate(message.parsed.steps):
# #     print(f"Step #{i}:", step)
# print("final_answer: ", message.parsed.final_answer)


#########################################################################
json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "number"},
            "history": {"type": "string"},
        },
        "required": ["name", "population", "history"],
    }
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "Introduce the capital of France in details."},
    ],
    temperature=0,
    max_tokens=1200,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
    },
)
text = response.choices[0].message.content
print(text)

# try:
#     js_obj = json.loads(text)
# except (TypeError, json.decoder.JSONDecodeError):
#     print("JSONDecodeError", text)
#     raise

# print(js_obj["name"])
# print(js_obj["population"])
# print(js_obj["history"])

# json_schema = json.dumps(
#     {
#         "type": "object",
#         "properties": {
#             "equation": {"type": "string"},
#             "solution": {"type": "number"},
#         },
#         "required": ["equation", "solution"]
#     }
# )

# response = client.chat.completions.create(
#     model="default",
#     messages=[
#         {"role": "system", "content": "You are a helpful expert math tutor."},
#         {"role": "user", "content": "Solve 8x + 31 = 2."},
#     ],
#     temperature=0,
#     max_tokens=300,
#     response_format={
#         "type": "json_schema",
#         "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
#     },
# )
# text = response.choices[0].message.content

# try:
#     js_obj = json.loads(text)
# except (TypeError, json.decoder.JSONDecodeError):
#     print("JSONDecodeError", text)
#     raise

# print(text)
# # print(js_obj["equation"])
# # print(js_obj["solution_steps"])
# # print(js_obj["solution"])