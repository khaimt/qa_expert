from qa_expert import get_inference_model, InferenceType
from qa_expert.prompt_utils import Message, Role, FunctionCall

context = """Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer..  He lives in Los Angeles, California..  He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange.". Scott Derrickson Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill..  It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.. Sinister (film)
""".strip()


model_path = "models/qa-expert-7B-V1.0.q4_0.gguf"
model = get_inference_model(InferenceType.llama_cpp, model_path, "khaimaitien/qa-expert-7B-V1.0-GGUF")
#model = get_inference_model(InferenceType.vllm, "models/qa-expert-7B-V1.0")
messages = [
        Message(role=Role.user, content="Were Scott Derrickson and Ed Wood of the same nationality?"),
        Message(role=Role.assistant, function_call=FunctionCall.from_query("What is the nationality of Scott Derrickson?")),
        Message(role=Role.function, content=context)
    ]
mess = model.generate_message(messages)
print(mess)