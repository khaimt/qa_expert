from qa_expert import get_inference_model, InferenceType
from qa_expert.prompt_utils import Message, Role


model_path = "models/qa_expert/qa-expert-7B-V1.0.q4_0.gguf"
model = get_inference_model(model_path, InferenceType.llama_cpp)
messages = [Message(role=Role.user, content="Were Scott Derrickson and Ed Wood of the same nationality?")]
mess = model.generate_message(messages)
print(mess)
