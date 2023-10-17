from enum import Enum
from qa_expert.base_inference import ModelInference


class InferenceType(str, Enum):
    hf = "hf"
    vllm = "vllm"
    server = "server"
    llama_cpp = "llama_cpp"


def get_inference_model(model_path_or_service: str, model_type: InferenceType, *args, **kwargs) -> ModelInference:
    if model_type == InferenceType.hf:
        from qa_expert.hf_inference import HFInference

        return HFInference(model_path_or_service, *args, **kwargs)
    if model_type == InferenceType.vllm:
        from qa_expert.vllm_inference import VllmInference

        return VllmInference(model_path_or_service, *args, **kwargs)
    if model_type == InferenceType.server:
        from qa_expert.server_inference import ServerInference

        return ServerInference(model_path_or_service, *args, **kwargs)
    if model_type == InferenceType.llama_cpp:
        from qa_expert.llama_cpp_inference import LLamaCppInference

        return LLamaCppInference(model_path_or_service, *args, **kwargs)

    raise Exception(f"model_type: {model_type} is not supported")
