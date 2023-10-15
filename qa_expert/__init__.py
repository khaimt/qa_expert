from enum import Enum
from qa_expert.base_inference import ModelInference


class InferenceType(str, Enum):
    hf = "hf"
    vllm = "vllm"
    server = "server"


def get_inference_model(model_path_or_service: str, model_type: InferenceType, **kwargs) -> ModelInference:
    if model_type == InferenceType.hf:
        from qa_expert.hf_inference import HFInference

        return HFInference(model_path_or_service, **kwargs)
    if model_type == InferenceType.vllm:
        from qa_expert.vllm_inference import VllmInference

        return VllmInference(model_path_or_service, **kwargs)
    if model_type == InferenceType.server:
        from qa_expert.server_inference import ServerInference

        return ServerInference(model_path_or_service, **kwargs)
    raise Exception(f"model_type: {model_type} is not supported")
