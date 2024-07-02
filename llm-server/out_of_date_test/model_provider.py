from typing import List, Tuple, Dict
import os
import torch
import gc
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel


on_server: bool = not os.path.exists("not-on-server")
if on_server:
    print("ON SERVER")
    device_name: str = "cuda:0"
    device: torch.device = torch.device(device_name)
    device_type: str = "gpu"
else:
    print("NOT ON SERVER")
    device_name: str = "cpu"
    device: torch.device = torch.device(device_name)
    device_type: str = "cpu"
    with open("huggingface-token", "r") as file:
        token = file.read().strip()
    login(token)
    del token   


class ModelProvider:
    
    
    generative_model_ids: List[str] = [
        "google/codegemma-2b",
        "checkpoints/finetuned_0"
    ]
    embedding_model_ids: List[str] = [
        "microsoft/codebert-base"
    ]

    generative_model_base_models: Dict[str, str | None] = {
        "google/codegemma-2b": None,
        "checkpoints/finetuned_0": "google/codegemma-2b"
    }

    _loaded_generative_models: Dict[str, Tuple[AutoModelForCausalLM, PreTrainedTokenizer] | None] = {
        model_id: None
        for model_id in generative_model_ids
    }
    _loaded_embedding_models: Dict[str, SentenceTransformer | None] = {
        model_id: None
        for model_id in embedding_model_ids
    }

    @classmethod
    def get_generative_model(cls, model_id: str) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
        if cls._loaded_generative_models[model_id] is None:
            if (base_model_id := cls.generative_model_base_models[model_id]) is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    device_map=device,
                    torch_dtype=(torch.bfloat16 if on_server else torch.float32)
                )
                cls._loaded_generative_models[model_id] = PeftModel.from_pretrained(
                    model,
                    model_id
                )

            else:
                cls._loaded_generative_models[model_id] = (
                    AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map=device,
                        torch_dtype=(torch.bfloat16 if on_server else torch.float32)
                    ),
                    AutoTokenizer.from_pretrained(model_id)
                )
        return cls._loaded_generative_models[model_id]
    
    @classmethod
    def unload_generative_model(cls, model_id: str):
        if cls._loaded_generative_models[model_id] is not None:
            cls._loaded_generative_models[model_id] = None
            gc.collect()
            if device_type == "gpu":
                torch.cuda.empty_cache()

    @classmethod
    def get_embedding_model(cls, model_id: str) -> SentenceTransformer:
        if cls._loaded_embedding_models[model_id] is None:
            cls._loaded_embedding_models[model_id] = SentenceTransformer(
                model_id,
                device=device_name
            )
        return cls._loaded_embedding_models[model_id]

    @classmethod
    def unload_embedding_model(cls, model_id: str):
        if cls._loaded_embedding_models[model_id] is not None:
            cls._loaded_embedding_models[model_id] = None
            gc.collect()
            if device_type == "gpu":
                torch.cuda.empty_cache()

    @classmethod
    def get_loaded_generative_models(cls):
        return {mid: model is not None for mid, model in cls._loaded_generative_models.items()}
    

    @classmethod
    def get_loaded_embedding_models(cls):
        return {mid: model is not None for mid, model in cls._loaded_embedding_models.items()}
    