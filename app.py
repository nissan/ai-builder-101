from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import tensorflow as tf


# GGML model required to fit Llama2-13B on a T4 GPU
GENERATIVE_AI_MODEL_REPO = "TheBloke/Llama-2-13B-chat-GGML"
GENERATIVE_AI_MODEL_FILE = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(
    repo_id=GENERATIVE_AI_MODEL_REPO,
    filename=GENERATIVE_AI_MODEL_FILE
)

llama2_model = Llama(
    model_path=model_path,
    n_gpu_layers=64,
    n_ctx=2000
)

# Test an inference
print(llama2_model(prompt="Hello ", max_tokens=1))


app = FastAPI()


# This defines the data json format expected for the endpoint, change as needed
class TextInput(BaseModel):
    inputs: str
    parameters: dict[str, Any] | None


@app.get("/")
def status_gpu_check() -> dict[str, str]:
    gpu_msg = "Available" if tf.test.is_gpu_available() else "Unavailable"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }


@app.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        params = data.parameters or {}
        response = llama2_model(prompt=data.inputs, **params)
        model_out = response['choices'][0]['text']
        return {"generated_text": model_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
