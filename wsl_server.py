"""Gemma-4 VLM Backend FastAPI Server.

Purpose:
    This module implements the backend REST server wrapper for the google/gemma-4-12B-it
    multimodal model. It runs inside the WSL2 Docker container to process base64 encoded
    images and prompt instructions, outputting structural JSON descriptions or unconstrained
    free-form text.

Architecture and Mechanics:
    1. Environment & Caching: Sets Hugging Face offline directories (/workspace/models/huggingface)
       and disables standard progress bars.
    2. Model Quantization Lifecycle: Optimizes memory foot-print by checking for a pre-saved
       4-bit quantized model checkpoint. If missing, it loads the base BF16 weights, quantizes
       on-the-fly using BitsAndBytes (NF4 double-quant), and saves the compiled checkpoint to
       skip subsequent boot delays.
    3. LayerNorm Monkey-Patch: Patches Gemma4UnifiedVisionEmbedder's forward pass to cast inputs
       properly, solving a known 4-bit LayerNorm precision mismatch bug in transformers.
    4. FastAPI Routing endpoints:
       - /describe: Accepts batch base64 images and generates structured JSON descriptions using a prompt suffix prefill.
       - /analyze: Accepts optional base64 images and prompts to execute free-form text generation in multimodal or text-only modes.
    5. VRAM Protection & Garbage Collection: Ensures strict post-inference manual cleaning (deleting input/output tensors, closing PIL images, invoking gc.collect(), and flushing CUDA caches) to prevent out-of-memory errors (OOM).

Execution Modes:
    - FastAPI/Uvicorn Service: Intended to be run under Uvicorn inside the Docker container.
      Command:
        uvicorn gemma_cataloger.wsl_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys


# Setup HF Cache Directory inside container (should map to /workspace/models/huggingface)
HF_CACHE_DIR: str = "/workspace/models/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import base64
import io
import logging
from typing import List, Dict, Optional, Any
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable decompression limit to allow high-resolution upscaled images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow reading truncated or slightly corrupted files
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize logger
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
    PT_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    # Fallback to standard UTC-7 offset (PDT) if zoneinfo database is unavailable in the container
    PT_TZ = timezone(timedelta(hours=-7))

def pt_converter(*args: Any) -> tuple:
    """Converts standard epoch timestamp to Pacific Time timetuple for logger formatting.

    Args:
        args: Variable arguments list. The last argument is expected to be the Unix epoch timestamp.

    Returns:
        Time structure tuple in PT.
    """
    timestamp = args[-1]
    return datetime.fromtimestamp(timestamp, tz=PT_TZ).timetuple()

# Apply the timezone converter globally to all logging formatters
logging.Formatter.converter = pt_converter

LOG_FILE: str = "/workspace/gemma_cataloger/uvicorn.log"
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# File handler
try:
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
except Exception as e:
    sys.stderr.write(f"Failed to add file handler: {e}\n")

logger = logging.getLogger(__name__)

# Model default config
MODEL_ID: str = "google/gemma-4-12B-it"
QUANTIZED_MODEL_PATH: str = "/workspace/models/gemma-4-12b-it-quantized-4bit"
REPETITION_PENALTY: float = 1.15  # Set to 1.15 to prevent infinite loops while preserving accuracy

# Load Hugging Face Token for authentication (optional since offline)
TOKEN_PATH: str = "/workspace/gemma_cataloger/auth/huggingface_token.txt"
if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH, "r", encoding="utf-8") as f:
        token: str = f.read().strip()
        os.environ["HF_TOKEN"] = token

app = FastAPI(title="Gemma 4 VLM Server", description="WSL2 Backend serving Gemma 4 VLM on-demand.")

# Global placeholders for the model and processor
model: Optional[AutoModelForMultimodalLM] = None
processor: Optional[AutoProcessor] = None
dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class DescriptionRequest(BaseModel):
    """Schema representing an image processing payload request."""
    images_base64: List[str]
    prompt_text: str
    temperature: Optional[float] = 0.7


class DescriptionResponse(BaseModel):
    """Schema representing the server's generated response."""
    raw_responses: List[str]


class AnalysisRequest(BaseModel):
    """Schema representing an analysis payload request."""
    prompt_text: str
    images_base64: Optional[List[str]] = None
    temperature: Optional[float] = 0.7
    max_new_tokens: Optional[int] = 512


class AnalysisResponse(BaseModel):
    """Schema representing the server's unconstrained text response."""
    response: str



def patch_gemma4_unified() -> None:
    """Monkey-patches the LayerNorm uint8 casting bug inside transformers.models.gemma4_unified.

    Args:
        None

    Returns:
        None
    """
    try:
        import transformers.models.gemma4_unified.modeling_gemma4_unified as m
        
        def patched_forward(self: Any, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
            target_dtype = self.patch_ln1.weight.dtype
            hidden_states = self.patch_ln1(pixel_values.to(target_dtype))
            hidden_states = self.patch_dense(hidden_states)
            hidden_states = self.patch_ln2(hidden_states)

            clamped = image_position_ids.clamp(min=0).long()
            valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
            axes = torch.arange(2, device=image_position_ids.device)
            pos_embs = (self.pos_embedding[clamped, axes] * valid).sum(-2)
            hidden_states = hidden_states + pos_embs
            hidden_states = self.pos_norm(hidden_states)

            hidden_states = self.multimodal_embedder(hidden_states)
            return hidden_states

        m.Gemma4UnifiedVisionEmbedder.forward = patched_forward
        logger.info("Monkey-patched Gemma4UnifiedVisionEmbedder.forward successfully.")
    except Exception as e:
        logger.error(f"Failed to apply monkey-patch: {e}")


@app.on_event("startup")
def load_model() -> None:
    """Loads the gemma-4-12B-it model and processor into VRAM on startup.

    Utilizes BitsAndBytes 4-bit NF4 double quantization with a bfloat16
    compute type to ensure the model fits and runs optimally within VRAM.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If model weight loading or device mapping fails.
    """
    global model, processor
    
    try:
        patch_gemma4_unified()
        gpu_name: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
        logger.info(f"Detected GPU: {gpu_name}")
        logger.info(f"Loading processor for: {MODEL_ID}")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        
        logger.info(f"Loading model with BitsAndBytes 4-bit config to fit {gpu_name} (RTX 5080)...")
        # We skip quantization for critical layers (vision and audio projection layers, plus output lm_head)
        # to prevent quantization degradation from corrupting multimodal inputs (image features).
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[
                "lm_head",
                "model.embed_vision.patch_dense",
                "model.embed_vision.multimodal_embedder.embedding_projection",
                "model.embed_audio.embedding_projection"
            ]
        )
        
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="cuda:0"
        )
        
        model.eval()
        
        enable_compile = os.environ.get("ENABLE_COMPILE", "0") == "1"
        if enable_compile:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead", backend="inductor")
            
        logger.info("Gemma 4 model loaded successfully on local server.")
            
    except Exception as e:
        logger.critical(f"Failed to load VLM model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")


@app.post("/describe", response_model=DescriptionResponse)
def describe_images(request: DescriptionRequest) -> DescriptionResponse:
    """Processes base64 images and generates structural descriptions.

    Args:
        request: A Pydantic model containing the prompt and base64 encoded images.

    Returns:
        DescriptionResponse with the VLM's generated JSON strings.

    Raises:
        HTTPException: If the server is not ready or if inference fails.
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model server is still initializing.")
    
    pil_images: List[Image.Image] = []
    inputs: Optional[Dict[str, torch.Tensor]] = None
    output_ids: Optional[torch.Tensor] = None
    try:
        for img_b64 in request.images_base64:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_images.append(img)
        
        if not pil_images:
            return DescriptionResponse(raw_responses=[])
        
        # Build individual prompts for each image to match local batched inference
        prompts: List[str] = []
        for _ in range(len(pil_images)):
            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": request.prompt_text
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": (
                            "Analyze this image according to the archival schema. "
                            "The JSON object must have exactly the following structure:\n"
                            "{\n"
                            "  \"primary_subject\": \"...\",\n"
                            "  \"environment\": \"...\",\n"
                            "  \"suggested_tags\": []\n"
                            "}\n"
                            "Return only the raw JSON string. Do not include markdown formatting code blocks."
                        )}
                    ]
                }
            ]
            base_prompt: str = processor.apply_chat_template(messages, add_generation_prompt=True)
            prefill: str = '{\n  "primary_subject": "'
            prompts.append(base_prompt + prefill)
        
        # Prepare batched inputs (wrap each image in its own list to match Gemma 4 batch format)
        inputs = processor(text=prompts, images=[[img] for img in pil_images], padding="longest", return_tensors="pt").to("cuda:0")
        
        # Adjust prompt inputs to match model_dtype
        if inputs is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            
        gen_temp = request.temperature if request.temperature is not None else 0.7
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=gen_temp,
                repetition_penalty=REPETITION_PENALTY
            )
            
        # Extract response slice starting after the input prompt tokens
        input_len = inputs["input_ids"].shape[1]
        
        raw_responses: List[str] = []
        prefill: str = '{\n  "primary_subject": "'
        for out_tokens in output_ids:
            generated_ids = out_tokens[input_len:]
            decoded_text: str = processor.decode(generated_ids, skip_special_tokens=True).strip()
            if decoded_text.startswith('"'):
                decoded_text = decoded_text[1:]
            full_response = prefill + decoded_text
            raw_responses.append(full_response)
            
        return DescriptionResponse(raw_responses=raw_responses)
        
    except Exception as e:
        logger.error(f"Inference processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")
    finally:
        # Explicit clean-up to prevent memory and VRAM leaks
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass
        del pil_images
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_content(request: AnalysisRequest) -> AnalysisResponse:
    """Processes optional base64 images and a custom prompt to generate unconstrained free-form text.

    If images are provided, it runs in VLM multimodal mode. If no images are
    provided, it runs in text-only LLM mode.

    Args:
        request: An AnalysisRequest containing the custom prompt text,
            optional list of base64-encoded images, sampling temperature,
            and max new token limit.

    Returns:
        An AnalysisResponse containing the raw, unconstrained generated text response.

    Raises:
        HTTPException: If the server is not ready or if inference fails.
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model server is still initializing.")
    
    pil_images: List[Image.Image] = []
    inputs = None
    output_ids = None
    try:
        # Decode optional base64 images
        if request.images_base64:
            for img_b64 in request.images_base64:
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_images.append(img)

        # Build prompt using chat template
        content_structure = []
        for _ in range(len(pil_images)):
            content_structure.append({"type": "image"})
        content_structure.append({"type": "text", "text": request.prompt_text})

        messages = [
            {
                "role": "user",
                "content": content_structure
            }
        ]
        base_prompt: str = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Tokenize based on input availability
        if pil_images:
            inputs = processor(text=[base_prompt], images=[pil_images], padding="longest", return_tensors="pt").to("cuda:0")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        else:
            inputs = processor(text=[base_prompt], return_tensors="pt").to("cuda:0")

        gen_temp = request.temperature if request.temperature is not None else 0.7
        max_tokens = request.max_new_tokens if request.max_new_tokens is not None else 512

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(gen_temp > 0.0),
                temperature=gen_temp if gen_temp > 0.0 else None,
                repetition_penalty=REPETITION_PENALTY
            )

        # Extract generated response slice
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        decoded_text: str = processor.decode(generated_ids, skip_special_tokens=True).strip()

        return AnalysisResponse(response=decoded_text)

    except Exception as e:
        logger.error(f"Analysis processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")
    finally:
        # Explicit clean-up to prevent memory and VRAM leaks
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass
        del pil_images
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
