# photo_critique_tts.py
import argparse, json, re
from io import BytesIO
from pathlib import Path
from typing import Tuple

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

# Import your StyleTTS2 wrapper
from styletts2 import tts  

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except Exception:
    from transformers import AutoModelForVision2Seq as AutoVLM

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

#1
# SYSTEM_PROMPT = (
   # "You are a photography coach. Analyze the photo and provide ONLY a short, "
   #"constructive critique (2–3 sentences) focused on what could be improved and how to fix it.\n"
   #"Consider: composition, exposure, focus, color balance, lighting, background separation, noise, post-processing.\n"
   #"Avoid any comments about identity or sensitive attributes. Do not praise the image."
#
SYSTEM_PROMPT = (
    "You are a photography assistant. Analyze the photo and provide ONLY a short, direct set of "
    "instructions (2–3 sentences) describing what can be done to make the image look great.\n"
    "Focus only on actionable improvements such as: adjust exposure, crop the frame, correct color balance, "
    "increase contrast, reduce noise, sharpen details, enhance background separation, or apply post-processing edits.\n"
)


# ----------------- IMAGE HELPERS -----------------

def load_image(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        r = requests.get(src, timeout=20); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(src).convert("RGB")
    return img

def maybe_downscale(img: Image.Image, max_side: int = 1280) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / float(max(w, h))
    return img.resize((int(w * s), int(h * s)), Image.LANCZOS)

# ----------------- VLM -----------------

def load_model() -> Tuple[AutoProcessor, AutoVLM]:
    quant = None
    if torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoVLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    print("VLM loaded.")
    return processor, model

def critique_image(processor, model, image_source: str,
                   max_new_tokens=160, temperature=0.2) -> str:
    image = maybe_downscale(load_image(image_source), 1280)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Please critique this photo."}
        ]},
    ]
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, return_tensors=None
    )
    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )
    gen_only = output_ids[0][prompt_len:]
    text_out = processor.decode(gen_only, skip_special_tokens=True).strip()

    parts = re.split(r"(?<=[.!?])\s+", text_out)
    return " ".join(parts[:2]).strip() or text_out

# ----------------- TTS -----------------

_engine = None
def get_tts_engine():
    global _engine
    if _engine is None:
        _engine = tts.StyleTTS2(model_checkpoint_path=None, config_path=None)
    return _engine

def synthesize(text: str, out_path: str = "critique.wav",
               steps: int = 5, alpha: float = 0.3, beta: float = 0.7) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    engine = get_tts_engine()
    print(f"Synthesizing speech to {out_path}…")
    engine.inference(
        text,
        output_wav_file=out_path,
        diffusion_steps=steps,
        alpha=alpha,
        beta=beta,
        embedding_scale=1.0,
    )
    return out_path

# ----------------- MAIN -----------------

def main():
    ap = argparse.ArgumentParser(description="Photo critique + TTS feedback")
    ap.add_argument("--image", required=True, help="Path or URL to the image")
    ap.add_argument("--out-audio", default="critique.wav", help="Output WAV path")
    ap.add_argument("--steps", type=int, default=5, help="TTS diffusion steps")
    ap.add_argument("--alpha", type=float, default=0.3, help="Timbre balance")
    ap.add_argument("--beta", type=float, default=0.7, help="Prosody balance")
    args = ap.parse_args()

    processor, model = load_model()
    critique = critique_image(processor, model, args.image)
    wav = synthesize(critique, args.out_audio,
                     steps=args.steps, alpha=args.alpha, beta=args.beta)

    result = {"critique": critique, "audio_file": wav}
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
