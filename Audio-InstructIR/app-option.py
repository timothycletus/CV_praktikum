import os
import uuid
import yaml
import torch
import whisper
import datetime
import argparse
import numpy as np
import gradio as gr
from PIL import Image
import torchaudio
from torchaudio.transforms import Resample
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForCTC

# Local modules
from models import instructir
from text.models import LanguageModel, LMHead


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


# ------------------ Model Setup ------------------
hf_hub_download(repo_id="marcosv/InstructIR", filename="im_instructir-7d.pt", local_dir="./")
hf_hub_download(repo_id="marcosv/InstructIR", filename="lm_instructir-7d.pt", local_dir="./")

CONFIG     = "configs/eval5d.yml"
LM_MODEL   = "lm_instructir-7d.pt"
MODEL_NAME = "im_instructir-7d.pt"

with open(CONFIG, "r") as f:
    cfg = dict2namespace(yaml.safe_load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image model
model = instructir.create_model(
    input_channels=cfg.model.in_ch,
    width=cfg.model.width,
    enc_blks=cfg.model.enc_blks,
    middle_blk_num=cfg.model.middle_blk_num,
    dec_blks=cfg.model.dec_blks,
    txtdim=cfg.model.textdim
).to(device)
model.load_state_dict(torch.load(MODEL_NAME, map_location=device), strict=True)

# Language model
language_model = LanguageModel(model=cfg.llm.model)
lm_head = LMHead(
    embedding_dim=cfg.llm.model_dim,
    hidden_dim=cfg.llm.embd_dim,
    num_classes=cfg.llm.nclasses
).to(device)
lm_head.load_state_dict(torch.load(LM_MODEL, map_location=device), strict=True)

# Whisper model
whisper_model = whisper.load_model("base")

# MMS model
mms_processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
mms_model = AutoModelForCTC.from_pretrained("facebook/mms-1b-all").to(device)


# ------------------ ASR Transcription ------------------
def transcribe(audio_file, model_choice):
    if audio_file is None:
        return ""

    if model_choice == "Whisper":
        result = whisper_model.transcribe(audio_file)
        return result["text"]

    elif model_choice == "MMS":
        waveform, sr = torchaudio.load(audio_file)
        if sr != 16000:
            waveform = Resample(orig_freq=sr, new_freq=16000)(waveform)

        inputs = mms_processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = mms_model(**inputs.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = mms_processor.batch_decode(predicted_ids)[0]
        return transcription

    else:
        return "[Unknown ASR Model]"


# ------------------ Image Restoration ------------------
def process_img(image, prompt):
    img = np.array(image).astype(np.float32) / 255.
    y = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    lm_embd = language_model(prompt).to(device)
    with torch.no_grad():
        text_embd, _ = lm_head(lm_embd)
        x_hat = model(y, text_embd)

    restored_img = x_hat.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((restored_img * 255).round().astype(np.uint8))


def pipeline(image, audio_file, asr_model_choice):
    prompt = transcribe(audio_file, asr_model_choice)
    restored_img = process_img(image, prompt)

    # Save YAML log
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = str(uuid.uuid4())[:8]
    os.makedirs("results", exist_ok=True)

    log = {
        "session": session_id,
        "timestamp": timestamp,
        "asr_model": asr_model_choice,
        "prompt": prompt,
        "input_image": "input.png",
        "output_image": f"output_{asr_model_choice}_{timestamp}.png",
        "audio_file": audio_file if isinstance(audio_file, str) else "browser_upload"
    }

    # Save images
    if isinstance(image, Image.Image):
        image.save("results/input.png")
        restored_img.save(os.path.join("results", log["output_image"]))

    # Save YAML
    with open(f"results/session_{asr_model_choice}_{timestamp}.yaml", "w") as f:
        yaml.dump(log, f)

    return restored_img, prompt


# ------------------ Gradio UI ------------------
title = "InstructIR ✏️🖼️ + 🎤 Whisper/MMS Voice Input"
description = "Upload an image, choose voice model, speak your instruction, and InstructIR will restore it."

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="🖼️ Upload Image")
            audio_input = gr.Audio(label="🎤 Record Instruction", type="filepath")
            asr_choice = gr.Dropdown(["Whisper", "MMS"], label="Select ASR Model", value="Whisper")
            run_button = gr.Button("🪄 Restore from Voice")

        with gr.Column():
            output_image = gr.Image(type="pil", label="🧽 Restored Image")
            prompt_box = gr.Textbox(label="📝 Transcribed Instruction")

    run_button.click(fn=pipeline, inputs=[input_image, audio_input, asr_choice], outputs=[output_image, prompt_box])

# ------------------ Launch ------------------
if __name__ == "__main__":
    demo.launch(share=True)
