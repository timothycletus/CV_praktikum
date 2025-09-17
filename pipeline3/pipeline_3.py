# app_gradio.py
import gradio as gr
from combined import load_model, critique_image, synthesize

# Load VLM once
processor, model = load_model()

def critique_and_speak(image):
    # Generate critique text
    critique = critique_image(processor, model, image)

    # Synthesize critique into audio
    audio_path = synthesize(critique, out_path="critique.wav")

    return critique, audio_path


with gr.Blocks(title="Photo Critique with TTS") as demo:
    gr.Markdown("## 📸 AI Photo Critique + Spoken Feedback")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload a Photo")
            submit_btn = gr.Button("Analyze & Speak")

        with gr.Column():
            critique_output = gr.Textbox(label="Critique", lines=3)
            audio_output = gr.Audio(label="Spoken Feedback")

    submit_btn.click(
        fn=critique_and_speak,
        inputs=image_input,
        outputs=[critique_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch()
