# ===================================================
# 📰 Fake News Generator & Detector (GPT-2 + BERT)
# ===================================================

!pip install -q transformers torch gradio

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-2 for Fake News Generation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Load BERT for Fake News Detection
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).to(device)

# Function: Generate Fake News
def generate_news(prompt, max_len):
    if not prompt.strip():
        return "⚠️ Please enter a prompt to generate news."

    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(
        inputs,
        max_length=max_len,
        num_return_sequences=1,
        temperature=0.8,
        top_k=60,
        top_p=0.92,
        no_repeat_ngram_size=2,
        do_sample=True,
        early_stopping=True
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function: Detect Fake News
def detect_fake_news(text):
    if not text.strip():
        return "⚠️ Please enter some text to analyze.", 0

    tokens = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**tokens)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]
    prediction = torch.argmax(probs).item()

    label = "✅ Real News" if prediction == 1 else "❌ Fake News"
    confidence_score = float(probs[prediction].item())

    return f"{label} (Confidence: {confidence_score:.2f})", confidence_score

# Utility: Count Characters
def count_characters(text):
    return f"Character Count: {len(text)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 📰 AI Powered Fake News Generator & Detector")
    gr.Markdown("Use GPT-2 to **generate fake news** and BERT to **detect if news is fake or real**.")

    # Tab 1: Generate Fake News
    with gr.Tab("🛠 Generate News"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="News Headline or Prompt",
                    placeholder="E.g., Scientists discovered a new planet near Jupiter...",
                    lines=2
                )
                max_length_slider = gr.Slider(50, 300, value=150, step=10, label="Max Output Length (Tokens)")

            with gr.Column():
                generate_button = gr.Button("🚀 Generate News")
                clear_button = gr.Button("🧹 Clear")

        generated_output = gr.Textbox(label="Generated News Article", lines=8)

        # Actions
        generate_button.click(generate_news, inputs=[prompt_input, max_length_slider], outputs=generated_output)
        clear_button.click(lambda: ("", ""), outputs=[prompt_input, generated_output])

    # Tab 2: Detect Fake News
    with gr.Tab("🔍 Detect Fake News"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Paste News Article",
                    placeholder="Paste a news paragraph here to check if it's fake or real...",
                    lines=6
                )
                char_count = gr.Textbox(label="Character Info", interactive=False)
                input_text.change(count_characters, inputs=input_text, outputs=char_count)

        with gr.Row():
            detect_button = gr.Button("🔍 Analyze")
            clear_detect_button = gr.Button("🧹 Clear Detection")

        with gr.Row():
            detection_result = gr.Textbox(label="Detection Result")
            confidence_bar = gr.Slider(0, 1, value=0, step=0.01, label="Confidence Level", interactive=False)

        # Actions
        detect_button.click(
            detect_fake_news,
            inputs=input_text,
            outputs=[detection_result, confidence_bar]
        )
        clear_detect_button.click(lambda: ("", 0), outputs=[detection_result, confidence_bar])

# Launch the App

app.launch()
