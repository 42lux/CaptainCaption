from tkinter import filedialog, Tk
import base64
import datetime
import io
import os
import time
from PIL import Image
import numpy as np
import gradio as gr
from gradio import Warning
from openai import OpenAI
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

FOLDER_SYMBOL = '\U0001f4c2'  # 📂
MAX_IMAGE_WIDTH = 2048
IMAGE_FORMAT = "JPEG"

def generate_description(api_key, image, prompt, detail, max_tokens):
    try:
        img = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image)
        img = scale_image(img)

        buffered = io.BytesIO()
        img.save(buffered, format=IMAGE_FORMAT)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        client = OpenAI(api_key=api_key)
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}", "detail": detail}}
                ]
            }],
            "max_tokens": max_tokens
        }

        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

history = [] 
columns = ["Time", "Prompt", "GPT4-Vision Caption"]

def clear_fields():
    global history
    history = []
    return "", []

def update_history(prompt, response):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    history.append({"Time": timestamp, "Prompt": prompt, "GPT4-Vision Caption": response})
    return [[entry[column] for column in columns] for entry in history]

def scale_image(img):
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / img.width
        new_height = int(img.height * ratio)
        return img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
    return img

def get_dir(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)

def get_folder_path(folder_path=''):
    current_folder_path = folder_path

    initial_dir, initial_file = get_dir(folder_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()

    if sys.platform == 'darwin':
        root.call('wm', 'attributes', '.', '-topmost', True)

    folder_path = filedialog.askdirectory(initialdir=initial_dir)
    root.destroy()

    if folder_path == '':
        folder_path = current_folder_path

    return folder_path

is_processing = True

def process_folder(api_key, folder_path, prompt, detail, max_tokens, pre_prompt="", post_prompt="", progress=gr.Progress(), num_workers=4):
    global is_processing
    is_processing = True 

    if not os.path.isdir(folder_path):
        return f"No such directory: {folder_path}"

    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    progress(0)

    def process_file(file):
        global is_processing
        if not is_processing:
            return "Processing canceled by user"

        image_path = os.path.join(folder_path, file)
        description = generate_description(api_key, image_path, prompt, detail, max_tokens)
        txt_path = os.path.join(folder_path, os.path.splitext(file)[0] + ".txt")

        with open(txt_path, 'w') as f:
            f.write(pre_prompt + ", " + description + " " + post_prompt)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, _ in enumerate(executor.map(process_file, file_list), 1):
            progress((i, len(file_list)))
            if not is_processing:
                break

    is_processing = False  
    return f"Processed {len(file_list)} images in folder {folder_path}"

with gr.Blocks() as app:
    api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="Enter your API key here", type="password", info="The OpenAI API is rate limited to 20 requests per second. A big dataset can take a long time to tag.")
    with gr.Tab("Prompt Engineering"):
        image_input = gr.Image(label="Upload Image")
        with gr.Row():
            prompt_input = gr.Textbox(scale=6, label="Prompt", value="What’s in this image? Provide a description without filler text like: The image depicts...", interactive=True)
            detail_level = gr.Radio(["high", "low", "auto"], scale=2, label="Detail", value="auto")
            max_tokens_input = gr.Number(scale=0, value=300, label="Max Tokens")
            submit_button = gr.Button("Generate Caption")
        output = gr.Textbox(label="GPT4-Vision Caption")    
        history_table = gr.Dataframe(headers=columns)
        clear_button = gr.Button("Clear")
        clear_button.click(clear_fields, inputs=[], outputs=[output, history_table])

        
    with gr.Tab("GPT4-Vision Tagging"):
        with gr.Row():
            folder_path_dataset = gr.Textbox(scale=8, label="Dataset Folder Path", placeholder="/home/user/dataset", interactive=True, info="The folder path select button is a bit of hack if it doesn't work you can just copy and paste the path to your dataset.")
            folder_button = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            folder_button.click(
                get_folder_path,
                outputs=folder_path_dataset,
                show_progress=False,
            )
        with gr.Row():
            prompt_input_dataset = gr.Textbox(scale=6, label="Prompt", value="What’s in this image? Provide a description without filler text like: The image depicts...", interactive=True)
            detail_level_dataset = gr.Radio(["high", "low", "auto"], scale=2, label="Detail", value="auto")
            max_tokens_input_dataset = gr.Number(scale=0, value=300, label="Max Tokens")
        with gr.Row():
            pre_prompt_input = gr.Textbox(scale=6, label="Prefix", placeholder="(Optional)", info="Will be added at the front of the caption.", interactive=True)
            post_prompt_input = gr.Textbox(scale=6, label="Postfix", placeholder="(Optional)", info="Will be added at the end of the caption.", interactive=True)
        with gr.Row():
            worker_slider = gr.Slider(minimum=1, maximum=4, value=2, step=1, scale=2, label="Number of Workers") 
            submit_button_dataset = gr.Button("Generate Captions", scale=3)
            cancel_button = gr.Button("Cancel", scale=3)
        with gr.Row():
            processing_results_output = gr.Textbox(label="Processing Results")

    def cancel_processing():
        global is_processing    
        is_processing = False
        return "Processing canceled"

    cancel_button.click(cancel_processing, inputs=[], outputs=[processing_results_output])

    def on_click(api_key, image, prompt, detail, max_tokens):
        if not api_key.strip():  
            raise Warning("Please enter your OpenAI API key.")

        if image is None:  
            raise Warning("Please upload an image.")

        description = generate_description(api_key, image, prompt, detail, max_tokens)  
        new_history = update_history(prompt, description)
        return description, new_history

    submit_button.click(on_click, inputs=[api_key_input, image_input, prompt_input, detail_level, max_tokens_input], outputs=[output, history_table])

    def on_click_folder(api_key, folder_path, prompt, detail, max_tokens, pre_prompt, post_prompt, worker_slider):
        if not api_key.strip():
            raise Warning("Please enter your OpenAI API key.")

        if not folder_path.strip():
            raise Warning("Please enter the folder path.")

        result = process_folder(api_key, folder_path, prompt, detail, max_tokens, pre_prompt, post_prompt)
        return result

    submit_button_dataset.click(
        on_click_folder, 
        inputs=[
            api_key_input, 
            folder_path_dataset, 
            prompt_input_dataset, 
            detail_level_dataset, 
            max_tokens_input_dataset, 
            pre_prompt_input, 
            post_prompt_input,
            worker_slider
        ], 
        outputs=[processing_results_output]
    )

app.launch()