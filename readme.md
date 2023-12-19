# CaptainCaption: GPT-4-Vision Based Image Caption Generator

A gradio based image captioning tool that uses the GPT-4-Vision API to generate detailed descriptions of images.

## Features

- **Prompt Engineering**: Customize the prompt for image description to get the most accurate and relevant captions.
- **Batch Processing**: Ability to process an entire folder of images with customized pre and post prompts.

## Screenshot
![image](https://github.com/42lux/CaptainCaption/assets/7535793/523ca006-732c-4da2-9f92-c31ef59a0ea9)

## Installation
Clone repository
```
git clone https://github.com/42lux/CaptainCaption
```
Install requirements
```
pip install -r requirements.txt
```
## Usage

1. **Setting Up API Key**: Enter your OpenAI API key in the provided textbox.

2. **Uploading Images**: In the "Prompt Engineering" tab, upload the image for which you need a caption.

3. **Customizing the Prompt**: Customize the prompt, detail level, and max tokens according to your requirements.

4. **Generating Captions**: Click on "Generate Caption" to receive the image description.

5. **Batch Processing**: In the "GPT4-Vision Tagging" tab, you can process an entire folder of images. Set the folder path, prompt details, and the number of workers for processing.

## Running the Application

Run the script and navigate to the provided URL (Standard http://127.0.0.1:7860) by Gradio to access the interface.

## Limitations and Considerations

- The accuracy of captions depends on the quality of the uploaded images and the clarity of the provided prompts.
- The OpenAI API is rate-limited, so consider this when processing large batches of images.
- Internet connectivity is required for API communication.
