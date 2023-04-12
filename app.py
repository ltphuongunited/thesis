import gradio as gr
import torch
from torchvision import transforms
import cv2

# Define a function to predict an image using your PyTorch model
def predict(img):
    return cv2.imread('output/_output/000000.png')

# Define your input component
input_image = gr.inputs.Image()

# Define your output component
output_image = gr.outputs.Image(type="pil")

# Create a Gradio interface
gr.Interface(fn=predict, inputs=input_image, outputs=output_image).launch()
