import gradio as gr
import cv2

def process_image(input_image):
    # Xử lý ảnh đầu vào để tạo ra ảnh đầu ra
    # Ví dụ: chuyển đổi ảnh thành ảnh xám
    output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return output_image

# Tạo một giao diện người dùng với một input là ảnh và một output là ảnh
input_image = gr.inputs.Image()
output_image = gr.outputs.Image()
interface = gr.Interface(fn=process_image, inputs=input_image, outputs=output_image)

# Chạy giao diện người dùng
interface.launch()
