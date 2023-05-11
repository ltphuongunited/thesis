import gradio as gr
import trimesh

def load_mesh(img):
    return 'output/_output/000000.png', 'output/meshes/0001/000000.obj'

# Định nghĩa input và output
input_image = gr.inputs.Image(label="Input Image").style(height=400)
output_image = gr.outputs.Image(type="pil", label="Output Image").style(height=300)
output_3d_model = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model").style(height=200)

# Tạo giao diện
css = """
    <style>
        body {
            background-color: #ffffff;  /* Đặt màu nền là màu trắng */
        }
    </style>
"""
interface = gr.Interface(
    fn=load_mesh,
    inputs=input_image,
    outputs=[output_image, output_3d_model],
    css=css
)

interface.launch()