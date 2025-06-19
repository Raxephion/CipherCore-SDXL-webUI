# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 2024

@author: raxephion
Basic Stable Diffusion XL Gradio App with local/Hub models (GPU-Only)
Optimized for GPU usage with automatic FP16 and VRAM management.
Modified to download Hub models to local MODELS_DIR (set to "checkpoints").
"""

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
# Import commonly used schedulers
from diffusers import DDPMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler
import os
import sys
from PIL import Image
import time # Optional: for timing generation
import random # Needed for random seed
import numpy as np # Needed for MAX_SEED

# --- GPU Check and Initialization ---
# This application is GPU-only. Exit if CUDA is not available.
if not torch.cuda.is_available():
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ERROR: No NVIDIA GPU detected by PyTorch.        !!!")
    print("!!! This application requires a CUDA-enabled GPU.    !!!")
    print("!!! Please check your NVIDIA drivers and PyTorch     !!!")
    print("!!! installation, then restart the application.      !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    input("Press Enter to exit...") # Prevents the console from closing immediately
    sys.exit(1)

DEVICE = "cuda"
print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
if torch.cuda.device_count() > 0:
    print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")


# --- Configuration ---
MODELS_DIR = "checkpoints" # Local directory for storing/caching all models
# Standard SDXL resolutions
SUPPORTED_SDXL_SIZES = ["1024x1024", "1152x896", "896x1152", "1216x832", "832x1216", "1344x768", "768x1344", "1536x640", "640x1536"]

# Mapping of friendly scheduler names to their diffusers classes
SCHEDULER_MAP = {
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DDPM": DDPMScheduler,
    "LMS": LMSDiscreteScheduler,
}
DEFAULT_SCHEDULER = "Euler"

DEFAULT_HUB_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "SG161222/RealVisXL_V4.0",
    "RunDiffusion/Juggernaut-XL-v9",
    "Yntec/RealCartoon-XL-v7"
]

# --- Constants for UI / Generation ---
MAX_SEED = np.iinfo(np.int32).max

# --- Global state for the loaded pipeline ---
current_pipeline = None
current_model_id = None

# --- Helper function to list available local models ---
def list_local_models(models_dir_param):
    if not os.path.exists(models_dir_param):
        os.makedirs(models_dir_param)
        print(f"Created directory: {models_dir_param}")
        return []
    local_models = [os.path.join(models_dir_param, d) for d in os.listdir(models_dir_param)
                    if os.path.isdir(os.path.join(models_dir_param, d))]
    return local_models

# --- Image Generation Function ---
def generate_image(model_identifier, prompt, negative_prompt, steps, cfg_scale, scheduler_name, size, seed, num_images):
    global current_pipeline, current_model_id, SCHEDULER_MAP, MAX_SEED, MODELS_DIR, DEVICE

    if not model_identifier or model_identifier == "No models found":
        raise gr.Error(f"No model selected. Please add models to '{MODELS_DIR}' or check Hub IDs.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    num_images_int = int(num_images)
    if num_images_int <= 0:
         raise gr.Error("Number of images must be at least 1.")

    # Use FP16 for compatible GPUs for better performance, otherwise FP32
    dtype_to_use = torch.float16 if torch.cuda.get_device_capability(0)[0] >= 7 else torch.float32
    if dtype_to_use == torch.float16:
        print("Compatible GPU detected, using torch.float16 for better performance.")
    else:
        print("GPU may not fully support FP16, using torch.float32.")

    # Load or switch model if necessary
    if current_pipeline is None or current_model_id != model_identifier:
        print(f"Loading model: {model_identifier} onto {DEVICE}...")
        if current_pipeline is not None:
             print(f"Unloading previous model '{current_model_id}'...")
             del current_pipeline # De-reference to free memory
             torch.cuda.empty_cache() # Clear VRAM
             print("Cleared CUDA cache.")

        try:
            is_local_path = os.path.isdir(model_identifier)
            
            if is_local_path:
                 print(f"Loading local model from: {model_identifier}")
                 pipeline = StableDiffusionXLPipeline.from_pretrained(model_identifier, torch_dtype=dtype_to_use)
            else:
                 print(f"Loading Hub model: {model_identifier} (will cache to '{MODELS_DIR}')")
                 pipeline = StableDiffusionXLPipeline.from_pretrained(
                     model_identifier,
                     torch_dtype=dtype_to_use,
                     cache_dir=MODELS_DIR
                 )

            pipeline.to(DEVICE)
            current_pipeline = pipeline
            current_model_id = model_identifier
            print(f"Model '{model_identifier}' loaded successfully on {DEVICE}.")

        except Exception as e:
            current_pipeline, current_model_id = None, None
            error_message_lower = str(e).lower()
            if "out of memory" in error_message_lower:
                 raise gr.Error(f"Out of VRAM loading model. Your GPU may not have enough memory for this model. Error: {e}")
            else:
                raise gr.Error(f"Failed to load model '{model_identifier}': {e}")

    if current_pipeline is None:
         raise gr.Error("Model failed to load. Cannot generate image.")

    # Configure the scheduler
    selected_scheduler_class = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP[DEFAULT_SCHEDULER])
    current_pipeline.scheduler = selected_scheduler_class.from_config(current_pipeline.scheduler.config)
    print(f"Scheduler set to: {scheduler_name}")

    # Parse image size
    try:
        w_str, h_str = size.split('x')
        width, height = int(w_str), int(h_str)
    except ValueError:
        raise gr.Error(f"Invalid size format: '{size}'. Use 'WidthxHeight' (e.g., 1024x1024).")

    # Prepare generator for seeding
    seed_int = int(seed)
    if seed_int == -1:
        seed_int = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(seed_int)

    print(f"Generating {num_images_int} image(s) with seed {seed_int}...")
    start_time = time.time()

    try:
        output = current_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg_scale),
            generator=generator,
            num_images_per_prompt=num_images_int,
        )
        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        return output.images, seed_int

    except Exception as e:
        error_message_lower = str(e).lower()
        if "out of memory" in error_message_lower:
             raise gr.Error(f"Out of VRAM during generation. Try a smaller image size, fewer images, or a less demanding model. Error: {e}")
        else:
             raise gr.Error(f"Image generation failed: {e}")

# --- Gradio Interface ---
local_models_list = list_local_models(MODELS_DIR)
model_choices = local_models_list + DEFAULT_HUB_MODELS

if not model_choices:
    initial_model_choices = ["No models found"]
    initial_default_model = "No models found"
    model_dropdown_interactive = False
else:
    initial_model_choices = model_choices
    if "stabilityai/stable-diffusion-xl-base-1.0" in model_choices:
         initial_default_model = "stabilityai/stable-diffusion-xl-base-1.0"
    elif local_models_list:
         initial_default_model = local_models_list[0]
    else:
         initial_default_model = model_choices[0]
    model_dropdown_interactive = True

scheduler_choices = list(SCHEDULER_MAP.keys())

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# CipherCore Stable Diffusion XL Generator (GPU-Only)")
    gr.Markdown(f"Create images with SDXL models. Models from Hugging Face will be cached in the `./{MODELS_DIR}` directory.")

    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(choices=initial_model_choices, value=initial_default_model, label="Select Model", interactive=model_dropdown_interactive)
            prompt_input = gr.Textbox(label="Positive Prompt", placeholder="e.g., an astronaut riding a horse on mars, cinematic, dramatic", lines=3, autofocus=True)
            negative_prompt_input = gr.Textbox(label="Negative Prompt (Optional)", placeholder="e.g., blurry, low quality, deformed, watermark", lines=2)

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(minimum=5, maximum=100, value=25, label="Inference Steps", step=1)
                    cfg_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.0, label="CFG Scale", step=0.1)
                with gr.Row():
                     scheduler_dropdown = gr.Dropdown(choices=scheduler_choices, value=DEFAULT_SCHEDULER, label="Scheduler")
                     size_dropdown = gr.Dropdown(choices=SUPPORTED_SDXL_SIZES, value="1024x1024", label="Image Size")
                with gr.Row():
                     seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                     num_images_slider = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")

            generate_button = gr.Button("✨ Generate Image ✨", variant="primary")

        with gr.Column(scale=3):
            output_gallery = gr.Gallery(label="Generated Images", show_label=True, interactive=False, columns=2)
            actual_seed_output = gr.Number(label="Actual Seed Used", precision=0, interactive=False)

    generate_button.click(
        fn=generate_image,
        inputs=[model_dropdown, prompt_input, negative_prompt_input, steps_slider, cfg_slider, scheduler_dropdown, size_dropdown, seed_input, num_images_slider],
        outputs=[output_gallery, actual_seed_output]
    )
    
    gr.Markdown(f"--- \n **Note:** The first time you load a model, it will be downloaded and loaded into VRAM, which may take several minutes.")

if __name__ == "__main__":
    print("\n--- Starting CipherCore Stable Diffusion XL Generator (GPU-Only) ---")
    print(f"Models will be loaded from/cached to: {os.path.abspath(MODELS_DIR)}")
    demo.launch(show_error=True, inbrowser=True)
    print("Gradio interface closed.")
