### - 19-06-2025: App still very early development

# CipherCore SDXL - FAST Stable Diffusion XL Local Image Generator Web UI (GPU Optimized)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to the CipherCore Stable Diffusion XL Generator! This user-friendly Gradio web application allows you to effortlessly generate high-resolution images using various Stable Diffusion XL (SDXL) models, running for free locally on your own PC. Whether you have local models or prefer popular ones from the Hugging Face Hub, this tool provides a simple, optimized interface to unleash your creativity on your NVIDIA GPU.

This project is designed for **Windows** users seeking a simple experience through easy-to-use batch files for a GPU-powered setup.

## Application Screenshot:
![Screenshot of the CipherCore Stable Diffusion XL UI](images/ciphercore01.png)

---

## ✨ Features
- **Flexible Model Selection:**
  - Load your own Stable Diffusion XL models (in `diffusers` format) from a local `./checkpoints` directory.
  - Access a curated list of popular SDXL models directly from the app (models are downloaded and cached locally on first use).
- **GPU Optimized:**
  - Designed to leverage your **NVIDIA GPU** for the fastest possible generation.
  - Automatic `float16` precision for compatible cards to improve speed and reduce VRAM usage.
  - Can still run on **CPU** if a GPU is not selected, but be warned: SDXL on a CPU is extremely slow and primarily for testing purposes.
- **Comprehensive Control:**
  - **Positive & Negative Prompts:** Guide the AI with detailed descriptions of what you want (and don’t want).
  - **Inference Steps:** Control the number of denoising steps to balance quality and speed.
  - **CFG Scale:** Adjust how strongly the image should conform to your prompt.
  - **Schedulers:** Experiment with different sampling algorithms (Euler, DPM++ 2M, DDPM, LMS) to find the perfect result.
  - **Image Sizes:** Choose from a range of standard high-resolution SDXL sizes (e.g., 1024x1024, 896x1152).
  - **Seed Control:** Set a specific seed for reproducible results, or use -1 to let the AI surprise you.
- **User-Friendly Interface:**
  - Clean and intuitive Gradio UI.
  - Organized controls with advanced settings tucked away neatly.
  - Direct image gallery with download and share options.
- **Safety First (Note):** The built-in safety checker is disabled to allow for maximum creative freedom. Please use this tool responsibly.

---

## 🔥 Why CipherCore?
While other UIs pack in every feature imaginable, **CipherCore is built for simplicity and performance.**

SDXL is demanding, but this interface is designed to be as lightweight as possible. It gets out of the way so your hardware can focus on one thing: generating images quickly.

- **Lean and Fast:** No background bloat, unnecessary extensions, or complex dependencies to slow things down.
- **Efficient Memory Management:** The app actively manages VRAM, unloading old models and clearing the cache when you switch to a new one.
- **Optimized for Lower-VRAM GPUs:** With automatic FP16 and a lightweight architecture, it's designed to give users with 8GB cards a fighting chance at generating beautiful 1024x1024 images.
- **Direct and to the Point:** If you just want to load a model, write a prompt, and get a great image without a dozen dropdowns, this is the tool for you.

---

## 🚀 Prerequisites
- **Windows Operating System:** The provided batch files (`.bat`) are for Windows.
- **Python:** 3.9 or higher. Ensure Python is installed and added to your system's PATH. You can download it from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
- **Git:** Recommended for easy updates, but not required if you download the ZIP.
- **Hardware:**
  - **NVIDIA GPU is strongly recommended.** SDXL is resource-intensive.
  - A minimum of **8GB VRAM** is recommended for generating 1024x1024 images. Users with 12GB+ VRAM will have a much smoother experience.
  - A modern CPU. While CPU-only generation is possible, it is not practical for regular use.
  - **Important:** You must have up-to-date NVIDIA drivers. Use the `nvidia-smi` command in your terminal to check your driver's CUDA version compatibility. This is critical for the setup to succeed.
- **Internet Connection:** Required for the initial setup and for downloading models from Hugging Face Hub.

---

## 📦 Easy Setup (Windows - Download & Run)
This is the recommended method for Windows users with a compatible NVIDIA GPU.

1.  **Download the project:**
    *   Go to the GitHub repository page: `https://github.com/Raxephion/CipherCore-SDXL-WebUI` (Note: URL updated for example)
    *   Click the green "Code" button and select "Download ZIP".
2.  **Extract the ZIP:** Extract the file to a permanent location on your computer (e.g., your Documents folder). This will create a folder like `CipherCore-SDXL-WebUI-main`. You can rename it.
3.  **Run the Setup Script:**
    *   Navigate into the extracted folder.
    *   **Double-click `setup.bat`**.
    *   A command prompt will open and automatically create a Python virtual environment (`venv`) and install all necessary dependencies, including the GPU version of PyTorch. This may take several minutes.
    *   **Please read the output in the command window.** If the installation fails, it will provide troubleshooting steps, which usually involve checking your NVIDIA drivers.
4.  **Prepare Local Models (Optional):**
    *   Inside the project folder, you will find a `checkpoints` directory.
    *   Place your Stable Diffusion XL models (in `diffusers` format – each model is a folder) inside this `checkpoints` directory.

---

## ▶️ Running the Application (Windows)
Once setup is complete, launching the app is simple:

1.  Navigate to the project folder.
2.  **Double-click the `run.bat` file.**
3.  A command prompt window will open, activate the environment, and start the application.
4.  Your default web browser should automatically open to the local URL (usually `http://127.0.0.1:7860`).

---

## 🛠️ Manual Setup & Running (Advanced Users / Other Platforms)
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Raxephion/CipherCore-SDXL-WebUI.git
    cd CipherCore-SDXL-WebUI
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:** The `requirements.txt` is configured for NVIDIA GPUs (CUDA 12.1).
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the App:**
    ```bash
    python main.py
    ```

---

## ⚙️ Uninstall
Simply **delete the entire project folder.** That's it. No installers, no registry entries.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.
