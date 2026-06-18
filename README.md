# SDXL Safetensors to Diffusers Converter

A clean, standalone desktop application built with Python and PyQt6 to convert single Stable Diffusion XL (SDXL) `.safetensors` checkpoints into the unrolled, multi-folder **Diffusers** pipeline format. 

This tool is specifically designed to be safe for environments running local AI web UIs (like ComfyUI), ensuring package loading conflicts are completely bypassed.

---

## ✨ Features

* **Graphical User Interface (GUI):** Easy point-and-click file and folder selection with a real-time progress bar and log terminal.
* **One-Click Windows Launcher:** Includes an automated batch script (`run_windows.bat`) that handles environment verification, automatic package installation/upgrades, and script launching seamlessly.
* **ComfyUI & Windows Isolation:** Automatically forces problematic global modules (`xformers`, `peft`, `torchaudio`) to `None` inside the runtime context. This guarantees that your local environment's custom DLLs or specific ComfyUI dependencies are completely safe and untouched.
* **Smart Memory Recovery (Meta-Tensor Patching):** Automatically catches, materializes, and fixes any unassigned `meta` device parameters or empty buffers that often break base `diffusers` conversion methods.
* **Low VRAM / RAM Support:** Includes an option to enforce **Half Precision (fp16)** conversion to drastically reduce required CPU/GPU memory overhead and prevent Out of Memory (OOM) crashes.
* **Flexible Format Output:** Choose whether your unrolled diffusers text encoders, VAE, and UNet modules save as standard weights or `.safetensors` sub-files.

---

## 🛠️ How to Setup and Run (Windows)

The absolute easiest way to get started is using the automated batch runner:

1. Place `sdxl_converter.py` and `run_windows.bat` in the same directory.
2. Double-click **`run_windows.bat`**.
3. The script will automatically:
   * Verify your Python installation.
   * Check and safely upgrade all required packages (`torch`, `diffusers`, `transformers`, `accelerate`, `PyQt6`) to stable versions.
   * Launch the Converter GUI.

---

## 💻 Manual Prerequisites & Execution

If you prefer to manage your python environments manually (or are running on Linux/macOS):

### 1. Installation
Install or upgrade the required deep learning and UI dependencies using your terminal:
```bash
pip install --upgrade torch torchvision diffusers transformers accelerate PyQt6

```

> **Note:** If you are running this alongside a ComfyUI or WebUI setup, you can run the tool within that specific virtual environment (`venv`) to leverage your existing PyTorch installation.

### 2. Execution

Launch the application by running the script directly:

```bash
python sdxl_converter.py

```

---

## 📖 How to Use

1. **Browse Safetensors...** – Click to select the single SDXL `.safetensors` checkpoint you want to convert.
2. **Set Output Folder...** – Select the main directory where you want your diffusers pipeline folders to live. The tool will automatically create a dedicated sub-folder named after your model checkpoint.
3. **Configure Options:**
* **Use Half Precision (fp16):** Highly recommended. Significantly reduces memory usage during the unrolling process.
* **Output as Safetensors format:** Ensures the individual sub-modules inside the final pipeline directory use safe serialization (`.safetensors` instead of old `.bin` weights).


4. **Start Conversion** – Click the button and watch the live progress log.

Once complete, the generated directory can be loaded instantly via the standard Diffusers library:

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained("path/to/your/output_folder")

```

---

## 💡 Troubleshooting & Common Issues

* **Out of GPU Memory (VRAM) / CPU RAM:** SDXL checkpoints are massive. If the script crashes during the `from_single_file` loading sequence, close memory-heavy applications (like your web UI or browser tabs) and ensure **Use Half Precision (fp16)** is checked.
* **Python Path Error:** If `run_windows.bat` throws a path error, ensure that Python 3.10+ is correctly installed from python.org and that the "Add Python to PATH" option was checked during installation.

```
