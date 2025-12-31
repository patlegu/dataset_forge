# ⚒️ Dataset Forge

**The All-in-One Local GUI for AI Dataset Preparation.** *Powered by Microsoft Florence-2 (Native Integration)*

![Dataset Forge Banner](dataset_forge_banner.jpg)

## 📖 Overview

**Dataset Forge** is a standalone, local Python application designed to streamline the creation of image datasets for AI training (LoRA, Checkpoints, etc.).

It started as a fork of [Joschek's tools](https://civitai.com/articles/24233) but evolved into a complete suite focused on **video extraction, smart processing, and independence**. Unlike other tools, Dataset Forge removes complex dependencies (no Docker, no Ollama) and runs **Microsoft's Florence-2** vision models natively via Hugging Face Transformers.

## ✨ Key Features

### 1. 🧠 Native Florence-2 Engine
- **No external APIs:** Runs locally using `transformers` and `torch`.
- **Dual Models:** Support for `Florence-2-Base` (Speed) and `Florence-2-Large` (Accuracy).
- **Optimized:** Runs in **FP16** mode to save VRAM and includes a custom patch for **SDPA** (Scaled Dot Product Attention) to avoid "Flash Attention" installation headaches on Windows.

### 2. 🎬 Robust Video Extractor ("The Dérushage Station")
- **Integrated Browser:** Browse video folders directly in the app.
- **Smart Player:** Frame-by-frame navigation to find perfect poses without motion blur.
- **Auto-Fix (Anti-Crash):** Automatically detects and crops videos with **odd dimensions** (e.g., 1919x1080) to prevent `swscaler` errors and FFmpeg crashes.
- **Batch Extraction:** Extract all frames or snapshot specific moments.

### 3. ✂️ Smart Cropping & Bucketing
- **Text-to-Crop:** Describe the object ("face", "hand", "dog") and the AI will find and crop it.
- **Bucket-Friendly:** Includes a **"Bucket Resize (Mod 64)"** mode. It ensures all cropped images have dimensions divisible by 64 (e.g., 1024x768), which is critical for optimal **SDXL** and **Pony** training.

### 4. 🏷️ Batch Captioning & Editing
- **Trigger Words:** Auto-inject prompts (Prefix/Suffix) into captions.
- **Manual Editor:** A split-view Quality Control station with keyboard shortcuts (`Ctrl+S`, `Alt+Arrows`) for rapid review.

---

## ⚙️ Installation

### Prerequisites
- **Python 3.10** or newer.
- A GPU with CUDA support (NVIDIA) is highly recommended.
- [Git](https://git-scm.com/) installed.

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/patlegu/dataset_forge.git
   cd dataset-forge
   ```
2. **Create a virtual environment (Recommended):**
  ```bash
      python -m venv venv
      source venv/bin/activate
  ```
3. **Install Dependencies:**
Pytorch can be found at [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)  
  ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers pillow opencv-python numpy psutil timm einops
    (Note: timm and einops are required for Florence-2 model loading).
  ```
4. **Run the App:**
```Bash
python dataset_forge.py
```

## 🛠️ Usage Guide
### Video Extractor:
- Select a source folder containing videos.
- Use the slider or **< >** buttons to navigate.
- Click **SNAPSHOT** to save the current frame, or **EXTRACT ALL** to dump the whole video frames.

### Smart Cropping:
- Choose your extracted frames folder as Input.  
- Set a target prompt (e.g., "face").  
- Select "Bucket Resize (Mod 64)" for training prep.  

### Batch Captioning:
- Point to your images.  
- Select <DETAILED_CAPTION> (Best for SDXL/Pony).  
- Add your Trigger Word in "Prefix" (e.g., ohwx man).  

### Manual Edit:
- Review your captions.  
- Use Ctrl+S to save changes instantly.

## 🐞 Troubleshooting & Known Issues
Here are common issues we encountered during development and how they are handled:

### 1. [swscaler @ ...] Slice parameters 0, xxx are invalid
**Context**: This happens when trying to display or process a video with odd dimensions (e.g., width is 1279px instead of 1280px). FFmpeg/OpenCV hates odd numbers for YUV conversion.

**Solution**: Dataset Forge automatically detects this and crops 1 pixel from the right or bottom on-the-fly. You don't need to do anything; the preview and extraction will work seamlessly.

### 2. "Flash Attention" Warnings
** Context**: You might see warnings in the console about flash_attn not being installed.
**Solution**: This is normal. We intentionally patched the model to use standard PyTorch attention (SDPA) to make installation easier on Windows. Performance is still excellent.

### 3. First Launch is Slow
**Context**: The app freezes for a moment when clicking "Load Engine".
**Solution**: The first time you run it, the app downloads the Florence-2 models (approx 1-3GB) from Hugging Face. Check your console/terminal to see the download progress.

📜 License & Credits
**Author**: Patlegu  
**Original Concept**: Based on Joschek's tools.  
**Model**: Uses Microsoft Florence-2.  
**Project released under MIT License.** 