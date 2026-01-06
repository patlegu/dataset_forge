# Changelog & Roadmap - Dataset Forge

This document tracks the project's modification history as well as future improvement paths.

## 🚀 Roadmap (Future Evolutions)

### Modifications In Progress
- [ ] **Tag Cleaning**: Smart filtering of Florence-2 captions to remove "parasite words" (e.g., "image of", "seen from") for cleaner SDXL natural language prompts.
- [ ] **Add Update Function in Batch Captioning**: Add an update function in Batch Captioning instead of make another reload of application.

### Short Term
- [ ] **Distribution Analyzer**: Dashboard to visualize dataset statistics (aspect ratios, resolution distribution) to ensure variety (Bucketing check).
- [ ] **Kohya_ss Export**: One-click export to generate the folder structure (`img/10_trigger`) and `config.toml` for Kohya training.
- [ ] **Batch Queue**: Ability to run captioning/cropping tasks in the background on multiple folders sequentially.

### Long Term
- [ ] **Multi-Model Support**: Integration of alternatives to Florence-2 (e.g., JoyTag, WD14) for tagging.
- [ ] **Format Conversion**: Integrated tools to convert images (PNG/JPG/WEBP) before processing.
- [ ] **Executable Version**: Creation of a standalone `.exe` to facilitate installation on Windows without Python management.

---

## 📋 Version History

### [v1.10.0] - 2025-01-06

#### ✨ New Features
- **Background Removal**: Integrated `rembg` (u2net) to automatically remove backgrounds from images.
  - Supports multiple models: `u2net` (General), `u2net_human_seg` (Optimized for humans), `isnet-general-use`.
  - Generates transparent PNGs ready for high-fidelity training.
  - Drag & Drop support added for this new tab.

### [v1.9.1] - 2025-01-06

#### ✨ New Features
- **Refresh Button**: Added a "🔄 Refresh" button in the Video Extractor tab. Allows reloading the file list if files were added manually via Windows Explorer.

### [v1.9.0] - 2025-01-06

#### ✨ New Features
- **Drag & Drop Support**: Integrated `tkinterdnd2`. You can now drop a folder directly onto the window to load it into the active tab (Video, Crop, Caption, Edit).
                        - This feature works seamlessly on Windows when script launched from windows or from Linux when script lauch from Linux.
                        Not working in WSL Linux.

### [v1.8.0] - 2025-01-05

#### ✨ New Features
- **Blur Detection Filter**: Added OpenCV Laplacian variance check to automatically skip blurry frames during extraction.
- **Quality Filter UI**: Added a new "Advanced Quality Settings" section in the Video Extractor tab with a toggle and a Threshold slider.
- **Smart Reporting**: The extraction log now details how many blurry frames were rejected.

### [v1.7.6] - 2026-01-03

#### ✨ New Features
- Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.

#### 🔧 Maintenance & Improvements
- Docs: README.md installation part updated

#### 🔧 Others
- README.md manual eidt image update
- deleted images sizes
- README.MD in version 2.1
- Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.

### [v1.7.5] - 2026-01-01

#### ✨ New Features
- Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.

#### 🔧 Others
- Fix: 'Smart JSON Unescaper'. Backported from v1.9.
        Detects and cleans escaped JSON strings (e.g. "{\"prompt\":...) in
        video metadata to display readable ComfyUI workflows/prompts.

### [v1.7.4] - 2026-01-01

#### ✨ New Features
- Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.

#### 🔧 Others
- - Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.


### [v1.7.2] - 2026-01-01

#### ✨ New Features
- Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.

#### 🔧 Others
- - Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.

### [v1.7.1] - 2026-01-01

#### 🔧 Others
- - Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.


### [v1.7] - 2026-01-01

#### 🔧 Others
- - Feature: Added 'METADATA / INFO' button in Video Extractor tab. - Uses 'ffprobe' to extract deep video info and potential ComfyUI metadata tags.


### [v1.6.0] - 2026-01-01
#### ✨ New Features
- Feature: 'BATCH FIX FOLDER'. Scans the entire source folder, detects videos
   with odd dimensions, and auto-repairs them in bulk (FFmpeg + Smart Swap).

### [v1.5.4] - 2026-01-01
#### 🔧 Others
- UX Fix: 'Scroll Memory'. Keeps list position after refresh.

### [v1.5.3] - 2026-01-01
#### ✨ New Features
- Feature: 'Smart Swap'. Repaired files replace originals; originals go to trash.

### [v1.5] - 2026-01-01
#### 🔧 Others
- Fix: System FFmpeg call for WSL compatibility.
  Allows robust MP4 (H.264) encoding on WSL/Linux systems.

### [v1.4.0] - 2026-01-01
#### ✨ New Features
- Feature: Added 'REPAIR / FIX VIDEO' tool to auto-fix odd dimensions (swscaler errors).

### [v1.3.0] - 2026-01-01
#### 🔧 Others
- NEW: Interactive Treeview for ComfyUI Metadata (Explore Nodes & Params) - NEW: Export Workflow to .json (Drag & Drop compatible with ComfyUI) - FIX: Advanced suppression of 'swscaler' warnings

### [v1.2.0] - 2026-01-01
#### ✨ New Features
- Features: - NEW: "Pretty" Metadata Inspector with Syntax Highlighting for ComfyUI

### [v1.0.0] - Initial Release
*Public launch of Dataset Forge.*

#### ✨ New Features
- **Native Florence-2 Engine**: Complete integration via `transformers` (no Docker/Ollama required).
- **SDPA Patch**: Native support for PyTorch attention to avoid complex `flash_attn` installation on Windows.
- **Video Extractor (Dérushage)**:
  - Integrated player with frame-by-frame navigation.
  - **Auto-Fix**: Automatic detection and correction of videos with odd dimensions (fixes `swscaler` / FFmpeg crash).
- **Smart Cropping**:
  - Cropping by text prompt (e.g., "face").
  - **Bucket Resize (Mod 64)** option for optimal SDXL/Pony compatibility.
- **Captioning & Editing**:
  - Manual editor with shortcuts (`Ctrl+S`, `Alt+Arrows`).
  - Automatic injection of Trigger Words (Prefix/Suffix).

#### 🔧 Origin
- Initial fork based on Joschek's tools, rewritten to be a standalone all-in-one suite.