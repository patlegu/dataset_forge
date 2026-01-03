# Changelog & Roadmap - Dataset Forge

This document tracks the project's modification history as well as future improvement paths.

## 🚀 Roadmap (Future Evolutions)

### Modifications In Progress
- [ ] **Refresh Video List**: Add a fefresh button to update the list of the video to avoid closing the application.
- [ ] **Add Update Fonction in Batch Captioning**: Add an update function in Batch Captioning instead of make another reload of application.

### Short Term
- [ ] **Drag & Drop Support**: Allow dragging and dropping video/image folders directly into the interface.
- [ ] **Batch Queue**: Ability to run captioning/cropping tasks in the background on multiple folders sequentially.
- [ ] **Quality Filters**: Integration of an aesthetic or blur score to automatically sort extracted images.

### Long Term
- [ ] **Multi-Model Support**: Integration of alternatives to Florence-2 (e.g., JoyTag, WD14) for tagging.
- [ ] **Format Conversion**: Integrated tools to convert images (PNG/JPG/WEBP) before processing.
- [ ] **Executable Version**: Creation of a standalone `.exe` to facilitate installation on Windows without Python management.

---

## 📋 Version History

### [v1.7.6] - 2026-01-01

#### ✨ New Features
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