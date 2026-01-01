#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATASET FORGE v1.2
(Formerly Joschek Fork)
A complete, local suite for AI Dataset preparation.

Features:
- NEW: "Pretty" Metadata Inspector with Syntax Highlighting for ComfyUI
- FIX: Advanced suppression of 'swscaler' warnings
- Native Florence-2 (Base/Large)
- Video Frame Extraction
- Smart Cropping (Bucket-Friendly)
- Batch Captioning
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser, simpledialog, scrolledtext
import threading
import os
import sys
import time
import json
import gc 
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# Machine Learning Imports
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageTk, PngImagePlugin
import cv2
import numpy as np

# Suppress FFmpeg/OpenCV noisy logs
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# ---------------- CONFIGURATION ----------------
APP_NAME = "Dataset Forge"
VERSION = "1.2"
CONFIG_FILE = Path.home() / ".config" / "dataset_forge.json"

MODELS = {
    "Florence-2 Base (Fast)": "microsoft/Florence-2-base",
    "Florence-2 Large (Best)": "microsoft/Florence-2-large"
}

DEFAULTS = {
    "model_name": "Florence-2 Base (Fast)",
    "crop_prompt": "face", 
    "crop_method": "pad", 
    "pad_color": "#000000",
    "caption_mode": "<DETAILED_CAPTION>", 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "last_batch_dir": "",
    "last_crop_in": "",
    "last_crop_out": "",
    "last_video_source": "",
    "last_video_output": "",
    "last_editor_dir": ""
}

# ---------------- THEME ----------------
BG = "#1e1e2e" 
CARD = "#313244"
INPUT = "#45475a"
TEXT = "#cdd6f4"
DIM = "#a6adc8"
ACCENT = "#89b4fa" 
SUCCESS = "#a6e3a1" 
WARNING = "#fab387" 
ERROR = "#f38ba8"   
HIGHLIGHT = "#cba6f7" 

# ---------------- CORE: FLORENCE ENGINE ----------------
class FlorenceEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = DEFAULTS["device"]
        self.current_model_id = None
        self.is_loaded = False

    def load(self, model_key, status_callback):
        target_id = MODELS[model_key]
        if self.is_loaded and self.current_model_id == target_id:
            status_callback("Ready.")
            return
        if self.is_loaded: self.unload()

        status_callback(f"Loading {model_key}...")
        
        def fixed_get_imports(filename):
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            if "flash_attn" in imports: imports.remove("flash_attn")
            return imports
        
        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                self.model = AutoModelForCausalLM.from_pretrained(
                    target_id, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=dtype 
                ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                target_id, trust_remote_code=True, clean_up_tokenization_spaces=True
            )
            self.current_model_id = target_id
            self.is_loaded = True
            status_callback("Ready.")
        except Exception as e:
            status_callback(f"Error: {e}")
            print(e)

    def unload(self):
        if self.model: del self.model; self.model = None
        if self.processor: del self.processor; self.processor = None
        self.is_loaded = False
        self.current_model_id = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def run_task(self, image: Image.Image, task_prompt: str, text_input: str = None) -> Any:
        if not self.is_loaded: return None
        prompt = task_prompt
        if text_input: prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        if self.device == "cuda": inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=False, num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        return parsed[task_prompt]

# ---------------- HELPER: ODD DIM FIX ----------------
def fix_odd_dims(frame):
    if frame is None: return None
    h, w = frame.shape[:2]
    trim_h = h % 2; trim_w = w % 2
    if trim_h > 0 or trim_w > 0: return frame[:h-trim_h, :w-trim_w]
    return frame

# ---------------- GUI ----------------
class ForgeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"{APP_NAME} v{VERSION}")
        root.geometry("1280x900")
        root.configure(bg=BG)
        root.option_add("*Font", ("Segoe UI", 10))
        root.protocol("WM_DELETE_WINDOW", self.on_close) 

        self.engine = FlorenceEngine()
        self.is_running = False 
        self.config = self.load_config()

        # State
        self.editor_files = []
        self.current_editor_index = -1
        self.current_editor_img_obj = None 
        
        self.cap = None
        self.video_path = None
        self.video_total_frames = 0
        self.video_output_dir = None
        self.video_source_dir = None
        self.video_files_cache = []
        self.current_video_frame = None
        self.current_frame_pos = 0

        self.style_setup()
        self.build_ui()
        self.bind_shortcuts()
        self.monitor_resources() 
        
    def style_setup(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TProgressbar", background=HIGHLIGHT, troughcolor=BG, borderwidth=0)
        s.configure("TCombobox", fieldbackground=INPUT, background=INPUT, foreground=TEXT, arrowcolor=TEXT)
        self.root.option_add("*TCombobox*Listbox*Background", CARD)
        self.root.option_add("*TCombobox*Listbox*Foreground", TEXT)
        s.configure("Horizontal.TScale", background=BG, troughcolor=CARD)

    def bind_shortcuts(self):
        self.root.bind("<Control-s>", lambda e: self.editor_save())
        self.root.bind("<Alt-Right>", lambda e: self.editor_nav(1))
        self.root.bind("<Alt-Left>", lambda e: self.editor_nav(-1))

    # --- CONFIG ---
    def load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for k, v in DEFAULTS.items():
                        if k not in data: data[k] = v
                    return data
            except: pass
        return DEFAULTS.copy()

    def save_config(self):
        try:
            self.config["last_batch_dir"] = self.batch_dir.get()
            self.config["last_crop_in"] = self.crop_in.get()
            self.config["last_crop_out"] = self.crop_out.get()
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e: print(f"Config Error: {e}")

    # --- UI BUILDING ---
    def build_ui(self):
        main = tk.Frame(self.root, bg=BG); main.pack(fill="both", expand=True)
        
        # Header
        header = tk.Frame(main, bg=BG); header.pack(side="top", fill="x", padx=20, pady=15)
        hl = tk.Frame(header, bg=BG); hl.pack(side="left")
        tk.Label(hl, text=APP_NAME, bg=BG, fg=ACCENT, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        self.res_frame = tk.Frame(hl, bg=BG); self.res_frame.pack(anchor="w", pady=(5,0))
        self.lbl_ram = tk.Label(self.res_frame, text="RAM: ...", bg=BG, fg=WARNING, font=("Consolas", 9))
        self.lbl_ram.pack(side="left", padx=(0, 15))
        self.lbl_vram = tk.Label(self.res_frame, text="VRAM: ...", bg=BG, fg=WARNING, font=("Consolas", 9))
        self.lbl_vram.pack(side="left")

        hr = tk.Frame(header, bg=BG); hr.pack(side="right")
        self.model_var = tk.StringVar(value=DEFAULTS["model_name"])
        ttk.Combobox(hr, textvariable=self.model_var, values=list(MODELS.keys()), state="readonly", width=25).pack(side="left", padx=10)
        self.lbl_status = tk.Label(hr, text="Engine Offline", bg=BG, fg=DIM)
        self.lbl_status.pack(side="left", padx=10)
        self.btn_load = tk.Button(hr, text="Load Engine", bg=ACCENT, fg=BG, bd=0, font=("Segoe UI", 9, "bold"), command=self.toggle_engine)
        self.btn_load.pack(side="left", padx=5, ipady=5, ipadx=10)
        tk.Button(hr, text="Quit", bg=ERROR, fg=BG, bd=0, font=("Segoe UI", 9, "bold"), command=self.on_close).pack(side="left", padx=5, ipady=5, ipadx=10)

        # Tabs
        self.content = tk.Frame(main, bg=BG); self.content.pack(side="bottom", fill="both", expand=True)
        self.tabs = {}; self.btns = {}; self.curr_tab = None
        bar = tk.Frame(main, bg=BG); bar.pack(side="top", fill="x", padx=20)
        
        for n in ["Video Extractor", "Smart Cropping", "Batch Captioning", "Manual Edit"]:
            self.tabs[n] = tk.Frame(self.content, bg=BG)
            b = tk.Label(bar, text=n, bg=BG, fg=DIM, font=("Segoe UI", 10, "bold"), cursor="hand2", padx=15, pady=10)
            b.pack(side="left"); b.bind("<Button-1>", lambda e, x=n: self.switch_tab(x))
            self.btns[n] = b
        tk.Frame(main, bg=INPUT, height=2).pack(side="top", fill="x") 
        
        self.build_video_tab(); self.build_crop_tab(); self.build_batch_tab(); self.build_editor_tab()
        self.switch_tab("Video Extractor")

    def monitor_resources(self):
        try:
            vm = psutil.virtual_memory()
            used_gb = (vm.total - vm.available) / (1024**3); total_gb = vm.total / (1024**3)
            self.lbl_ram.config(text=f"RAM: {used_gb:.1f}/{total_gb:.0f} GB")
            if torch.cuda.is_available():
                res_gb = torch.cuda.memory_reserved(0) / (1024**3); tot_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.lbl_vram.config(text=f"VRAM: {res_gb:.1f}/{tot_vram:.0f} GB")
            else: self.lbl_vram.config(text="GPU: N/A")
        except: pass
        self.root.after(1000, self.monitor_resources)

    def on_close(self):
        self.save_config(); self.is_running = False
        if self.cap: self.cap.release()
        self.engine.unload()
        self.root.destroy()

    def switch_tab(self, name):
        if self.curr_tab: self.tabs[self.curr_tab].pack_forget(); self.btns[self.curr_tab].config(fg=DIM, bg=BG)
        self.curr_tab = name; self.tabs[name].pack(fill="both", expand=True); self.btns[name].config(fg=HIGHLIGHT, bg=BG)

    # --- ENGINE ---
    def toggle_engine(self):
        selected_model = self.model_var.get()
        self.btn_load.config(state="disabled", text="Loading...")
        threading.Thread(target=self.engine.load, args=(selected_model, self.update_load_status,), daemon=True).start()
    
    def update_load_status(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))
        if msg == "Ready.":
            self.root.after(0, lambda: [self.btn_load.pack_forget(), self.lbl_status.config(fg=SUCCESS, text=f"Online: {self.model_var.get()}")])

    # ==========================
    # TAB: VIDEO EXTRACTOR
    # ==========================
    def build_video_tab(self):
        f = self.tabs["Video Extractor"]
        t = tk.Frame(f, bg=BG); t.pack(fill="x", padx=10, pady=10)
        tk.Button(t, text="Select Source Folder", bg=ACCENT, fg=BG, bd=0, command=self.vid_browse_src).pack(side="left", padx=5, ipady=5)
        tk.Button(t, text="Set Output Folder", bg=CARD, fg=TEXT, bd=0, command=self.vid_browse_out).pack(side="left", padx=5, ipady=5)
        self.lbl_vid_info = tk.Label(t, text="No Video", bg=BG, fg=DIM); self.lbl_vid_info.pack(side="left", padx=15)

        pane = tk.Frame(f, bg=BG); pane.pack(fill="both", expand=True, padx=10, pady=5)
        left = tk.Frame(pane, bg=BG, width=320); left.pack(side="left", fill="y", padx=(0,10))
        
        vf = tk.LabelFrame(left, text=" Videos ", bg=BG, fg=TEXT, font=("Segoe UI", 9, "bold"))
        vf.pack(side="top", fill="both", expand=True, pady=(0,10))
        self.vid_search = tk.Entry(vf, bg=INPUT, fg=TEXT, bd=0); self.vid_search.pack(fill="x", padx=5, pady=5)
        self.vid_search.bind("<KeyRelease>", self.vid_filter)
        self.lst_vid = tk.Listbox(vf, bg=CARD, fg=TEXT, bd=0, highlightthickness=0, selectbackground=HIGHLIGHT)
        self.lst_vid.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.lst_vid.bind('<<ListboxSelect>>', self.vid_select)
        sb1 = tk.Scrollbar(vf, command=self.lst_vid.yview); sb1.pack(side="right", fill="y"); self.lst_vid.config(yscrollcommand=sb1.set)

        sf = tk.LabelFrame(left, text=" Extracted Frames ", bg=BG, fg=HIGHLIGHT, font=("Segoe UI", 9, "bold"))
        sf.pack(side="bottom", fill="both", expand=True)
        self.lst_frames = tk.Listbox(sf, bg=CARD, fg=TEXT, bd=0, highlightthickness=0)
        self.lst_frames.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        sb2 = tk.Scrollbar(sf, command=self.lst_frames.yview); sb2.pack(side="right", fill="y"); self.lst_frames.config(yscrollcommand=sb2.set)

        right = tk.Frame(pane, bg=BG); right.pack(side="left", fill="both", expand=True)
        self.vid_canvas = tk.Label(right, bg="black", text="Select a video", fg="white")
        self.vid_canvas.pack(fill="both", expand=True)
        
        ctrl = tk.Frame(right, bg=BG); ctrl.pack(fill="x", pady=10)
        self.vid_slider = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal", command=self.vid_slide)
        self.vid_slider.pack(fill="x", pady=5)
        
        c_btns = tk.Frame(ctrl, bg=BG); c_btns.pack()
        tk.Button(c_btns, text="<< -1s", bg=CARD, fg=TEXT, bd=0, command=lambda: self.vid_step(-30)).pack(side="left", padx=2)
        tk.Button(c_btns, text="< -1fr", bg=CARD, fg=TEXT, bd=0, command=lambda: self.vid_step(-1)).pack(side="left", padx=2)
        self.lbl_vid_cnt = tk.Label(c_btns, text="0 / 0", bg=BG, fg=TEXT, width=15); self.lbl_vid_cnt.pack(side="left", padx=5)
        tk.Button(c_btns, text="+1fr >", bg=CARD, fg=TEXT, bd=0, command=lambda: self.vid_step(1)).pack(side="left", padx=2)
        tk.Button(c_btns, text="+1s >>", bg=CARD, fg=TEXT, bd=0, command=lambda: self.vid_step(30)).pack(side="left", padx=2)
        
        tk.Button(c_btns, text="SNAPSHOT", bg=SUCCESS, fg=BG, bd=0, font=("Segoe UI", 9, "bold"), command=self.vid_snapshot).pack(side="left", padx=20)
        self.btn_extract_all = tk.Button(c_btns, text="EXTRACT ALL", bg=HIGHLIGHT, fg=BG, bd=0, font=("Segoe UI", 9, "bold"), command=self.vid_extract_all)
        self.btn_extract_all.pack(side="left", padx=5)
        self.vid_progress = ttk.Progressbar(right, mode="determinate"); self.vid_progress.pack(fill="x", padx=10, pady=5)

        if self.config.get("last_video_source"):
            self.video_source_dir = Path(self.config["last_video_source"]); self.vid_refresh_files()
        if self.config.get("last_video_output"):
            self.video_output_dir = Path(self.config["last_video_output"])

    def vid_browse_src(self):
        d = filedialog.askdirectory(initialdir=self.config.get("last_video_source", "/"))
        if d: self.video_source_dir = Path(d); self.config["last_video_source"] = d; self.save_config(); self.vid_refresh_files()

    def vid_browse_out(self):
        d = filedialog.askdirectory(initialdir=self.config.get("last_video_output", "/"))
        if d: self.video_output_dir = Path(d); self.config["last_video_output"] = d; self.save_config(); self.vid_refresh_saved()

    def vid_refresh_files(self):
        self.lst_vid.delete(0, tk.END); self.video_files_cache = []
        if not self.video_source_dir: return
        for ext in ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"]:
            self.video_files_cache.extend(list(self.video_source_dir.glob(ext)))
        self.video_files_cache.sort(); self.vid_filter(None)

    def vid_filter(self, event):
        q = self.vid_search.get().lower(); self.lst_vid.delete(0, tk.END)
        for p in self.video_files_cache:
            if q in p.name.lower(): self.lst_vid.insert(tk.END, p.name)

    def vid_select(self, event):
        sel = self.lst_vid.curselection()
        if not sel: return
        self.vid_load(self.video_source_dir / self.lst_vid.get(sel[0]))

    def vid_load(self, path):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(str(path))
        self.video_path = path; self.video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_pos = 0; self.vid_slider.configure(to=self.video_total_frames-1)
        self.lbl_vid_info.config(text=f"{path.name} ({self.video_total_frames} fr)")
        if not self.video_output_dir:
            self.video_output_dir = path.parent / "extracted"; self.video_output_dir.mkdir(exist_ok=True)
        self.vid_refresh_saved(); self.vid_seek(0, update_slider=True)

    def vid_slide(self, val):
        if self.cap: self.vid_seek(int(float(val)), update_slider=False)

    def vid_step(self, delta):
        self.vid_seek(self.current_frame_pos + delta, update_slider=True)

    def vid_seek(self, frame_no, update_slider=True):
        if not self.cap: return
        frame_no = max(0, min(frame_no, self.video_total_frames-1))
        self.current_frame_pos = frame_no
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        
        if ret and frame is not None and frame.size > 0:
            frame = fix_odd_dims(frame) # FIX: Apply cropping immediately
            self.current_video_frame = frame
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                w, h = img.size; max_h = 500
                if h > max_h: ratio = max_h/h; img = img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
                self.tk_vid_img = ImageTk.PhotoImage(img)
                self.vid_canvas.config(image=self.tk_vid_img, text="")
            except: pass
            
            self.lbl_vid_cnt.config(text=f"{frame_no}/{self.video_total_frames}")
            if update_slider: self.vid_slider.set(frame_no)

    def vid_snapshot(self):
        if self.cap and self.video_output_dir and self.current_video_frame is not None:
            name = f"{self.video_path.stem}_{self.current_frame_pos:06d}.jpg"
            cv2.imwrite(str(self.video_output_dir / name), self.current_video_frame)
            self.vid_refresh_saved()
            orig = self.vid_canvas.cget("bg"); self.vid_canvas.config(bg=SUCCESS)
            self.root.after(100, lambda: self.vid_canvas.config(bg="black"))

    def vid_extract_all(self):
        if self.is_running:
            self.is_running = False
            self.btn_extract_all.config(text="Stopping...", state="disabled")
        else:
            self.is_running = True; self.btn_extract_all.config(text="STOP", bg=ERROR)
            threading.Thread(target=self.vid_extract_thread, daemon=True).start()

    def vid_extract_thread(self):
        cap = cv2.VideoCapture(str(self.video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); count = 0
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret: break
            frame = fix_odd_dims(frame) 
            cv2.imwrite(str(self.video_output_dir / f"{self.video_path.stem}_{count:06d}.jpg"), frame)
            count += 1
            if count % 20 == 0: self.root.after(0, lambda c=count, t=total: self.vid_progress.configure(value=(c/t)*100))
        cap.release(); self.is_running = False
        self.root.after(0, lambda: [self.btn_extract_all.config(text="EXTRACT ALL", bg=HIGHLIGHT, state="normal"), self.vid_progress.configure(value=0), self.vid_refresh_saved()])

    def vid_refresh_saved(self):
        self.lst_frames.delete(0, tk.END)
        if self.video_output_dir and self.video_output_dir.exists():
            files = sorted(list(self.video_output_dir.glob(f"{self.video_path.stem}*.jpg")))
            for f in files: self.lst_frames.insert(tk.END, f.name)

    # ==========================
    # TAB: SMART CROPPING
    # ==========================
    def build_crop_tab(self):
        f = self.tabs["Smart Cropping"]; f.configure(padx=30, pady=20)
        self.crop_in = tk.StringVar(value=self.config.get("last_crop_in", ""))
        self.crop_out = tk.StringVar(value=self.config.get("last_crop_out", ""))
        self.field(f, "Input Folder", self.crop_in, True)
        self.field(f, "Output Folder", self.crop_out, True)
        
        cfg = tk.Frame(f, bg=BG); cfg.pack(fill="x", pady=20)
        c1 = tk.Frame(cfg, bg=BG); c1.pack(side="left", fill="x", expand=True)
        tk.Label(c1, text="Target Object", bg=BG, fg=DIM).pack(anchor="w")
        self.crop_target = tk.StringVar(value=DEFAULTS["crop_prompt"])
        tk.Entry(c1, textvariable=self.crop_target, bg=INPUT, fg=TEXT, bd=0).pack(fill="x", ipady=5)
        
        c2 = tk.Frame(cfg, bg=BG); c2.pack(side="left", fill="x", expand=True, padx=20)
        tk.Label(c2, text="Resize Mode", bg=BG, fg=DIM).pack(anchor="w")
        self.crop_meth = tk.StringVar(value="Bucket Resize (Mod 64)")
        ttk.Combobox(c2, textvariable=self.crop_meth, values=["Bucket Resize (Mod 64)", "Simple Pad (1024x1024)"], state="readonly").pack(fill="x", ipady=4)
        
        c3 = tk.Frame(cfg, bg=BG); c3.pack(side="left", fill="x", expand=True)
        tk.Label(c3, text="Pad Color", bg=BG, fg=DIM).pack(anchor="w")
        self.pad_col = tk.StringVar(value=DEFAULTS["pad_color"])
        pframe = tk.Frame(c3, bg=BG); pframe.pack(fill="x")
        self.btn_col = tk.Button(pframe, bg=self.pad_col.get(), width=3, command=self.pick_color, bd=0); self.btn_col.pack(side="left")
        tk.Entry(pframe, textvariable=self.pad_col, bg=INPUT, fg=TEXT, bd=0).pack(side="left", fill="x", ipady=5)

        self.btn_crop = tk.Button(f, text="START CROPPING", bg=HIGHLIGHT, fg=BG, bd=0, font=("Segoe UI", 10, "bold"), command=self.toggle_crop)
        self.btn_crop.pack(pady=20, ipady=10, fill="x")
        self.crop_log = tk.Label(f, text="Idle", bg=BG, fg=DIM); self.crop_log.pack()

    def pick_color(self):
        c = colorchooser.askcolor(color=self.pad_col.get())
        if c[1]: self.pad_col.set(c[1]); self.btn_col.config(bg=c[1])

    def toggle_crop(self):
        if not self.engine.is_loaded: return messagebox.showerror("Err", "Load Engine first!")
        if self.is_running:
            self.is_running = False
            self.btn_crop.config(text="Stopping...", state="disabled")
        else:
            self.save_config(); self.is_running = True
            self.btn_crop.config(text="STOP", bg=ERROR)
            threading.Thread(target=self.crop_worker, daemon=True).start()

    def crop_worker(self):
        src = Path(self.crop_in.get()); dst = Path(self.crop_out.get()); dst.mkdir(exist_ok=True)
        imgs = list(src.glob("*.jpg")) + list(src.glob("*.png"))
        target = self.crop_target.get(); mode = self.crop_meth.get()
        hc = self.pad_col.get().lstrip('#'); rgb = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4)); bgr = (rgb[2], rgb[1], rgb[0])

        for i, path in enumerate(imgs):
            if not self.is_running: break
            try:
                img_pil = Image.open(path).convert("RGB")
                res = self.engine.run_task(img_pil, "<CAPTION_TO_PHRASE_GROUNDING>", target)
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
                for idx, box in enumerate(res.get('bboxes', [])):
                    x1, y1, x2, y2 = map(int, box)
                    m = 20; x1=max(0,x1-m); y1=max(0,y1-m); x2=min(img_pil.width,x2+m); y2=min(img_pil.height,y2+m)
                    crop = img_cv[y1:y2, x1:x2]
                    
                    if "Bucket" in mode:
                        h, w = crop.shape[:2]; aspect = w / h; target_area = 1024 * 1024
                        new_h = int(np.sqrt(target_area / aspect)); new_w = int(new_h * aspect)
                        new_w = round(new_w / 64) * 64; new_h = round(new_h / 64) * 64
                        if new_w < 64: new_w = 64
                        if new_h < 64: new_h = 64
                        final = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        final_sz = 1024
                        ch, cw = crop.shape[:2]; scale = final_sz/max(ch, cw)
                        nw, nh = int(cw*scale), int(ch*scale); resized = cv2.resize(crop, (nw, nh))
                        dw, dh = final_sz-nw, final_sz-nh
                        final = cv2.copyMakeBorder(resized, dh//2, dh-(dh//2), dw//2, dw-(dw//2), cv2.BORDER_CONSTANT, value=bgr)
                    
                    cv2.imwrite(str(dst / f"{path.stem}_{idx}.jpg"), final)
                self.root.after(0, lambda x=i, t=len(imgs): self.crop_log.config(text=f"Processing {x+1}/{t}"))
            except Exception as e: print(e)
            
        self.is_running = False
        self.root.after(0, lambda: [self.crop_log.config(text="Done"), self.btn_crop.config(text="START CROPPING", bg=HIGHLIGHT, state="normal")])

    # ==========================
    # TAB: BATCH CAPTION
    # ==========================
    def build_batch_tab(self):
        f = self.tabs["Batch Captioning"]; f.configure(padx=30, pady=20)
        cfg = tk.Frame(f, bg=BG); cfg.pack(fill="x")
        tk.Label(cfg, text="Caption Mode:", bg=BG, fg=DIM).pack(side="left")
        self.cap_mode = tk.StringVar(value=DEFAULTS["caption_mode"])
        ttk.Combobox(cfg, textvariable=self.cap_mode, values=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"], state="readonly", width=30).pack(side="left", padx=10)
        
        tf = tk.Frame(f, bg=BG); tf.pack(fill="x", pady=15)
        t1 = tk.Frame(tf, bg=BG); t1.pack(side="left", fill="x", expand=True, padx=(0,5))
        tk.Label(t1, text="Prefix (Trigger Word)", bg=BG, fg=SUCCESS, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.txt_prefix = tk.Entry(t1, bg=INPUT, fg=TEXT, bd=0); self.txt_prefix.pack(fill="x", ipady=5)
        t2 = tk.Frame(tf, bg=BG); t2.pack(side="left", fill="x", expand=True, padx=(5,0))
        tk.Label(t2, text="Suffix (Style Tags)", bg=BG, fg=DIM).pack(anchor="w")
        self.txt_suffix = tk.Entry(t2, bg=INPUT, fg=TEXT, bd=0); self.txt_suffix.pack(fill="x", ipady=5)

        inp = tk.Frame(f, bg=BG); inp.pack(fill="x", pady=10)
        self.batch_dir = tk.StringVar(value=self.config.get("last_batch_dir", ""))
        self.field(inp, "Image Folder", self.batch_dir, True)
        
        self.btn_batch = tk.Button(f, text="START CAPTIONING", bg=ACCENT, fg=BG, bd=0, font=("Segoe UI", 10, "bold"), command=self.toggle_batch)
        self.btn_batch.pack(fill="x", pady=10, ipady=10)
        self.batch_bar = ttk.Progressbar(f, mode="determinate"); self.batch_bar.pack(fill="x")
        self.batch_log = tk.Label(f, text="Idle", bg=BG, fg=DIM); self.batch_log.pack()

    def toggle_batch(self):
        if not self.engine.is_loaded: return messagebox.showerror("Err", "Load Engine first!")
        if not self.batch_dir.get(): return
        self.save_config()
        if self.is_running:
            self.is_running = False; self.btn_batch.config(text="Stopping...", state="disabled")
        else:
            self.is_running = True; self.btn_batch.config(text="STOP", bg=ERROR)
            threading.Thread(target=self.batch_worker, daemon=True).start()

    def batch_worker(self):
        folder = Path(self.batch_dir.get())
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.webp"))
        mode = self.cap_mode.get(); prefix = self.txt_prefix.get().strip(); suffix = self.txt_suffix.get().strip()
        
        for i, path in enumerate(imgs):
            if not self.is_running: break
            try:
                img = Image.open(path).convert("RGB")
                generated = self.engine.run_task(img, mode)
                parts = []
                if prefix: parts.append(prefix)
                parts.append(generated)
                if suffix: parts.append(suffix)
                final = ", ".join(parts).replace(",,", ",")
                path.with_suffix(".txt").write_text(final, encoding="utf-8")
                self.root.after(0, lambda x=i, t=len(imgs): [self.batch_bar.configure(value=(x/t)*100), self.batch_log.config(text=f"Processed {x+1}/{t}")])
            except: pass
        self.is_running = False
        self.root.after(0, lambda: [self.batch_log.config(text="Done"), self.btn_batch.config(text="START CAPTIONING", bg=ACCENT, state="normal")])

    # ==========================
    # TAB: EDITOR
    # ==========================
    def build_editor_tab(self):
        f = self.tabs["Manual Edit"]
        t = tk.Frame(f, bg=BG); t.pack(fill="x", padx=10, pady=10)
        tk.Button(t, text="Open Folder", bg=CARD, fg=TEXT, bd=0, command=self.ed_load_btn).pack(side="left", padx=5, ipady=5)
        
        # METADATA BUTTON
        self.btn_meta = tk.Button(t, text="METADATA", bg=WARNING, fg=BG, bd=0, font=("Segoe UI", 9, "bold"), command=self.ed_show_meta, state="disabled")
        self.btn_meta.pack(side="left", padx=20, ipady=5)

        tk.Label(t, text="(Ctrl+S to Save, Alt+Arrows to Nav)", bg=BG, fg=DIM).pack(side="right")
        
        pane = tk.Frame(f, bg=BG); pane.pack(fill="both", expand=True, padx=10)
        left = tk.Frame(pane, bg=CARD, width=250); left.pack(side="left", fill="y")
        self.ed_list = tk.Listbox(left, bg=CARD, fg=TEXT, bd=0, highlightthickness=0, selectbackground=HIGHLIGHT)
        self.ed_list.pack(side="left", fill="both", expand=True)
        self.ed_list.bind('<<ListboxSelect>>', self.ed_select)
        sb = tk.Scrollbar(left, command=self.ed_list.yview); sb.pack(side="right", fill="y"); self.ed_list.config(yscrollcommand=sb.set)
        
        right = tk.Frame(pane, bg=BG); right.pack(side="left", fill="both", expand=True, padx=(10,0))
        self.ed_preview = tk.Label(right, bg="black", text="No Image", fg="white")
        self.ed_preview.pack(fill="both", expand=True)
        self.ed_txt = tk.Text(right, height=5, bg=INPUT, fg=TEXT, bd=0, font=("Segoe UI", 11), insertbackground=TEXT)
        self.ed_txt.pack(fill="x", pady=10)
        
        b = tk.Frame(right, bg=BG); b.pack(fill="x")
        tk.Button(b, text="SAVE (Ctrl+S)", bg=SUCCESS, fg=BG, bd=0, font=("Segoe UI", 9, "bold"), command=self.editor_save).pack(side="right", ipady=5, ipadx=10)
        tk.Button(b, text="< Prev", bg=CARD, fg=TEXT, bd=0, command=lambda: self.editor_nav(-1)).pack(side="left", ipady=5, ipadx=10)
        tk.Button(b, text="Next >", bg=CARD, fg=TEXT, bd=0, command=lambda: self.editor_nav(1)).pack(side="left", padx=5, ipady=5, ipadx=10)
        
        if self.config.get("last_editor_dir"): self.load_ed_dir(Path(self.config["last_editor_dir"]))

    def ed_load_btn(self):
        d = filedialog.askdirectory(initialdir=self.config.get("last_editor_dir", "/"))
        if d:
            self.config["last_editor_dir"] = d; self.save_config()
            self.load_ed_dir(Path(d))

    def load_ed_dir(self, path):
        self.editor_files = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")))
        self.ed_list.delete(0, tk.END)
        for p in self.editor_files: self.ed_list.insert(tk.END, p.name)
        if self.editor_files: self.editor_nav(0, abs_idx=0)

    def ed_select(self, event):
        sel = self.ed_list.curselection()
        if sel: self.load_ed_item(sel[0])

    def editor_nav(self, delta, abs_idx=None):
        if not self.editor_files: return
        idx = abs_idx if abs_idx is not None else max(0, min(len(self.editor_files)-1, self.current_editor_index + delta))
        self.ed_list.selection_clear(0, tk.END); self.ed_list.selection_set(idx); self.ed_list.see(idx); self.load_ed_item(idx)

    def load_ed_item(self, index):
        self.current_editor_index = index; path = self.editor_files[index]
        try:
            self.current_editor_img_obj = Image.open(path)
            # Check Metadata
            if 'workflow' in self.current_editor_img_obj.info or 'prompt' in self.current_editor_img_obj.info:
                self.btn_meta.config(state="normal", bg=WARNING)
            else:
                self.btn_meta.config(state="disabled", bg=CARD)
            
            w, h = self.current_editor_img_obj.size; max_h = 500
            if h > max_h: ratio = max_h/h; img = self.current_editor_img_obj.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
            else: img = self.current_editor_img_obj
            self.tk_ed_img = ImageTk.PhotoImage(img); self.ed_preview.config(image=self.tk_ed_img, text="")
        except: self.ed_preview.config(image="", text="Err")
        txt = path.with_suffix(".txt"); self.ed_txt.delete("1.0", tk.END)
        if txt.exists(): self.ed_txt.insert("1.0", txt.read_text(encoding="utf-8"))

    def editor_save(self):
        if self.current_editor_index < 0: return
        path = self.editor_files[self.current_editor_index]
        path.with_suffix(".txt").write_text(self.ed_txt.get("1.0", tk.END).strip(), encoding="utf-8")
        orig = self.ed_txt.cget("bg"); self.ed_txt.config(bg=SUCCESS); self.root.after(200, lambda: self.ed_txt.config(bg=orig))

    # --- METADATA INSPECTOR ---
    def ed_show_meta(self):
        if not self.current_editor_img_obj: return
        
        info = self.current_editor_img_obj.info
        top = tk.Toplevel(self.root); top.title("ComfyUI Metadata"); top.geometry("900x700"); top.configure(bg=BG)
        st = scrolledtext.ScrolledText(top, bg=INPUT, fg=TEXT, font=("Consolas", 10)); st.pack(fill="both", expand=True)
        
        # Configure Highlighting Tags
        st.tag_config("NODE", foreground=ACCENT, font=("Consolas", 10, "bold"))
        st.tag_config("KEY", foreground=WARNING)
        st.tag_config("VAL", foreground=SUCCESS)
        st.tag_config("HEADER", foreground=HIGHLIGHT, font=("Segoe UI", 12, "bold"))

        def pretty_print(data, indent=0):
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict) and "class_type" in v: # It's a Node
                        st.insert(tk.END, "  " * indent + f"[{k}] {v['class_type']}\n", "NODE")
                        if "inputs" in v:
                            pretty_print(v["inputs"], indent + 1)
                    else:
                        st.insert(tk.END, "  " * indent + f"{k}: ", "KEY")
                        st.insert(tk.END, f"{v}\n", "VAL")
            else:
                st.insert(tk.END, f"{data}\n", "VAL")

        # 1. PARSE PROMPT (The actual generation params)
        if 'prompt' in info:
            st.insert(tk.END, "=== GENERATION PARAMETERS (API) ===\n\n", "HEADER")
            try:
                j = json.loads(info['prompt'])
                pretty_print(j)
            except: st.insert(tk.END, str(info['prompt']))
            st.insert(tk.END, "\n" + "-"*50 + "\n\n")

        # 2. PARSE WORKFLOW (The Graph)
        if 'workflow' in info:
            st.insert(tk.END, "=== WORKFLOW GRAPH (JSON) ===\n\n", "HEADER")
            try:
                j = json.loads(info['workflow'])
                st.insert(tk.END, json.dumps(j, indent=2)) # Dump full graph
            except: st.insert(tk.END, str(info['workflow']))

    # --- HELPERS ---
    def field(self, p, l, v, b):
        tk.Label(p, text=l, bg=BG, fg=DIM).pack(anchor="w")
        r = tk.Frame(p, bg=BG); r.pack(fill="x", pady=(0, 10))
        tk.Entry(r, textvariable=v, bg=INPUT, fg=TEXT, bd=0).pack(side="left", fill="x", expand=True, ipady=6)
        if b: 
            def browse():
                d = filedialog.askdirectory(initialdir=v.get() if v.get() else "/")
                if d: v.set(d)
            tk.Button(r, text="...", command=browse, bg=CARD, fg=TEXT, bd=0).pack(side="right", padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    ForgeApp(root)
    root.mainloop()