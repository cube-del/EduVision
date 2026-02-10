Local Handwriting OCR (Qwen-VL Optimized)

Optimized Local Handwriting OCR
This repository contains a high-performance Python script for recognizing handwritten text from images using a local Vision-Language Model (VLM).

The project is optimized for consumer-grade GPUs using 6-bit quantization (NF4) and dynamic image slicing. This allows large multimodal models (such as Qwen-VL or Qolda) to run locally with minimal VRAM usage and low latency, reducing inference time from minutes to seconds.


Key Features:
Offline Execution: All processing is done locally on your machine. No data is sent to external servers.

Memory Efficiency: Uses bitsandbytes to load models in 4-bit precision, reducing VRAM requirements by approximately 70%.

High-Resolution Support: Implements a dynamic slicing mechanism that cuts images into high-resolution tiles to preserve details essential for reading small handwriting.

Automatic Optimization: Automatically detects CUDA availability and handles mixed-precision (FP16) casting.


Prerequisites:
Operating System: Linux (recommended) or Windows (via WSL2).

Python: Version 3.10 or higher.

Hardware: NVIDIA GPU with CUDA support (Minimum 6GB VRAM recommended).

Drivers: Up-to-date NVIDIA drivers and CUDA Toolkit.

There is an application based on gradio
