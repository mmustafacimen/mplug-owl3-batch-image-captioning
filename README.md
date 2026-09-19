# mPLUG-Owl3 Batch Image Captioning

Batch image captioning and keyword extraction pipeline using the mPLUG-Owl3 multimodal LLM.

## Features

- Multimodal image-to-text generation using mPLUG-Owl3-7B
- Detailed scene description and keyword extraction
- Three distinct execution modes optimized for different hardware profiles:
  - FP16 full precision for high-end GPUs
  - 4-bit quantization for mid-range GPUs
  - Adaptive device mapping and CPU offloading for low-VRAM environments
- Automatic extension filtering and corrupted file handling
- Structured JSON output logging with inference execution time

## Hardware Requirements

| Script | Precision / Quantization | Target VRAM |
| :--- | :--- | :--- |
| full_gpu_fp16.py | FP16 (No quantization) | 16GB+ VRAM |
| full_gpu_4bit.py | 4-bit BitsAndBytes | 10–12GB VRAM |
| low_vram_auto.py | 4-bit + Adaptive Offload | 6–8GB VRAM |

## Installation

Install dependencies:

pip install -r requirements.txt

> Windows Note: Large checkpoint loading requires setting the Windows Paging File (Virtual Memory) to at least 16GB–32GB to avoid system memory allocation errors.

## Usage

Place input images in the data_with_blip/ directory and execute the script matching your hardware capacity:

For systems with 10–12GB VRAM:
python full_gpu_4bit.py

For memory-constrained systems (6–8GB VRAM):
python low_vram_auto.py

For high-end GPUs (16GB+ VRAM):
python full_gpu_fp16.py

## Output Format

Results are saved to the text/ directory in JSON Lines format:

{
  "image": "data_with_blip\\sample.jpg",
  "answer": "['1. Describe: Detailed image description.\\n2. Keywords: keyword1, keyword2']",
  "time_seconds": 10.45
}

## Project Structure

mplug-owl3-batch-image-captioning
│
├── data_with_blip/
│   └── example.jpg
├── owl3/
├── text/
│   ├── mplug3_output_fp16.txt
│   ├── mplug3_output_4bit.txt
│   └── mplug3_output_low_vram.txt
├── full_gpu_fp16.py
├── full_gpu_4bit.py
├── low_vram_auto.py
├── requirements.txt
├── .gitignore
└── README.md

## License

This project is open-source and available under the MIT License.
