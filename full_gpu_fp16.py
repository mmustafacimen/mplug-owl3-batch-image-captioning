import os
import time
import json
import gc
from PIL import Image
import torch
from transformers import AutoTokenizer
from owl3.configuration_mplugowl3 import mPLUGOwl3Config
from owl3.modeling_mplugowl3 import mPLUGOwl3Model

def main():
    model_path = "mPLUG/mPLUG-Owl3-7B-240728"
    input_folder = "data_with_blip"
    output_dir = "text"
    output_file = os.path.join(output_dir, "mplug3_output_fp16.txt")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model configuration...")
    config = mPLUGOwl3Config.from_pretrained(model_path)

    print("Loading model in FP16 precision directly to GPU...")
    model = mPLUGOwl3Model.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype=torch.half
    )
    model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)
    print("Model and processor initialized successfully. Starting inference loop...")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for file_name in files:
        name, ext = os.path.splitext(file_name)
        if ext.lower() not in valid_extensions:
            continue

        file_path = os.path.join(input_folder, file_name)
        try:
            image = Image.open(file_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"Skipping corrupted image {file_name}: {e}")
            continue

        images = [image]

        messages = [
            {
                "role": "user",
                "content": (
                    "<|image|>\n"
                    "Answer these two questions;\n"
                    "1. Describe what the image represents in detail?\n"
                    "2. Provide basic keywords that describe the objects, people, clothing, or environment in the image. "
                    "Provide your answer in the following format;\n"
                    '1.describe = ""\n'
                    '2.keywords = "". '
                ),
            },
            {"role": "assistant", "content": ""}
        ]

        inputs = processor(messages, images=images, videos=None)
        inputs.to("cuda")
        inputs.update({
            "tokenizer": tokenizer,
            "max_new_tokens": 200,
            "decode_text": True,
            "repetition_penalty": 1.1,
        })

        start_time = time.time()
        outputs = model.generate(**inputs)
        elapsed_time = time.time() - start_time

        print(f"Processed: {file_name} | Elapsed Time: {elapsed_time:.2f}s")

        record = {
            "image": str(file_path),
            "answer": str(outputs),
            "time_seconds": elapsed_time
        }

        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

        del inputs, outputs, images, image, messages
        torch.cuda.empty_cache()
        gc.collect()

    print("Batch processing completed successfully.")

if __name__ == "__main__":
    main()