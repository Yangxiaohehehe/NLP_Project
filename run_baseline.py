import os
import json
import copy
import torch
import requests
from PIL import Image
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

# 设置 HF 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


BENCHMARK_FILE = "data/benchmark.json"       # json文件
OUTPUT_FILE = "results_baseline.json"    
pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"
conv_template = "llava_llama_3" 
device = "cuda"
device_map = "auto"


print(f"正在加载模型: {pretrained}...")
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)

model.eval()
model.tie_weights()


if not os.path.exists(BENCHMARK_FILE):
    raise FileNotFoundError(f"找不到文件: {BENCHMARK_FILE}，请确保文件在当前目录下。")

with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
    benchmark_data = json.load(f)

print(f"共加载 {len(benchmark_data)} 个测试样本。")

results = []


for item in tqdm(benchmark_data, desc="Running Inference"):
    sample_id = item['sample_id']
    image_path = item['image_path']
    question_text = item['question']
    

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"\n[Error] 无法读取图片 {image_path}: {e}")
        results.append({
            "sample_id": sample_id,
            "error": f"Image load failed: {str(e)}"
        })
        continue

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    image_sizes = [image.size]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer 
    
    final_question = DEFAULT_IMAGE_TOKEN + "\n" + question_text
    conv.append_message(conv.roles[0], final_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)


    with torch.inference_mode(): 
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,    
            temperature=0,
            max_new_tokens=256,
            modalities=["image"]*input_ids.shape[0]
        )

    # --- Decode ---
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    
    # --- 收集结果 ---
    result_item = {
        "sample_id": sample_id,
        "question": question_text,
        "golden_answer": item.get('golden_answer', ""), 
        "baseline_answer": text_outputs.strip()
    }
    results.append(result_item)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:

    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"结果已保存至 {OUTPUT_FILE}")