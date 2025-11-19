"""
可视化对比训练前后的检测效果
在图像上标注 bbox 并保存对比图
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
import json
import re
import sys
import os

def parse_bbox(response_text):
    """从模型回答中提取 bbox 坐标"""
    # 尝试匹配 JSON 格式: {"bbox_2d": [x1, y1, x2, y2]}
    json_match = re.search(r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
    if json_match:
        return [int(x) for x in json_match.groups()]
    
    # 尝试匹配数组格式: [x1, y1, x2, y2]
    array_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
    if array_match:
        return [int(x) for x in array_match.groups()]
    
    return None

def load_model_and_predict(image_path, model_path, lora_checkpoint=None, task_type="detect"):
    """加载模型并预测"""
    # 加载模型
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if lora_checkpoint:
        model = PeftModel.from_pretrained(model, lora_checkpoint)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 构建不同任务的提示词
    if task_type == "detect":
        prompt = "Locate the logo in this image and output the bbox coordinates in JSON format."
    elif task_type == "classify":
        prompt = "Identify the logo in this image. What is the industry and company name?"
    else:
        prompt = "What industry does this logo belong to?"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = generated_text.split("assistant\n")[-1].strip()
    
    return answer, image

def draw_bbox_on_image(image, bbox, color, label, thickness=3):
    """在图像上绘制 bbox"""
    draw = ImageDraw.Draw(image)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # 添加标签
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制标签背景
        text_bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x1, y1-25), label, fill="white", font=font)
        
        # 显示坐标
        coord_text = f"[{x1},{y1},{x2},{y2}]"
        draw.text((x1, y2+5), coord_text, fill=color, font=font)
    
    return image

def create_comparison_visualization(image_path, base_model_path, lora_checkpoint, output_dir="visualization"):
    """创建对比可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_name = os.path.basename(image_path)
    print(f"\n{'='*80}")
    print(f"可视化对比: {image_name}")
    print(f"{'='*80}\n")
    
    # 1. 基础模型预测
    print("🔵 测试基础模型（未训练）...")
    base_response, original_image = load_model_and_predict(
        image_path, base_model_path, None, "detect"
    )
    base_bbox = parse_bbox(base_response)
    print(f"基础模型回答: {base_response}")
    print(f"提取的 bbox: {base_bbox}")
    
    # 2. LoRA 模型预测
    print("\n🟢 测试 LoRA 模型...")
    lora_response, _ = load_model_and_predict(
        image_path, base_model_path, lora_checkpoint, "detect"
    )
    lora_bbox = parse_bbox(lora_response)
    print(f"LoRA 模型回答: {lora_response}")
    print(f"提取的 bbox: {lora_bbox}")
    
    # 3. 获取分类结果
    print("\n📊 获取分类结果...")
    base_classify, _ = load_model_and_predict(image_path, base_model_path, None, "classify")
    lora_classify, _ = load_model_and_predict(image_path, base_model_path, lora_checkpoint, "classify")
    
    # 4. 创建可视化对比图
    print("\n🎨 生成可视化图像...")
    
    # 基础模型可视化
    base_image = original_image.copy()
    base_image = draw_bbox_on_image(
        base_image, base_bbox, 
        color="blue", 
        label="Base Model",
        thickness=4
    )
    
    # LoRA 模型可视化
    lora_image = original_image.copy()
    lora_image = draw_bbox_on_image(
        lora_image, lora_bbox, 
        color="green", 
        label="LoRA Model",
        thickness=4
    )
    
    # 叠加对比（两个框都显示）
    combined_image = original_image.copy()
    if base_bbox:
        combined_image = draw_bbox_on_image(
            combined_image, base_bbox,
            color="blue",
            label="Base",
            thickness=3
        )
    if lora_bbox:
        combined_image = draw_bbox_on_image(
            combined_image, lora_bbox,
            color="green",
            label="LoRA",
            thickness=3
        )
    
    # 创建拼接图（横向）
    width, height = original_image.size
    comparison = Image.new('RGB', (width * 3, height), (255, 255, 255))
    comparison.paste(base_image, (0, 0))
    comparison.paste(lora_image, (width, 0))
    comparison.paste(combined_image, (width * 2, 0))
    
    # 添加标题
    draw = ImageDraw.Draw(comparison)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        title_font = ImageFont.load_default()
    
    # 保存结果
    output_path = os.path.join(output_dir, f"comparison_{image_name}")
    comparison.save(output_path)
    
    # 保存单独的图像
    base_image.save(os.path.join(output_dir, f"base_{image_name}"))
    lora_image.save(os.path.join(output_dir, f"lora_{image_name}"))
    combined_image.save(os.path.join(output_dir, f"overlay_{image_name}"))
    
    print(f"\n✅ 可视化结果已保存:")
    print(f"   - 对比图: {output_path}")
    print(f"   - 基础模型: {output_dir}/base_{image_name}")
    print(f"   - LoRA模型: {output_dir}/lora_{image_name}")
    print(f"   - 叠加图: {output_dir}/overlay_{image_name}")
    
    # 打印分类对比
    print(f"\n{'='*80}")
    print("分类结果对比:")
    print(f"{'='*80}")
    print(f"🔵 基础模型: {base_classify[:100]}...")
    print(f"🟢 LoRA模型: {lora_classify[:100]}...")
    
    return {
        'base_bbox': base_bbox,
        'lora_bbox': lora_bbox,
        'base_classify': base_classify,
        'lora_classify': lora_classify,
        'output_path': output_path
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python visualize_comparison.py <图像路径>")
        print("示例: python visualize_comparison.py logo_images/test/test_000000.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
    LORA_CHECKPOINT = "/home/jiahuawang/test/classVLM/output/qwen3-vl-2b-logo-lora/checkpoint-564"
    
    result = create_comparison_visualization(image_path, BASE_MODEL, LORA_CHECKPOINT)
    
    print(f"\n{'='*80}")
    print("✅ 完成！请查看 visualization/ 目录")
    print(f"{'='*80}")
