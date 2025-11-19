"""
批量可视化测试集，生成 HTML 报告
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
import json
import re
import os
from tqdm import tqdm
import base64
from io import BytesIO

def parse_bbox(response_text):
    """从模型回答中提取 bbox 坐标"""
    json_match = re.search(r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
    if json_match:
        return [int(x) for x in json_match.groups()]
    array_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
    if array_match:
        return [int(x) for x in array_match.groups()]
    return None

def calculate_iou(box1, box2):
    """计算 IoU"""
    if not box1 or not box2:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def image_to_base64(image):
    """将 PIL 图像转为 base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def draw_bbox(image, bbox, color, label):
    """在图像上绘制 bbox"""
    if not bbox:
        return image
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox((x1, y1-22), label, font=font)
    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
    draw.text((x1, y1-22), label, fill="white", font=font)
    return image

def batch_visualize(test_json, base_model_path, lora_checkpoint, num_samples=5, output_dir="batch_visualization"):
    """批量可视化测试"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    # 只选择检测任务的样本（包含 "Locate" 或 "bbox" 的）
    detection_samples = []
    for sample in test_data:
        question = sample['conversations'][0]['value']
        if 'Locate' in question or 'bbox' in question.lower():
            detection_samples.append(sample)
    
    print(f"总样本数: {len(test_data)}")
    print(f"检测任务样本: {len(detection_samples)}")
    print(f"将测试前 {num_samples} 个检测样本\n")
    
    # 只取前 num_samples 个不同的图像
    selected_samples = detection_samples[:num_samples]
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    base_model.eval()
    
    # 加载 LoRA 模型（独立实例）
    print("加载 LoRA 模型...")
    lora_base = AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(lora_base, lora_checkpoint)
    lora_processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    lora_model.eval()
    
    results = []
    
    for i, sample in enumerate(tqdm(selected_samples, desc="处理中")):
        image_path = sample['image']
        gt_text = sample['conversations'][1]['value']
        gt_bbox = parse_bbox(gt_text)
        
        print(f"\n样本 {i}: {image_path}")
        print(f"  Ground Truth: {gt_bbox}")
        
        image = Image.open(image_path).convert("RGB")
        
        # 构建提示
        prompt = "Locate the logo in this image and output the bbox coordinates in JSON format."
        
        # 基础模型预测
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = base_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = base_processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        
        print("  预测基础模型...")
        with torch.no_grad():
            base_output = base_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        base_response = base_processor.batch_decode(base_output, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        base_bbox = parse_bbox(base_response)
        print(f"  基础模型: {base_bbox}")
        
        # LoRA 模型预测（重新构建 inputs）
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = lora_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = lora_processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        
        print("  预测 LoRA 模型...")
        with torch.no_grad():
            lora_output = lora_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        lora_response = lora_processor.batch_decode(lora_output, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        lora_bbox = parse_bbox(lora_response)
        print(f"  LoRA 模型: {lora_bbox}")
        print(f"  Ground Truth: {gt_bbox}")
        
        # 计算 IoU
        base_iou = calculate_iou(gt_bbox, base_bbox) if gt_bbox else 0
        lora_iou = calculate_iou(gt_bbox, lora_bbox) if gt_bbox else 0
        
        # 绘制对比图
        vis_image = image.copy()
        if gt_bbox:
            vis_image = draw_bbox(vis_image, gt_bbox, "red", "GT")
        if base_bbox:
            vis_image = draw_bbox(vis_image, base_bbox, "blue", f"Base IoU={base_iou:.2f}")
        if lora_bbox:
            vis_image = draw_bbox(vis_image, lora_bbox, "green", f"LoRA IoU={lora_iou:.2f}")
        
        vis_path = os.path.join(output_dir, f"sample_{i:03d}.jpg")
        vis_image.save(vis_path)
        print(f"  保存: {vis_path}")
        
        results.append({
            'id': i,
            'image': image_path,
            'gt_bbox': gt_bbox,
            'base_bbox': base_bbox,
            'lora_bbox': lora_bbox,
            'base_iou': base_iou,
            'lora_iou': lora_iou,
            'vis_path': f"sample_{i:03d}.jpg",
            'image_b64': image_to_base64(vis_image),
            'base_response': base_response,
            'lora_response': lora_response
        })
    
    # 生成 HTML 报告
    html = f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>模型对比报告</title>
    <style>
        body {{font-family: Arial; margin: 20px;}}
        .summary {{background: #f0f0f0; padding: 15px; margin-bottom: 20px; border-radius: 5px;}}
        .sample {{border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px;}}
        .sample img {{max-width: 800px; border: 1px solid #ccc;}}
        .metrics {{display: flex; gap: 20px; margin: 10px 0;}}
        .metric {{background: #e8f4f8; padding: 10px; border-radius: 3px;}}
        .better {{color: green; font-weight: bold;}}
        .worse {{color: red;}}
    </style></head><body>
    <h1>🎯 LoRA 微调效果对比报告</h1>
    <div class="summary">
        <h2>总体统计</h2>
        <p>测试样本数: {len(results)}</p>
        <p>平均 IoU (基础模型): {sum(r['base_iou'] for r in results)/len(results):.3f}</p>
        <p>平均 IoU (LoRA模型): {sum(r['lora_iou'] for r in results)/len(results):.3f}</p>
        <p class="{"better" if sum(r['lora_iou'] for r in results) > sum(r['base_iou'] for r in results) else "worse"}">
            提升: {(sum(r['lora_iou'] for r in results) - sum(r['base_iou'] for r in results))/len(results):.3f}
        </p>
    </div>
    """
    
    for r in results:
        better = "better" if r['lora_iou'] > r['base_iou'] else "worse"
        html += f"""
        <div class="sample">
            <h3>样本 {r['id'] + 1} - {os.path.basename(r['image'])}</h3>
            <img src="data:image/jpeg;base64,{r['image_b64']}">
            <div class="metrics">
                <div class="metric">Ground Truth: {r['gt_bbox']}</div>
                <div class="metric">基础模型 IoU: {r['base_iou']:.3f}</div>
                <div class="metric {better}">LoRA模型 IoU: {r['lora_iou']:.3f}</div>
            </div>
            <details>
                <summary>查看详细响应</summary>
                <p><strong>🔵 基础模型:</strong><br>{r['base_response'][:200]}...</p>
                <p><strong>🟢 LoRA模型:</strong><br>{r['lora_response'][:200]}...</p>
            </details>
            <p>🔵 基础预测: {r['base_bbox']}</p>
            <p>🟢 LoRA预测: {r['lora_bbox']}</p>
            <p>🔴 真实标注: {r['gt_bbox']}</p>
        </div>
        """
    
    html += "</body></html>"
    
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ 报告已生成: {report_path}")
    print(f"平均 IoU 提升: {(sum(r['lora_iou'] for r in results) - sum(r['base_iou'] for r in results))/len(results):.3f}")

if __name__ == "__main__":
    batch_visualize(
        "logo_test.json",
        "Qwen/Qwen3-VL-2B-Instruct",
        "/home/jiahuawang/test/classVLM/output/qwen3-vl-2b-logo-lora/checkpoint-564",
        num_samples=10
    )
