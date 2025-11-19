"""
对比训练前后模型的效果
比较基础模型 vs LoRA 微调后模型
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import json
import sys
from tqdm import tqdm

def load_base_model(model_path):
    """加载基础模型（未训练）"""
    print(f"加载基础模型: {model_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def load_lora_model(base_model_path, lora_checkpoint):
    """加载 LoRA 微调后的模型"""
    print(f"加载 LoRA 模型: {lora_checkpoint}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_checkpoint)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    return model, processor

def predict(model, processor, image_path, prompt):
    """使用模型进行预测"""
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text], 
        images=[image], 
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    
    generated_text = processor.batch_decode(
        output_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    answer = generated_text.split("assistant\n")[-1].strip()
    return answer

def compare_on_test_set(base_model, base_processor, lora_model, lora_processor, 
                        test_json, num_samples=10):
    """在测试集上对比两个模型"""
    
    # 加载测试数据
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 只测试前 num_samples 个
    test_samples = test_data[:num_samples]
    
    print(f"\n{'='*80}")
    print(f"在 {num_samples} 个测试样本上对比模型效果")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, sample in enumerate(tqdm(test_samples, desc="测试中")):
        image_path = sample['image']
        prompt = sample['conversations'][0]['value'].replace('<image>\n', '')
        ground_truth = sample['conversations'][1]['value']
        
        # 基础模型预测
        base_answer = predict(base_model, base_processor, image_path, prompt)
        
        # LoRA 模型预测
        lora_answer = predict(lora_model, lora_processor, image_path, prompt)
        
        result = {
            'id': i + 1,
            'image': image_path,
            'prompt': prompt,
            'ground_truth': ground_truth,
            'base_model': base_answer,
            'lora_model': lora_answer
        }
        results.append(result)
        
        # 打印对比
        print(f"\n{'─'*80}")
        print(f"样本 {i+1}/{num_samples}")
        print(f"{'─'*80}")
        print(f"📷 图像: {image_path.split('/')[-1]}")
        print(f"❓ 问题: {prompt[:100]}...")
        print(f"\n✅ 标准答案:")
        print(f"   {ground_truth}")
        print(f"\n🔵 基础模型:")
        print(f"   {base_answer}")
        print(f"\n🟢 LoRA模型:")
        print(f"   {lora_answer}")
        
        # 简单评估（是否包含关键词）
        gt_lower = ground_truth.lower()
        base_match = any(word in base_answer.lower() for word in gt_lower.split()[:3])
        lora_match = any(word in lora_answer.lower() for word in gt_lower.split()[:3])
        
        if lora_match and not base_match:
            print(f"   ✨ LoRA 模型更准确！")
        elif base_match and not lora_match:
            print(f"   ⚠️  基础模型更准确")
        elif lora_match and base_match:
            print(f"   ✓ 两个模型都正确")
    
    return results

def save_comparison_report(results, output_file="comparison_report.json"):
    """保存对比报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比报告已保存: {output_file}")

if __name__ == "__main__":
    # 配置
    BASE_MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
    LORA_CHECKPOINT = "/home/jiahuawang/test/classVLM/output/qwen3-vl-2b-logo-lora/checkpoint-564"
    TEST_JSON = "logo_test.json"
    NUM_SAMPLES = 10  # 测试样本数量
    
    print("="*80)
    print("模型对比测试")
    print("="*80)
    print(f"基础模型: {BASE_MODEL_PATH}")
    print(f"LoRA 模型: {LORA_CHECKPOINT}")
    print(f"测试数据: {TEST_JSON}")
    print(f"测试样本: {NUM_SAMPLES}")
    print("="*80)
    
    # 加载模型
    print("\n[1/3] 加载基础模型...")
    base_model, base_processor = load_base_model(BASE_MODEL_PATH)
    base_model.eval()
    
    print("\n[2/3] 加载 LoRA 模型...")
    lora_model, lora_processor = load_lora_model(BASE_MODEL_PATH, LORA_CHECKPOINT)
    lora_model.eval()
    
    # 开始对比
    print("\n[3/3] 开始对比测试...")
    results = compare_on_test_set(
        base_model, base_processor,
        lora_model, lora_processor,
        TEST_JSON, NUM_SAMPLES
    )
    
    # 保存结果
    save_comparison_report(results)
    
    print("\n" + "="*80)
    print("✅ 对比测试完成！")
    print("="*80)
