"""
快速测试单张图像 - 对比训练前后效果
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import sys

def test_single_image(image_path, use_lora=True):
    """测试单张图像"""
    
    BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
    LORA_CHECKPOINT = "/home/jiahuawang/test/classVLM/output/qwen3-vl-2b-logo-lora/checkpoint-564"
    
    print(f"\n{'='*80}")
    if use_lora:
        print("使用 LoRA 微调后的模型")
        print(f"Checkpoint: {LORA_CHECKPOINT}")
    else:
        print("使用基础模型（未训练）")
    print(f"{'='*80}\n")
    
    # 加载模型
    print("加载模型中...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if use_lora:
        model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
    else:
        model = base_model
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model.eval()
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    print(f"图像: {image_path}")
    print(f"尺寸: {image.size}")
    
    # 定义测试任务
    tasks = [
        ("分类任务", "Identify the logo in this image. What is the industry and company name?"),
        ("行业识别", "What industry does this logo belong to?"),
        ("目标检测", "Locate the logo in this image and output the bbox coordinates in JSON format.")
    ]
    
    print(f"\n{'='*80}")
    print("开始测试")
    print(f"{'='*80}\n")
    
    for task_name, prompt in tasks:
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
        
        print(f"📌 {task_name}")
        print(f"问题: {prompt}")
        print(f"回答: {answer}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python quick_test.py <图像路径>")
        print("示例: python quick_test.py logo_images/test/test_000001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # 先测试基础模型
    print("\n" + "🔵 "*40)
    print("测试 1: 基础模型（未训练）")
    print("🔵 "*40)
    test_single_image(image_path, use_lora=False)
    
    # 再测试 LoRA 模型
    print("\n" + "🟢 "*40)
    print("测试 2: LoRA 微调模型")
    print("🟢 "*40)
    test_single_image(image_path, use_lora=True)
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
