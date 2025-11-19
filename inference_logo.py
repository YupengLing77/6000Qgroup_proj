"""
使用训练好的 LoRA 模型进行 Logo 识别推理
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import sys

def load_model(base_model_path, lora_checkpoint_path):
    """加载基础模型和 LoRA 权重"""
    print(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"加载 LoRA 权重: {lora_checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    model.eval()
    
    print("加载处理器...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    return model, processor


def predict_logo(model, processor, image_path, task="classify"):
    """
    预测 logo 信息
    
    Args:
        task: "classify" (分类), "detect" (检测), "industry" (行业)
    """
    image = Image.open(image_path).convert("RGB")
    
    # 根据任务选择提示词
    if task == "classify":
        prompt = "Identify the logo in this image. What is the industry and company name?"
    elif task == "detect":
        prompt = "Locate the logo in this image and output the bbox coordinates in JSON format."
    elif task == "industry":
        prompt = "What industry does this logo belong to?"
    else:
        prompt = task  # 自定义提示词
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 准备输入
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
    
    # 生成预测
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    
    # 解码输出
    generated_text = processor.batch_decode(
        output_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # 提取回答（去除输入部分）
    answer = generated_text.split("assistant\n")[-1].strip()
    
    return answer


if __name__ == "__main__":
    # 配置路径
    BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # 或本地路径
    LORA_CHECKPOINT = "output/qwen3-vl-2b-logo-lora/checkpoint-1500"  # 修改为实际 checkpoint
    
    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python inference_logo.py <图像路径> [任务类型]")
        print("任务类型: classify (默认), detect, industry")
        print(f"\n示例: python inference_logo.py logo_images/test/test_000001.jpg classify")
        sys.exit(1)
    
    image_path = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else "classify"
    
    # 加载模型
    model, processor = load_model(BASE_MODEL, LORA_CHECKPOINT)
    
    # 预测
    print(f"\n{'='*50}")
    print(f"图像: {image_path}")
    print(f"任务: {task}")
    print(f"{'='*50}\n")
    
    result = predict_logo(model, processor, image_path, task)
    
    print("预测结果:")
    print(result)
    print(f"\n{'='*50}")


# 批量测试示例
def batch_test(model, processor, image_dir, num_samples=5):
    """批量测试多张图像"""
    import os
    from glob import glob
    
    image_files = glob(os.path.join(image_dir, "*.jpg"))[:num_samples]
    
    print(f"\n批量测试 {len(image_files)} 张图像...\n")
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {os.path.basename(img_path)}")
        
        result = predict_logo(model, processor, img_path, "classify")
        print(f"结果: {result}\n")
