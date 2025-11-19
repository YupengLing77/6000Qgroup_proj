"""
将 LogoDet-3K 数据集转换为 Qwen3-VL 微调格式
"""
import json
import os
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image

def convert_to_qwen_format(dataset, output_json, image_output_dir, split_name):
    """
    转换数据集为 Qwen3-VL 格式
    
    数据格式：
    - image_path: PIL Image
    - industry_name: 行业类别
    - company_name: 公司名称
    - bbox: [x1, y1, x2, y2]
    """
    os.makedirs(image_output_dir, exist_ok=True)
    
    qwen_data = []
    
    print(f"正在转换 {split_name} 数据...")
    for idx, sample in enumerate(tqdm(dataset)):
        # 保存图像
        image = sample['image_path']
        image_filename = f"{split_name}_{idx:06d}.jpg"
        image_path = os.path.join(image_output_dir, image_filename)
        image.save(image_path, quality=95)
        
        # 获取数据
        industry = sample['industry_name']
        company = sample['company_name']
        bbox = sample['bbox']  # [x1, y1, x2, y2]
        
        # 创建 Qwen3-VL 格式的对话数据
        # 方式1: 分类任务 - 识别 logo 的行业和公司
        qwen_sample = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nIdentify the logo in this image. What is the industry and company name?"
                },
                {
                    "from": "gpt",
                    "value": f"Industry: {industry}\nCompany: {company}"
                }
            ]
        }
        qwen_data.append(qwen_sample)
        
        # 方式2: 增加目标检测任务 - 定位 logo 位置
        qwen_sample_bbox = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nLocate the {company} logo in this image and output the bbox coordinates in JSON format."
                },
                {
                    "from": "gpt",
                    "value": f'{{\n  "bbox_2d": [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n}}'
                }
            ]
        }
        qwen_data.append(qwen_sample_bbox)
        
        # 方式3: 只问行业类别（简化任务）
        qwen_sample_industry = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat industry does this logo belong to?"
                },
                {
                    "from": "gpt",
                    "value": f"{industry}"
                }
            ]
        }
        qwen_data.append(qwen_sample_industry)
    
    # 保存为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(qwen_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共生成 {len(qwen_data)} 条样本")
    print(f"JSON 文件: {output_json}")
    print(f"图像目录: {image_output_dir}")
    return len(qwen_data)


if __name__ == "__main__":
    # 加载数据集
    print("加载数据集...")
    dataset = load_from_disk("logo_data")
    
    # ⚠️  选择数据量 ⚠️
    # 选项1: 小规模测试（1K图，适合快速验证）
    # train_subset = dataset['train'].select(range(1000))
    # test_subset = dataset['test'].select(range(200))
    
    # 选项2: 中等规模（10K图，适合单卡训练）
    # train_subset = dataset['train'].select(range(10000))
    # test_subset = dataset['test'].select(range(1000))
    
    # 选项3: 全量数据（126K图，推荐多卡训练）
    train_subset = dataset['train']  # 全部 126,923 张
    test_subset = dataset['test'].select(range(1000))  # 测试集用1000张够了
    
    # 转换训练集
    print("\n" + "="*50)
    print(f"转换训练集... (共 {len(train_subset)} 张图)")
    train_count = convert_to_qwen_format(
        train_subset,
        output_json="logo_train.json",
        image_output_dir="logo_images/train",
        split_name="train"
    )
    
    # 转换测试集
    print("\n" + "="*50)
    print(f"转换测试集... (共 {len(test_subset)} 张图)")
    test_count = convert_to_qwen_format(
        test_subset,
        output_json="logo_test.json",
        image_output_dir="logo_images/test",
        split_name="test"
    )
    
    print("\n" + "="*50)
    print("数据转换完成统计:")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print("\n下一步:")
    print("1. 检查生成的 JSON 文件格式")
    print("2. 修改 Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py 添加数据配置")
    print("3. 运行 LoRA 微调脚本")
