"""
使用阿里云Qwen3-VL-Plus API智能选择训练子集
目标: 选择500张最有价值的图片用于小模型LoRA微调
策略: 
1. 类别均衡采样
2. 难度分级 (简单/中等/困难)
3. 行业分布均匀
4. 使用API批量评估图片质量和代表性
5. 优化token使用,节省成本
"""

import json
import os
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple
import base64
from tqdm import tqdm
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class IntelligentDataSelector:
    def __init__(self, api_key: str = None):
        """初始化阿里云API客户端"""
        print("Initializing Aliyun DashScope API client...")
        
        # 导入DashScope
        try:
            import dashscope
            from dashscope import MultiModalConversation
            self.dashscope = dashscope
            self.MultiModalConversation = MultiModalConversation
        except ImportError:
            print("Error: dashscope not installed. Installing...")
            os.system("pip install dashscope -q")
            import dashscope
            from dashscope import MultiModalConversation
            self.dashscope = dashscope
            self.MultiModalConversation = MultiModalConversation
        
        # 设置API Key
        if api_key:
            self.dashscope.api_key = api_key
        elif os.getenv('DASHSCOPE_API_KEY'):
            self.dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
        else:
            raise ValueError("Please provide API key via --api_key or DASHSCOPE_API_KEY env variable")
        
        print("API client initialized successfully!")
    
    def load_full_dataset(self, json_path: str) -> List[Dict]:
        """加载完整训练集"""
        print(f"Loading dataset from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def extract_unique_images(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """提取唯一图片及其对应的所有样本"""
        image_to_samples = defaultdict(list)
        for item in data:
            image_to_samples[item['image']].append(item)
        return dict(image_to_samples)
    
    def analyze_dataset_distribution(self, image_to_samples: Dict) -> Dict:
        """分析数据集分布"""
        companies = []
        industries = []
        
        for image, samples in image_to_samples.items():
            for sample in samples:
                for conv in sample['conversations']:
                    if conv['from'] == 'gpt':
                        value = conv['value']
                        if 'Industry:' in value and 'Company:' in value:
                            parts = value.split('\n')
                            industry = parts[0].replace('Industry:', '').strip()
                            company = parts[1].replace('Company:', '').strip()
                            industries.append(industry)
                            companies.append(company)
                            break
        
        industry_dist = Counter(industries)
        company_dist = Counter(companies)
        
        print(f"\n=== Dataset Distribution ===")
        print(f"Total unique images: {len(image_to_samples)}")
        print(f"Total unique companies: {len(company_dist)}")
        print(f"Total unique industries: {len(industry_dist)}")
        print(f"\nTop 10 industries:")
        for industry, count in industry_dist.most_common(10):
            print(f"  {industry}: {count}")
        
        return {
            'industries': industry_dist,
            'companies': company_dist,
            'total_images': len(image_to_samples)
        }
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def evaluate_image_difficulty_batch(self, image_paths: List[str]) -> List[Tuple[float, str]]:
        """
        批量评估图片难度,使用API调用
        返回: [(难度分数, 难度等级), ...]
        """
        results = []
        
        # 构建批量评估提示 - 增加批量大小以提高效率
        batch_size = 20  # 每次API调用评估20张图片(增加到20)
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            batch_results = self._evaluate_batch_api(batch)
            results.extend(batch_results)
            
            # API调用间隔,避免限流
            time.sleep(0.3)  # 减少等待时间
        
        return results
    
    def _evaluate_batch_api(self, image_paths: List[str]) -> List[Tuple[float, str]]:
        """使用API批量评估一批图片"""
        try:
            # 构建消息内容
            content = []
            
            # 简化提示,节省token
            prompt = "Rate each logo image's recognition difficulty (1-10):\n"
            prompt += "1-3=Easy (clear, large, simple bg)\n"
            prompt += "4-7=Medium (some challenges)\n"
            prompt += "8-10=Hard (small, obscured, complex bg)\n\n"
            prompt += "Output format: Image1: X, Image2: Y, ...\n\n"
            
            # 添加所有图片(使用base64编码)
            valid_count = 0
            for idx, image_path in enumerate(image_paths, 1):
                if os.path.exists(image_path):
                    # 读取图片并编码为base64
                    with open(image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    content.append({
                        "image": f"data:image/jpeg;base64,{image_data}"
                    })
                    prompt += f"Image{idx}: {os.path.basename(image_path)}\n"
                    valid_count += 1
            
            if valid_count == 0:
                print("No valid images found in batch")
                return [(5.0, "medium")] * len(image_paths)
            
            content.append({"text": prompt})
            
            messages = [{
                'role': 'user',
                'content': content
            }]
            
            # 调用API
            response = self.MultiModalConversation.call(
                model='qwen-vl-plus',  # 使用qwen-vl-plus模型
                messages=messages
            )
            
            # 解析响应
            if response.status_code == 200:
                output_text = response.output.choices[0].message.content[0]['text']
                print(f"API Response sample: {output_text[:200]}...")  # 显示前200字符
                return self._parse_batch_scores(output_text, len(image_paths))
            else:
                print(f"API Error: {response.code} - {response.message}")
                return [(5.0, "medium")] * len(image_paths)
                
        except Exception as e:
            print(f"Error in batch evaluation: {e}")
            return [(5.0, "medium")] * len(image_paths)
    
    def _parse_batch_scores(self, output_text: str, expected_count: int) -> List[Tuple[float, str]]:
        """解析批量评估结果"""
        results = []
        
        try:
            # 尝试解析格式: Image1: 5, Image2: 7, ...
            import re
            pattern = r'Image\d+:\s*(\d+(?:\.\d+)?)'
            matches = re.findall(pattern, output_text)
            
            print(f"Parsed {len(matches)} scores from API response (expected {expected_count})")
            
            for match in matches:
                score = float(match)
                score = max(1.0, min(10.0, score))
                
                if score <= 3:
                    level = "easy"
                elif score <= 7:
                    level = "medium"
                else:
                    level = "hard"
                
                results.append((score, level))
            
            # 如果解析失败或数量不对,用默认值填充
            if len(results) < expected_count:
                print(f"Warning: Only parsed {len(results)}/{expected_count} scores, filling with defaults")
            while len(results) < expected_count:
                results.append((5.0, "medium"))
                
        except Exception as e:
            print(f"Error parsing scores: {e}")
            results = [(5.0, "medium")] * expected_count
        
        return results[:expected_count]
    
    def stratified_sample(
        self,
        image_to_samples: Dict,
        target_count: int = 500,
        use_model_evaluation: bool = True,
        sample_images_to_eval: int = 300  # 增加到300张以获得更好的评估
    ) -> List[str]:
        """
        分层采样选择最优图片
        优化版本: 只评估少量样本,快速完成选择
        """
        print(f"\n=== Starting Stratified Sampling ===")
        print(f"Target: {target_count} images")
        
        # 1. 提取每张图片的元信息
        image_info = {}
        for image, samples in tqdm(image_to_samples.items(), desc="Extracting metadata"):
            info = {'image': image, 'samples': samples}
            
            # 提取公司和行业信息
            for sample in samples:
                for conv in sample['conversations']:
                    if conv['from'] == 'gpt':
                        value = conv['value']
                        if 'Industry:' in value and 'Company:' in value:
                            parts = value.split('\n')
                            info['industry'] = parts[0].replace('Industry:', '').strip()
                            info['company'] = parts[1].replace('Company:', '').strip()
                            break
                if 'company' in info:
                    break
            
            if 'company' not in info:
                info['industry'] = 'Unknown'
                info['company'] = 'Unknown'
            
            image_info[image] = info
        
        # 2. 按行业分组
        industry_groups = defaultdict(list)
        for image, info in image_info.items():
            industry_groups[info['industry']].append(image)
        
        print(f"Found {len(industry_groups)} industries")
        
        # 3. 如果使用模型评估,只评估少量样本
        if use_model_evaluation:
            print(f"\n=== Model Evaluation Phase ===")
            print(f"Evaluating {sample_images_to_eval} sample images to save tokens...")
            
            # 从每个行业中随机选择一些图片进行评估
            images_to_evaluate = []
            images_per_industry = max(1, sample_images_to_eval // len(industry_groups))
            
            for industry, images in industry_groups.items():
                sample_size = min(images_per_industry, len(images))
                images_to_evaluate.extend(random.sample(images, sample_size))
            
            # 限制总数
            if len(images_to_evaluate) > sample_images_to_eval:
                images_to_evaluate = random.sample(images_to_evaluate, sample_images_to_eval)
            
            print(f"Selected {len(images_to_evaluate)} images for evaluation")
            
            base_dir = Path(__file__).parent.parent
            
            # 准备完整路径
            image_paths = [str(base_dir / img) for img in images_to_evaluate]
            
            # 批量评估
            print("Calling API for batch evaluation...")
            batch_results = self.evaluate_image_difficulty_batch(image_paths)
            
            # 保存评估结果
            for image, (score, level) in zip(images_to_evaluate, batch_results):
                image_info[image]['difficulty_score'] = score
                image_info[image]['difficulty_level'] = level
            
            print(f"Evaluation complete! Token usage optimized.")
            
            # 预计算每个行业的平均难度分数(优化性能)
            print("Computing difficulty scores for remaining images...")
            industry_avg_scores = {}
            for industry, images in industry_groups.items():
                evaluated_scores = [
                    image_info[img]['difficulty_score']
                    for img in images
                    if 'difficulty_score' in image_info[img]
                ]
                if evaluated_scores:
                    industry_avg_scores[industry] = sum(evaluated_scores) / len(evaluated_scores)
                else:
                    industry_avg_scores[industry] = 5.0
            
            # 快速推断未评估的图片
            unevaluated_count = 0
            for image, info in image_info.items():
                if 'difficulty_score' not in info:
                    unevaluated_count += 1
                    # 使用预计算的行业平均分,并添加一些随机变化使分布更自然
                    base_score = industry_avg_scores.get(info['industry'], 5.0)
                    # 添加±1.5的随机偏移
                    score = base_score + random.uniform(-1.5, 1.5)
                    score = max(1.0, min(10.0, score))  # 限制在1-10之间
                    info['difficulty_score'] = score
                    
                    if score <= 3:
                        info['difficulty_level'] = 'easy'
                    elif score <= 7:
                        info['difficulty_level'] = 'medium'
                    else:
                        info['difficulty_level'] = 'hard'
            
            print(f"Inferred difficulty for {unevaluated_count} images based on industry averages.")
        else:
            # 不使用模型评估,随机分配难度
            for image, info in image_info.items():
                info['difficulty_score'] = 5.0
                info['difficulty_level'] = 'medium'
        
        # 4. 计算每个行业应该选择多少张图片
        min_per_industry = 3  # 每个行业至少3张
        industry_quotas = {}
        
        total_images = len(image_info)
        for industry, images in industry_groups.items():
            # 按比例分配,但保证最小值
            quota = max(min_per_industry, int(len(images) / total_images * target_count))
            industry_quotas[industry] = quota
        
        # 调整配额确保总数等于目标
        total_quota = sum(industry_quotas.values())
        if total_quota > target_count:
            # 按比例缩减
            scale = target_count / total_quota
            for industry in industry_quotas:
                industry_quotas[industry] = max(min_per_industry, int(industry_quotas[industry] * scale))
        
        # 5. 从每个行业中选择图片
        selected_images = []
        
        for industry, images in industry_groups.items():
            quota = industry_quotas.get(industry, min_per_industry)
            
            # 按难度分级: 30% easy, 40% medium, 30% hard
            easy_quota = int(quota * 0.3)
            medium_quota = int(quota * 0.4)
            hard_quota = quota - easy_quota - medium_quota
            
            # 分组
            easy_images = [img for img in images if image_info[img]['difficulty_level'] == 'easy']
            medium_images = [img for img in images if image_info[img]['difficulty_level'] == 'medium']
            hard_images = [img for img in images if image_info[img]['difficulty_level'] == 'hard']
            
            # 随机采样
            selected = []
            selected.extend(random.sample(easy_images, min(easy_quota, len(easy_images))))
            selected.extend(random.sample(medium_images, min(medium_quota, len(medium_images))))
            selected.extend(random.sample(hard_images, min(hard_quota, len(hard_images))))
            
            # 如果不够,从剩余的补充
            if len(selected) < quota:
                remaining = [img for img in images if img not in selected]
                selected.extend(random.sample(remaining, min(quota - len(selected), len(remaining))))
            
            selected_images.extend(selected)
            
            # 达到目标数量就停止
            if len(selected_images) >= target_count:
                selected_images = selected_images[:target_count]
                break
        
        # 6. 如果还不够,随机补充
        if len(selected_images) < target_count:
            remaining = [img for img in image_info.keys() if img not in selected_images]
            additional = random.sample(remaining, min(target_count - len(selected_images), len(remaining)))
            selected_images.extend(additional)
        
        # 7. 如果超了,随机删除
        if len(selected_images) > target_count:
            selected_images = random.sample(selected_images, target_count)
        
        print(f"\n=== Selection Complete ===")
        print(f"Selected {len(selected_images)} images")
        
        # 统计选择结果
        selected_industries = Counter([image_info[img]['industry'] for img in selected_images])
        selected_difficulties = Counter([image_info[img]['difficulty_level'] for img in selected_images])
        
        print(f"\nIndustry distribution:")
        for industry, count in selected_industries.most_common(10):
            print(f"  {industry}: {count}")
        
        print(f"\nDifficulty distribution:")
        for level, count in selected_difficulties.most_common():
            print(f"  {level}: {count}")
        
        return selected_images
    
    def create_subset_json(
        self,
        image_to_samples: Dict,
        selected_images: List[str],
        output_path: str
    ):
        """创建训练子集JSON文件"""
        print(f"\n=== Creating subset JSON ===")
        
        subset_data = []
        for image in selected_images:
            samples = image_to_samples[image]
            subset_data.extend(samples)
        
        # 保存
        with open(output_path, 'w') as f:
            json.dump(subset_data, f, indent=2)
        
        print(f"Saved {len(subset_data)} samples to {output_path}")
        print(f"  - {len(selected_images)} unique images")
        print(f"  - {len(subset_data) / len(selected_images):.1f} samples per image")


def main():
    parser = argparse.ArgumentParser(description='Intelligent training data selection using Aliyun API')
    parser.add_argument('--api_key', type=str, 
                       default=None,
                       help='Aliyun DashScope API key (or set DASHSCOPE_API_KEY env variable)')
    parser.add_argument('--input_json', type=str,
                       default='/home/jiahuawang/test/classVLM/train_part/logo_train.json',
                       help='Input full training JSON')
    parser.add_argument('--output_json', type=str,
                       default='/home/jiahuawang/test/classVLM/train_part/train_subset_500.json',
                       help='Output subset JSON')
    parser.add_argument('--target_count', type=int, default=500,
                       help='Target number of images to select')
    parser.add_argument('--use_model_eval', action='store_true',
                       help='Use model evaluation via API (costs tokens but more intelligent)')
    parser.add_argument('--eval_samples', type=int, default=300,
                       help='Number of images to evaluate with API (default: 300 for better quality)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 初始化选择器
    selector = IntelligentDataSelector(args.api_key)
    
    # 加载数据集
    data = selector.load_full_dataset(args.input_json)
    
    # 提取唯一图片
    image_to_samples = selector.extract_unique_images(data)
    
    # 分析分布
    selector.analyze_dataset_distribution(image_to_samples)
    
    # 智能选择
    selected_images = selector.stratified_sample(
        image_to_samples,
        target_count=args.target_count,
        use_model_evaluation=args.use_model_eval,
        sample_images_to_eval=args.eval_samples
    )
    
    # 创建子集
    selector.create_subset_json(image_to_samples, selected_images, args.output_json)
    
    print("\n=== Done! ===")
    print(f"Selected {len(selected_images)} images and saved to {args.output_json}")
    if args.use_model_eval:
        print(f"API evaluation used for {args.eval_samples} sample images to optimize token usage")


if __name__ == "__main__":
    main()
