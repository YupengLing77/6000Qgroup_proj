"""
测试批量API调用是否正确工作
"""
import os
import sys
from pathlib import Path

# 添加父目录到path
sys.path.insert(0, str(Path(__file__).parent))

from select_image import IntelligentDataSelector

def main():
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        sys.exit(1)
    
    # 初始化选择器
    selector = IntelligentDataSelector(api_key)
    
    # 获取几张测试图片
    base_dir = Path(__file__).parent.parent
    test_images = list((base_dir / 'logo_images' / 'train').glob('*.jpg'))[:3]
    
    if not test_images:
        print("No test images found")
        sys.exit(1)
    
    image_paths = [str(img) for img in test_images]
    
    print(f"Testing batch API call with {len(image_paths)} images:")
    for img in image_paths:
        print(f"  - {img}")
    
    print("\nCalling API...")
    results = selector.evaluate_image_difficulty_batch(image_paths)
    
    print(f"\n=== Results ===")
    for i, (path, (score, level)) in enumerate(zip(image_paths, results), 1):
        print(f"Image {i}: {os.path.basename(path)}")
        print(f"  Score: {score}, Level: {level}")
    
    # 检查结果
    if all(score == 5.0 and level == 'medium' for score, level in results):
        print("\n⚠️  Warning: All images got default scores!")
        print("This suggests API call might have failed or parsing failed.")
    else:
        print("\n✅ API batch evaluation working correctly!")

if __name__ == "__main__":
    main()
