"""
测试图片上传和API调用
1. 上传本地图片到图床获取公网URL
2. 使用URL调用API识别图片
"""

import os
import sys
import base64
import requests
from pathlib import Path

def test_with_base64(image_path: str, api_key: str):
    """方案1: 使用base64编码直接发送图片"""
    print(f"\n=== 测试方案1: Base64编码 ===")
    print(f"图片: {image_path}")
    
    try:
        import dashscope
        from dashscope import MultiModalConversation
        
        dashscope.api_key = api_key
        
        # 读取图片并编码为base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        print(f"图片大小: {len(image_data)} bytes (base64)")
        
        # 构建消息
        messages = [{
            'role': 'user',
            'content': [
                {
                    'image': f'data:image/jpeg;base64,{image_data}'
                },
                {
                    'text': 'Describe this logo image briefly. What do you see?'
                }
            ]
        }]
        
        print("调用API...")
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            output = response.output.choices[0].message.content[0]['text']
            print(f"✅ API响应成功!")
            print(f"回答: {output}")
            return True
        else:
            print(f"❌ API错误: {response.code} - {response.message}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def upload_to_smms(image_path: str) -> str:
    """上传图片到sm.ms图床(免费,无需注册)"""
    print(f"\n=== 上传图片到sm.ms图床 ===")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'smfile': f}
            response = requests.post(
                'https://sm.ms/api/v2/upload',
                files=files,
                timeout=30
            )
        
        result = response.json()
        
        if result['success']:
            url = result['data']['url']
            print(f"✅ 上传成功!")
            print(f"URL: {url}")
            return url
        else:
            print(f"❌ 上传失败: {result.get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ 上传错误: {e}")
        return None


def upload_to_imgbb(image_path: str, api_key: str = None) -> str:
    """上传图片到imgbb图床(需要API key,免费)"""
    print(f"\n=== 上传图片到imgbb图床 ===")
    
    if not api_key:
        print("⚠️  需要imgbb API key,跳过")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            'https://api.imgbb.com/1/upload',
            data={
                'key': api_key,
                'image': image_data
            },
            timeout=30
        )
        
        result = response.json()
        
        if result['success']:
            url = result['data']['url']
            print(f"✅ 上传成功!")
            print(f"URL: {url}")
            return url
        else:
            print(f"❌ 上传失败: {result.get('error', {}).get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ 上传错误: {e}")
        return None


def test_with_url(image_url: str, api_key: str):
    """方案2: 使用公网URL调用API"""
    print(f"\n=== 测试方案2: 公网URL ===")
    print(f"URL: {image_url}")
    
    try:
        import dashscope
        from dashscope import MultiModalConversation
        
        dashscope.api_key = api_key
        
        # 构建消息
        messages = [{
            'role': 'user',
            'content': [
                {
                    'image': image_url
                },
                {
                    'text': 'Describe this logo image briefly. What do you see?'
                }
            ]
        }]
        
        print("调用API...")
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            output = response.output.choices[0].message.content[0]['text']
            print(f"✅ API响应成功!")
            print(f"回答: {output}")
            return True
        else:
            print(f"❌ API错误: {response.code} - {response.message}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    print("=" * 60)
    print("图片上传与API调用测试")
    print("=" * 60)
    
    # 获取API Key
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)
    
    # 选择一张测试图片
    base_dir = Path(__file__).parent.parent
    test_images = list((base_dir / 'logo_images' / 'train').glob('*.jpg'))[:3]
    
    if not test_images:
        print("❌ 找不到测试图片")
        sys.exit(1)
    
    image_path = str(test_images[0])
    print(f"\n使用测试图片: {image_path}")
    
    # 检查图片大小
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    print(f"图片大小: {size_mb:.2f} MB")
    
    # 方案1: 直接使用base64(推荐,不需要上传)
    print("\n" + "=" * 60)
    print("方案1: Base64编码(推荐)")
    print("=" * 60)
    success_base64 = test_with_base64(image_path, api_key)
    
    if success_base64:
        print("\n✅ Base64方案可用,推荐使用这个方案!")
        print("不需要上传图片,直接编码发送,更快更安全")
        return
    
    # 方案2: 上传到图床
    print("\n" + "=" * 60)
    print("方案2: 上传到免费图床")
    print("=" * 60)
    
    # 尝试sm.ms
    url = upload_to_smms(image_path)
    
    if url:
        test_with_url(url, api_key)
    else:
        print("\n⚠️  图床上传失败,建议使用Base64方案")


if __name__ == "__main__":
    main()
