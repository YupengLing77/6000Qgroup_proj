"""
测试阿里云DashScope API连接
验证API Key是否正确配置
"""

import os
import sys

def test_api_connection():
    print("Testing Aliyun DashScope API connection...")
    print("-" * 50)
    
    # 检查API Key
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not set")
        print("Please run: export DASHSCOPE_API_KEY='your-api-key'")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # 尝试导入dashscope
    try:
        import dashscope
        from dashscope import MultiModalConversation
        print("✅ dashscope package installed")
    except ImportError:
        print("⚠️  dashscope not installed, installing...")
        os.system("pip install dashscope -q")
        import dashscope
        from dashscope import MultiModalConversation
        print("✅ dashscope package installed")
    
    # 设置API Key
    dashscope.api_key = api_key
    
    # 测试API调用
    print("\nTesting API call with a simple text query...")
    try:
        messages = [{
            'role': 'user',
            'content': [
                {'text': 'Hello, this is a test. Please respond with "API connection successful".'}
            ]
        }]
        
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            output = response.output.choices[0].message.content[0]['text']
            print(f"✅ API Response: {output}")
            print("\n" + "=" * 50)
            print("✅ API connection test PASSED!")
            print("You can now run the data selection script.")
            print("=" * 50)
            return True
        else:
            print(f"❌ API Error: {response.code} - {response.message}")
            return False
            
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        return False

if __name__ == "__main__":
    success = test_api_connection()
    sys.exit(0 if success else 1)
