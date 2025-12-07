## 图片传输方案说明

### ✅ 当前使用方案: Base64编码

脚本现在使用 **base64编码** 方式将本地图片发送给API:

**工作原理**:
1. 读取本地图片文件
2. 编码为base64字符串
3. 以 `data:image/jpeg;base64,{data}` 格式发送给API
4. API接收并解析图片内容

**优点**:
- ✅ 无需上传到外部服务器
- ✅ 数据安全,不会泄露到第三方
- ✅ 速度快,直接传输
- ✅ 不需要额外的图床API key

**缺点**:
- ⚠️ 会增加token消耗(base64编码后体积增大约33%)
- ⚠️ 单张图片不能太大(建议<5MB)

### Token消耗估算

**Base64方案的token消耗**:
- 小图片(20KB): 约200-300 tokens/张
- 中等图片(100KB): 约800-1200 tokens/张
- 大图片(500KB): 约4000-6000 tokens/张

**批量评估100张图片的总成本**:
- 假设平均每张图片50KB
- 每次评估10张 = 约5000 tokens输入
- 10次调用 = 约50,000 tokens输入
- 加上输出和prompt = 约55,000 total tokens
- **预估成本: 约0.5-1元人民币**

### 如何降低成本

1. **减少评估样本数**:
```bash
python select_image/select_image.py --eval_samples 50  # 只评估50张
```

2. **使用快速模式**(不评估):
```bash
./select_image/run_selection_fast.sh  # 完全免费
```

3. **压缩图片**(如果图片很大):
```python
# 可以在脚本中添加图片压缩逻辑
from PIL import Image
img = Image.open(path)
img.thumbnail((800, 800))  # 缩放到最大800px
```

### 测试

运行测试脚本验证API连接:
```bash
python select_image/test_image_upload.py
```

这会:
1. 选择一张测试图片
2. 用base64编码发送给API
3. 显示API的识别结果
4. 验证整个流程是否正常
