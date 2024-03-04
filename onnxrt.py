import cv2
import numpy as np
import onnxruntime as ort

# 读取图片
image_path = "/home/ubuntu/DataSet/FULL_middle_png/test/images/AS-0.46-1 (8).png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 如果图片是 BGR 格式，转换为 RGB

# 预处理图片，具体的预处理操作要根据你的模型而定
# 这里仅仅是一个简单的示例，你可能需要根据你的模型要求进行适当的预处理
image = cv2.resize(image, (2560, 2560))
image = image / 255.0  # 归一化

# 加载 ONNX 模型
model_path = "/home/ubuntu/GITHUG/ultralytics/runs/detect/train9/weights/best.onnx"
sess = ort.InferenceSession(model_path)

# 输入 ONNX 模型的数据
input_name = sess.get_inputs()[0].name
input_data = image.astype(np.float32)[np.newaxis, ...]  # 添加 batch 维度
arr_reshaped = np.transpose(input_data, (0, 3, 1, 2))
# 进行推理
output_name = sess.get_outputs()[0].name
outputs = sess.run([output_name], {input_name: arr_reshaped})
a=[]
m=np.transpose(outputs[0], (0, 2, 1))
for i in range(134400):
    if outputs[0][0][4][i] > 0.5:
        a.append(m[0][i])
# 输出推理结果
print(a)
# print("Inference result:", outputs[0])
