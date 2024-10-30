import os
from PIL import Image

# 设定要统计的图片文件夹路径
folder_path = r'F:\WJH\Ours\data\COCO2017\train2017'

# 初始化宽度和高度的列表
res = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否为图片格式
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 打开图片
        with Image.open(file_path) as img:
            # 获取图片的宽度和高度
            width, height = img.size
            # 将宽度和高度添加到列表中
            res.append(max(width, height))

# 计算宽度和高度的最大值

max_res = max(res) if res else 0

# 计算宽度和高度的平均值
avg_max_res = sum(res) / len(res) if res else 0


# 输出结果
# print(f"最大宽度: {max_width}")
# print(f"最大高度: {max_height}")
# print(f"平均最大宽度: {avg_max_width:.2f}")
print(f"平均最大高度: {avg_max_res:.2f}")
