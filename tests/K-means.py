import json
import glob
import random
import xml.etree.ElementTree as ET

import numpy as np


# 请确保将以下路径替换为你的JSON文件的实际路径
json_file_path = r'F:\WJH\Ours\datasets\UAV123\UAV123.json'

def read_and_filter_json(file_path):
    # 打开文件并读取JSON数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化一个字典来保存筛选后的数据
    filtered_data = {}
    areas = []

    # 筛选以"init_rect"开头的键
    for key, value in data.items():
        filtered_data[key] = value['init_rect'][-2:]
        areas.append([value['init_rect'][-2], value['init_rect'][-1]])
    # 返回筛选后的数据
    return filtered_data,areas

def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    # 取出一共有多少框
    row = box.shape[0]

    # 每个框各个点的位置
    distance = np.empty((row, k))

    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # 取出最小点
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)

        last_clu = near

    return cluster


def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height <= 0 or width <= 0:
            continue

        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


if __name__ == '__main__':
    # 运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成yolo_anchors.txt
    # 调用函数并打印结果
    filtered_json_data, filter_areas  = read_and_filter_json(json_file_path)
    # print(filter_areas)
    data = np.array(filter_areas)
    SIZE = 1
    anchors_num = 3
    ###############################
    # 载入数据集，可以使用VOC的xml
    # path = "/home/ubuntu/Z/XYZ/DET/z-yolov3-cfg-linux/data/Annotations"
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    # data = load_data(path)

    # 使用k聚类算法
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print(out * SIZE)
    data = out * SIZE
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()
