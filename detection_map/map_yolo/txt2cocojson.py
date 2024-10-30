'''
提交一份转换json数据代码，该代码已经进行了防错逻辑处理，
可以防止有xml没img或有img无xml情况，</p>
该代码处理多个文件夹或一个文件夹情况(有xml或无xml情况)。

'''

import os
import json
# import xml.etree.ElementTree as ET
import cv2  # 无xml时候需要读取图片高与宽

from tqdm import tqdm

import numpy as np

# 按行读取txt格式文件
def read_txt(path):
    txt_info_lst = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            txt_info_lst.append(list(line.strip('\n').split()))
    txt_info_lst = np.array(txt_info_lst)
    return txt_info_lst
# 图像上打印中文的文字
def chinese2img(img, str,  coord=(0, 0),label_size=20,label_color=(255, 0, 0)):
    # 将具有中文的字符打印到图上
    from PIL import Image, ImageDraw, ImageFont
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("font/simsun.ttc", label_size, encoding="utf-8")
    # font = ImageFont.truetype("./simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(tuple(coord), str, label_color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2charimg

def get_root_lst(root, suffix='jpg', suffix_n=3):
    root_lst, name_lst = [], []

    for dir, file, names in os.walk(root):
        root_lst = root_lst + [os.path.join(dir, name) for name in names if name[-suffix_n:] == suffix]
        name_lst = name_lst + [name for name in names if name[-suffix_n:] == suffix]

    return root_lst, name_lst

def get_codename(code_file):
    code_name = []
    with open(code_file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if '\n' in line:
                code_name.append(line[:-1])
            else:
                code_name.append(line)
    return code_name


names_cls=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

# names_cls = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

categories = {}
for i, cls in enumerate(names_cls):
    categories[i] = cls




def get_strfile(file_str, pos=-1):
    '''
    得到file_str / or \\ 的最后一个名称
    '''
    endstr_f_filestr = file_str.split('\\')[pos] if '\\' in file_str else file_str.split('/')[pos]
    return endstr_f_filestr


def build_files(root):
    '''
    :得到该路径下的所有文件
    '''
    files = [os.path.join(root, file) for file in os.listdir(root)]
    files_true = []
    for file in files:
        if not os.path.isfile(file):
            files_true.append(file)
    return files_true


def build_dir(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir


def drwa_yolo(img, txt_info):
    height, width = img.shape[:2]
    for info in txt_info:
        label = str(info[0])
        x, y, w, h = float(info[1]) * width, float(info[2]) * height, float(info[3]) * width, float(info[4]) * height

        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
        img = chinese2img(img, label, coord=(int(x - w / 2), int(y - h / 2)))
    return img


def main_yolov5txt2cocojson(root_data, txt_root=None, categories=None, out_dir=None, json_name='coco.json',save_img=False):
    '''
    json文件中的file_name包含文件夹/名字
    :param json_name: 保存json文件名字
    :param categories: 类别信息，为None则将self.root文件夹的名字作为类别信息
    :return:
    '''
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    image_id = 10000000
    anation_id = 10000000

    txt_root = txt_root if txt_root is not None else root_data
    img_roots_lst, img_names_lst = get_root_lst(root_data, suffix='jpg', suffix_n=3)
    txt_roots_lst, txt_names_lst = get_root_lst(txt_root, suffix='txt', suffix_n=3)

    assert categories is not None, 'lackimg categories'
    if isinstance(categories,list):
        categories = {i:cls for i, cls in enumerate(names_cls)}


    info_dict = {'label_int': []}

    for i, img_root in tqdm(enumerate(img_roots_lst)):
        img_name = img_names_lst[i]
        txt_name = img_name[:-3] + 'txt'
        img = cv2.imread(img_root)
        if txt_name in txt_names_lst:

            txt_index = list(txt_names_lst).index(str(txt_name))

            height, width = img.shape[:2]
            image_id = image_id + 1
            # image_id = img_name[:-4]  # yolov6格式
            image = {'file_name': img_name, 'height': height, 'width': width,
                     'id': image_id}


            txt_info = read_txt(txt_roots_lst[txt_index])

            # draw_img=drwa_yolo(img, txt_info)
            # show_img(draw_img)
            # print('o')
            for info in txt_info:
                label_int = int(info[0])
                if label_int not in info_dict['label_int']:
                    info_dict['label_int'].append(label_int)
                label = categories[label_int]
                if label not in categories.values():
                    print('skip code for num is spare {}:{}'.format(label_int, label))
                    continue
                # opzealot debug
                if image not in json_dict['images']:
                    json_dict['images'].append(image)  # 将图像信息添加到json中
                x, y, w, h = float(info[1]) * width, float(info[2]) * height, float(info[3]) * width, float(
                    info[4]) * height

                img = cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                    (255, 0, 0), 2)
                img = chinese2img(img, label, coord=(int(x - w / 2), int(y - h / 2)))

                category_id = label_int + 1  # 给出box对应标签索引为类
                anation_id = anation_id + 1

                xmin, ymin, o_width, o_height = int(x - w / 2), int(y - h / 2), int(w), int(h)

                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                       'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': anation_id, 'ignore': 0,
                       'segmentation': []}
                json_dict['annotations'].append(ann)
        if save_img:
            save_img_path = build_dir(os.path.join(out_dir, 'save_draw_img'))
            cv2.imwrite(os.path.join(save_img_path,img_name), img)

    for cid, cate in enumerate(categories.values()):
        cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
        json_dict['categories'].append(cat)
    if out_dir is not None:
        build_dir(out_dir)
        out_dir = os.path.join(out_dir, json_name)
    else:
        out_dir = os.path.join(root_path, json_name)
    with open(out_dir, 'w') as f:
        json.dump(json_dict, f, indent=4)  # indent表示间隔长度
    print('[info_count]:{}'.format(info_dict))





if __name__ == '__main__':

    root_path = r'E:\project\project_paper\project_LVF\code\yolov5-6.1-LVF\coco128\images\train'
    txt_root = r'E:\project\project_paper\project_LVF\code\yolov5-6.1-LVF\coco128\labels\train'

    main_yolov5txt2cocojson(root_path,
                            txt_root=txt_root,
                            out_dir='.',
                            categories=names_cls,
                            json_name='gt_coco.json',
                            save_img=True
                            )
