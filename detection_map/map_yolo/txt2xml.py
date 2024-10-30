'''
提交一份转换json数据代码，该代码已经进行了防错逻辑处理，
可以防止有xml没img或有img无xml情况，</p>
该代码处理多个文件夹或一个文件夹情况(有xml或无xml情况)。

'''

import os
import json
import xml.etree.ElementTree as ET
import cv2  # 无xml时候需要读取图片高与宽
# from cope_data.cope_utils import *
from tqdm import tqdm

import numpy as np
from lxml.etree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET



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


names_cls=[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

# names_cls = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

categories = {}
for i, cls in enumerate(names_cls):
    categories[i] = cls



# 按行读取txt格式文件
def read_txt(path):
    txt_info_lst = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            txt_info_lst.append(list(line.strip('\n').split()))
    txt_info_lst = np.array(txt_info_lst)
    return txt_info_lst

# 读取具有中文名称的图像
def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return cv_img  # 中文图片的保存


def saving_chinese_img(img, out_dir_img_root, img_suffix='jpg'):
    # 图像  保存路径.jpg  后缀
    cv2.imencode('.' + str(img_suffix), img)[1].tofile(out_dir_img_root)

def product_xml(name_img, boxes, codes, img=None, wh=None):
    '''
    :param img: 以读好的图片
    :param name_img: 图片名字
    :param boxes: box为列表
    :param codes: 为列表
    :return:
    '''
    if img is not None:
        width = img.shape[0]
        height = img.shape[1]
    else:
        assert wh is not None
        width = wh[0]
        height = wh[1]
    # print('xml w:{} h:{}'.format(width,height))

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = name_img  # 图片名字

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i, code in enumerate(codes):
        box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = code
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box[0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box[1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box[3]))

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)

    name = name_img[:-4] + '.xml'

    tree = ElementTree(node_root)

    # print('name:{},dom:{}'.format(name, dom))
    return tree, name



class Coco_json():
    '''
    输出json文件格式的训练与验证数据
    '''

    def __init__(self, root, out_dir=None):
        self.root = root
        self.out_dir = out_dir

    def train_multifiles(self, json_name='train.json', categories=None):
        '''
        json文件中的file_name包含文件夹/名字
        :param json_name: 保存json文件名字
        :param categories: 类别信息，为None则将self.root文件夹的名字作为类别信息
        :return:
        '''
        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

        image_id = 10000000
        anation_id = 10000000

        files = [get_strfile(file_str) for file_str in build_files(self.root)]

        categories = sorted(files) if categories is None else sorted(categories)

        info_dict = {}
        for file_category in files:
            file_path = os.path.join(self.root, file_category)
            names = os.listdir(file_path)

            count = 0
            for name in names:
                a, b = os.path.splitext(name)
                if b != '.xml' and a + '.xml' in names:
                    img_name = name

                    xml_file = os.path.join(file_path, a + '.xml')
                    try:
                        with open(xml_file) as f:
                            print('[doing file]:{}'.format(xml_file))
                            root = ET.parse(f).getroot()  # 获取根节点

                        image_id = image_id + 1

                        file_name = file_category + '/' + img_name  # 只记录图片名字

                        size = root.find('size')
                        width = int(size.find('width').text)
                        height = int(size.find('height').text)

                        image = {'file_name': file_name, 'height': height, 'width': width,
                                 'id': image_id}

                        for obj in root.findall('object'):

                            category = obj.find('name').text

                            if category not in categories:
                                print('skip code for num is spare {}'.format(category))
                                continue
                            # opzealot debug
                            if image not in json_dict['images']:
                                json_dict['images'].append(image)  # 将图像信息添加到json中

                            category_id = categories.index(category) + 1  # 给出box对应标签索引为类
                            anation_id = anation_id + 1
                            bndbox = obj.find('bndbox')
                            xmin = int(bndbox.find('xmin').text)  # bndbox.find('xmin').text
                            ymin = int(bndbox.find('ymin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymax = int(bndbox.find('ymax').text)

                            if (xmax <= xmin) or (ymax <= ymin):
                                print('{} error'.format(xml_file))  # 打印错误的box
                                continue
                            o_width = (xmax - xmin) + 1
                            o_height = (ymax - ymin) + 1

                            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                                   'bbox': [xmin, ymin, o_width, o_height],
                                   'category_id': category_id, 'id': anation_id, 'ignore': 0,
                                   'segmentation': []}
                            json_dict['annotations'].append(ann)

                        count += 1
                    except NotImplementedError:
                        print('xml {} file have a problem!'.format(xml_file))

            info_dict[file_category] = count

        for cid, cate in enumerate(categories):
            cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
            json_dict['categories'].append(cat)
        if self.out_dir is not None:

            build_dir(self.out_dir)
            out_dir = os.path.join(self.out_dir, json_name)
        else:
            out_dir = os.path.join(self.root, json_name)
        with open(out_dir, 'w') as f:
            json.dump(json_dict, f, indent=4)  # indent表示间隔长度
        print('[info_count]:{}'.format(info_dict))

    def train_multifiles_filename(self, json_name='train.json', categories=None):
        '''
        json文件中的file_name将只有名字，无文件夹
        :param json_name: 保存json文件名字
        :param categories: 类别信息，为None则将self.root文件夹的名字作为类别信息
        '''
        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

        image_id = 10000000
        anation_id = 10000000
        files = [get_strfile(file_str) for file_str in build_files(self.root)]

        categories = sorted(files) if categories is None else sorted(categories)

        info_dict = {}
        for file_category in files:
            file_path = os.path.join(self.root, file_category)
            names = os.listdir(file_path)

            count = 0
            for name in tqdm(names):
                a, b = os.path.splitext(name)
                if b != '.xml' and a + '.xml' in names:
                    img_name = name
                    xml_file = os.path.join(file_path, a + '.xml')
                    try:
                        with open(xml_file, 'r', encoding='utf-8') as f:
                            root = ET.parse(f).getroot()  # 获取根节点

                        image_id = image_id + 1

                        file_name = img_name  # 只记录图片名字

                        size = root.find('size')
                        width = int(size.find('width').text)
                        height = int(size.find('height').text)

                        image = {'file_name': file_name, 'height': height, 'width': width,
                                 'id': image_id}

                        for obj in root.findall('object'):

                            category = obj.find('name').text

                            if category not in categories:
                                print('skip code for num is spare {}'.format(category))
                                continue
                            # opzealot debug
                            if image not in json_dict['images']:
                                json_dict['images'].append(image)  # 将图像信息添加到json中

                            category_id = categories.index(category) + 1  # 给出box对应标签索引为类
                            anation_id = anation_id + 1
                            bndbox = obj.find('bndbox')
                            xmin = int(bndbox.find('xmin').text)  # bndbox.find('xmin').text
                            ymin = int(bndbox.find('ymin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymax = int(bndbox.find('ymax').text)

                            if (xmax <= xmin) or (ymax <= ymin):
                                print('{} error'.format(xml_file))  # 打印错误的box
                                continue
                            o_width = (xmax - xmin) + 1
                            o_height = (ymax - ymin) + 1

                            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                                   'bbox': [xmin, ymin, o_width, o_height],
                                   'category_id': category_id, 'id': anation_id, 'ignore': 0,
                                   'segmentation': []}
                            json_dict['annotations'].append(ann)

                        count += 1
                    except NotImplementedError:
                        print('xml {} file error!'.format(xml_file))

            info_dict[file_category] = count

        for cid, cate in enumerate(categories):
            cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
            json_dict['categories'].append(cat)
        if self.out_dir is not None:

            build_dir(self.out_dir)
            out_dir = os.path.join(self.out_dir, json_name)
        else:
            out_dir = os.path.join(self.root, json_name)
        with open(out_dir, 'w') as f:
            json.dump(json_dict, f, indent=4)  # indent表示间隔长度
        print('[info_count]:{}'.format(info_dict))

    def test_multifiles(self, json_name='test.json', categories=None):
        '''
        json文件中的file_name包含文件夹/名字
        :param json_name: 保存json文件名字
        :param categories: 类别信息，为None则将self.root文件夹的名字作为类别信息
        :return:
        '''
        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

        image_id = 10000000
        anation_id = 10000000
        files = [get_strfile(file_str) for file_str in build_files(self.root) if not file_str.endswith('.json')]

        categories = sorted(files) if categories is None else sorted(categories)
        info_dict = {'del_img_names': []}
        for file_category in tqdm(files):
            file_path = os.path.join(self.root, file_category)
            names = os.listdir(file_path)
            count = 0
            for name in tqdm(names):

                if os.path.splitext(name)[-1] != '.xml':
                    try:
                        image_id = image_id + 1
                        file_name = file_category + '/' + name  # 只记录图片名字
                        img = cv2.imread(os.path.join(self.root, file_name))
                        if img is not None:
                            img_shape = img.shape
                            height = int(img_shape[0])
                            width = int(img_shape[1])
                            image_id = image_id + 1  # 产生一个id
                            anation_id = anation_id + 1
                            image = {'file_name': file_name, 'height': height, 'width': width,
                                     'id': image_id}  # 高宽需要手动改嘛
                            json_dict['images'].append(image)

                            ann = {'area': None, 'iscrowd': 0, 'image_id': image_id,
                                   'bbox': [],
                                   'category_id': None, 'id': anation_id, 'ignore': 0,
                                   'segmentation': []}
                            json_dict['annotations'].append(ann)
                            count += 1
                        else:
                            os.remove(os.path.join(file_path, name))
                            info_dict['del_img_names'].append(name)
                    except NotImplementedError:

                        print('no exist img: {} '.format(name))

            info_dict[file_category] = count

        for cid, cate in enumerate(categories):
            cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
            json_dict['categories'].append(cat)
        if self.out_dir is not None:

            build_dir(self.out_dir)
            out_dir = os.path.join(self.out_dir, json_name)
        else:
            out_dir = os.path.join(self.root, json_name)
        with open(out_dir, 'w') as f:
            json.dump(json_dict, f, indent=4)  # indent表示间隔长度
        print('[position root]:{}\n[info_dict]:{}'.format(out_dir, info_dict))

    def test_multifiles_filename(self, json_name='test.json', categories=None):
        '''
         json文件中的file_name将只有名字，无文件夹
        :param json_name: 保存json文件名字
        :param categories: 类别信息，为None则将self.root文件夹的名字作为类别信息
        '''
        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

        image_id = 10000000
        anation_id = 10000000
        files = [get_strfile(file_str) for file_str in build_files(self.root)]

        categories = sorted(files) if categories is None else sorted(categories)

        info_dict = {}
        for file_category in files:
            file_path = os.path.join(self.root, file_category)
            names = os.listdir(file_path)
            count = 0
            for name in names:
                if os.path.splitext(name)[-1] != '.xml':
                    try:
                        image_id = image_id + 1
                        file_name = file_category + '/' + name  # 只记录图片名字
                        img_shape = cv2.imread(os.path.join(self.root, file_name)).shape
                        height = int(img_shape[0])
                        width = int(img_shape[1])
                        image_id = image_id + 1  # 产生一个id
                        anation_id = anation_id + 1
                        image = {'file_name': name, 'height': height, 'width': width,
                                 'id': image_id}  # 高宽需要手动改嘛
                        json_dict['images'].append(image)

                        ann = {'area': None, 'iscrowd': 0, 'image_id': image_id,
                               'bbox': [],
                               'category_id': None, 'id': anation_id, 'ignore': 0,
                               'segmentation': []}
                        json_dict['annotations'].append(ann)
                        count += 1
                    except NotImplementedError:
                        print('no exist img: {} '.format(name))
                    print('name:{}'.format(name))
            info_dict[file_category] = count

        for cid, cate in enumerate(categories):
            cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
            json_dict['categories'].append(cat)
        if self.out_dir is not None:

            build_dir(self.out_dir)
            out_dir = os.path.join(self.out_dir, json_name)
        else:
            out_dir = os.path.join(self.root, json_name)
        with open(out_dir, 'w') as f:
            json.dump(json_dict, f, indent=4)  # indent表示间隔长度
        print('[position root]:{}\n[info_count]:{}'.format(out_dir, info_dict))


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

def drwa_yolo(img, txt_info):
    height, width = img.shape[:2]
    for info in txt_info:
        label = str(info[0])
        x, y, w, h = float(info[1]) * width, float(info[2]) * height, float(info[3]) * width, float(info[4]) * height

        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
        img = chinese2img(img, label, coord=(int(x - w / 2), int(y - h / 2)))
    return img


def get_root_lst(folder_path, format='.jpg'):
    root_lst=[]
    name_lst=[]
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((format)):
                path = os.path.join(root, file)
                name = file
                root_lst.append(path)
                name_lst.append(name)
    return root_lst,name_lst

def main_yolov5txt2cocojson(root_data, txt_root=None):

    img_roots_lst, img_names_lst = get_root_lst(root_data, format='.jpg')
    txt_roots_lst, txt_names_lst = get_root_lst(txt_root, format='.txt')
    out_dir = build_dir(root_data)

    label_str_lst = []

    for i, img_root in tqdm(enumerate(img_roots_lst)):
        img_name = img_names_lst[i]
        txt_name = img_name[:-3] + 'txt'
        if txt_name in txt_names_lst:
            txt_index = list(txt_names_lst).index(str(txt_name))
            img = cv2.imread(img_root)
            height, width = img.shape[:2]

            txt_info = read_txt(txt_roots_lst[txt_index])

            # draw_img=drwa_yolo(img, txt_info)
            # show_img(draw_img)
            labels_lst, boxes_lst = [], []
            for info in txt_info:
                label_str = str(info[0])
                if label_str not in label_str_lst:
                    label_str_lst.append(label_str)

                x, y, w, h = float(info[1]) * width, float(info[2]) * height, float(info[3]) * width, float(
                    info[4]) * height
                xmin, ymin, xmax, ymax = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                labels_lst.append(label_str)
                boxes_lst.append([xmin, ymin, xmax, ymax])
            if len(labels_lst) > 0:
                tree, xml_name = product_xml(img_name, boxes_lst, labels_lst, wh=[w, h])
                tree.write(os.path.join(out_dir, xml_name), pretty_print=True)
    print('label:', label_str_lst)
    print('save root:',out_dir)


if __name__ == '__main__':
    root_path = r'E:\project\project_paper\project_LVF\code\yolov5-6.1-LVF\coco128\images\train'
    txt_root = r'E:\project\project_paper\project_LVF\code\yolov5-6.1-LVF\coco128\labels\train'

    main_yolov5txt2cocojson(root_path, txt_root=txt_root)
