import numpy as np
import os
import json
import cv2
class Computer_map():
    '''
    主代码样列
    def computer_main(data_root, model):#data_root:任何文件夹，但必须保证每个图片与对应xml必须放在同一个文件夹中，model:模型，用于预测
        C = Computer_map()
        img_root_lst = C.get_img_root_lst(data_root)  # 获得图片绝对路径与图片产生image_id映射关系

        # 在self.coco_json中保存categories，便于产生coco_json和predetect_json
        categories = model.CLASSES  # 可以给txt路径读取，或直接给列表  #*********************得到classes,需要更改的地方***********##
        C.get_categories(categories)

        # 产生coco_json格式
        xml_root_lst = [name[:-3] + 'xml' for name in img_root_lst]
        for xml_root in xml_root_lst: C.xml2cocojson(xml_root)  # 产生coco json 并保存到self.coco_json中

        # 产生预测的json
        for img_path in img_root_lst:

            parse_result = predict(model, img_path)  ####**********************需要更改的地方***********************####

            result, classes = parse_result['result'], parse_result['classes']
            # restult 格式为列表[x1,y1,x2,y2,score,label],若无结果为空
            img_name = C.get_strfile(img_path)
            C.detect2json(result, img_name)
        C.computer_map()  # 计算map

    '''

    def __init__(self):
        self.img_format = ['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg']
        self.coco_json = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}
        self.predetect_json = []  # 保存字典
        self.image_id = 10000000  # 图像的id，每增加一张图片便+1
        self.anation_id = 10000000
        self.imgname_map_id = {}  # 图片名字映射id


    def read_txt(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read().splitlines()
        return content

    def get_categories(self, categories):
        '''
        categories:为字符串，指绝对路径；为列表，指类本身
        return:将categories存入coco json中
        '''
        if isinstance(categories, str):

            categories = self.read_txt(categories)


        elif isinstance(categories, list or tuple):
            categories = list(categories)

        category_json = [{"supercategory": cat, "id": i + 1, "name": cat} for i, cat in enumerate(categories)]
        self.coco_json['categories'] = category_json

    def computer_map(self, coco_json_path=None, predetect_json_path=None):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from collections import defaultdict
        import time
        import json
        from pycocotools import mask as maskUtils
        import numpy as np
        # 继承修改coco json文件
        class COCO_modify(COCO):
            def __init__(self, coco_json_data=None):
                """
                Constructor of Microsoft COCO helper class for reading and visualizing annotations.
                :param annotation_file (str): location of annotation file
                :param image_folder (str): location to the folder that hosts images.
                :return:
                """
                # load dataset
                self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
                self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
                if coco_json_data is not None:
                    print('loading annotations into memory...')
                    tic = time.time()
                    if isinstance(coco_json_data, str):
                        with open(coco_json_data, 'r') as f:
                            dataset = json.load(f)
                        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
                        print('Done (t={:0.2f}s)'.format(time.time() - tic))
                    else:
                        dataset = coco_json_data
                    self.dataset = dataset
                    self.createIndex()

            def loadRes(self, predetect_json_data):
                import copy
                """
                Load result file and return a result api object.
                :param   resFile (str)     : file name of result file
                :return: res (obj)         : result api object
                """
                res = COCO_modify()
                res.dataset['images'] = [img for img in self.dataset['images']]

                print('Loading and preparing results...')
                tic = time.time()

                if isinstance(predetect_json_data, str):
                    with open(predetect_json_data, 'r') as f:
                        anns = json.load(f)

                    print('Done (t={:0.2f}s)'.format(time.time() - tic))
                else:
                    anns = predetect_json_data

                assert type(anns) == list, 'results in not an array of objects'
                annsImgIds = [ann['image_id'] for ann in anns]
                assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
                    'Results do not correspond to current coco set'
                if 'caption' in anns[0]:
                    imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
                    res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
                    for id, ann in enumerate(anns):
                        ann['id'] = id + 1
                elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
                    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                    for id, ann in enumerate(anns):
                        bb = ann['bbox']
                        x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                        if not 'segmentation' in ann:
                            ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                        ann['area'] = bb[2] * bb[3]
                        ann['id'] = id + 1
                        ann['iscrowd'] = 0
                elif 'segmentation' in anns[0]:
                    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                    for id, ann in enumerate(anns):
                        # now only support compressed RLE format as segmentation results
                        ann['area'] = maskUtils.area(ann['segmentation'])
                        if not 'bbox' in ann:
                            ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                        ann['id'] = id + 1
                        ann['iscrowd'] = 0
                elif 'keypoints' in anns[0]:
                    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                    for id, ann in enumerate(anns):
                        s = ann['keypoints']
                        x = s[0::3]
                        y = s[1::3]
                        x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                        ann['area'] = (x1 - x0) * (y1 - y0)
                        ann['id'] = id + 1
                        ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
                print('DONE (t={:0.2f}s)'.format(time.time() - tic))

                res.dataset['annotations'] = anns
                res.createIndex()
                return res

        coco_json_data = coco_json_path if coco_json_path is not None else self.coco_json
        cocoGt = COCO_modify(coco_json_data)  # 标注文件的路径及文件名，json文件形式
        predetect_json_data = predetect_json_path if predetect_json_path is not None else self.predetect_json
        cocoDt = cocoGt.loadRes(predetect_json_data)  # 自己的生成的结果的路径及文件名，json文件形式

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map_value = cocoEval.stats
        return map_value

    def get_img_root_lst(self, root_data):
        import os
        img_root_lst = []
        for dir, file, names in os.walk(root_data):
            img_lst = [os.path.join(dir, name) for name in names if name[-3:] in self.img_format]
            img_root_lst = img_root_lst + img_lst
            for na in img_lst:  # 图片名字映射image_id
                self.image_id += 1
                self.imgname_map_id[self.get_strfile(na)] = self.image_id
        return img_root_lst  # 得到图片绝对路径

    def get_strfile(self, file_str, pos=-1):
        '''
        得到file_str / or \\ 的最后一个名称
        '''
        endstr_f_filestr = file_str.split('\\')[pos] if '\\' in file_str else file_str.split('/')[pos]
        return endstr_f_filestr

    def read_xml(self, xml_root):
        '''
        :param xml_root: .xml文件
        :return: dict('cat':['cat1',...],'bboxes':[[x1,y1,x2,y2],...],'whd':[w ,h,d])
        '''

        import xml.etree.ElementTree as ET
        import os

        dict_info = {'cat': [], 'bboxes': [], 'box_wh': [], 'whd': []}
        if os.path.splitext(xml_root)[-1] == '.xml':
            tree = ET.parse(xml_root)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
            root = tree.getroot()  # 获取根节点
            whd = root.find('size')
            whd = [int(whd.find('width').text), int(whd.find('height').text), int(whd.find('depth').text)]
            xml_filename = root.find('filename').text
            dict_info['whd'] = whd
            dict_info['xml_filename'] = xml_filename
            for obj in root.findall('object'):  # 找到根节点下所有“object”节点
                cat = str(obj.find('name').text)  # 找到object节点下name子节点的值（字符串）
                bbox = obj.find('bndbox')
                x1, y1, x2, y2 = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
                b_w = x2 - x1 + 1
                b_h = y2 - y1 + 1

                dict_info['cat'].append(cat)
                dict_info['bboxes'].append([x1, y1, x2, y2])
                dict_info['box_wh'].append([b_w, b_h])

        else:
            print('[inexistence]:{} suffix is not xml '.format(xml_root))
        return dict_info

    def xml2cocojson(self, xml_root):
        '''
        处理1个xml，将其真实json保存到self.coco_json中
        '''
        assert len(self.coco_json['categories']) > 0, 'self.coco_json[categories] must exist v'
        categories = [cat_info['name'] for cat_info in  self.coco_json['categories']]
        xml_info = self.read_xml(xml_root)
        if len(xml_info['cat']) > 0:
            xml_filename = xml_info['xml_filename']
            xml_name = self.get_strfile(xml_root)
            img_name = xml_name[:-3] + xml_filename[-3:]
            # 转为coco json时候，若add_file为True则在coco json文件的file_name增加文件夹名称+图片名字

            image_id = self.imgname_map_id[img_name]
            w, h, d = xml_info['whd']
            # 构建json文件字典
            image_json = {'file_name': img_name, 'height': h, 'width': w, 'id': image_id}
            ann_json = []
            for i, category in enumerate(xml_info['cat']):
                # 表示有box存在，可以添加images信息

                category_id = categories.index(category) + 1  # 给出box对应标签索引为类
                self.anation_id = self.anation_id + 1
                xmin, ymin, xmax, ymax = xml_info['bboxes'][i]

                o_width, o_height = xml_info['box_wh'][i]

                if (xmax <= xmin) or (ymax <= ymin):
                    print('code:[{}] will be abandon due to  {} min of box w or h more than max '.format(category,
                                                                                                         xml_root))  # 打印错误的box
                else:
                    ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                           'bbox': [xmin, ymin, o_width, o_height],
                           'category_id': category_id, 'id': self.anation_id, 'ignore': 0,
                           'segmentation': []}
                    ann_json.append(ann)

            if len(ann_json) > 0:  # 证明存在 annotation
                for ann in ann_json:  self.coco_json['annotations'].append(ann)
                self.coco_json['images'].append(image_json)

    def detect2json(self, predetect_result, img_name, score_thr=-1):
        '''
        predetect_result:为列表，每个列表中包含[x1, y1, x2, y2, score, label]
        img_name: 图片的名字
        '''
        if len(predetect_result) > 0:
            categories = [cat_info['name'] for cat_info in  self.coco_json['categories']]
            for result in predetect_result:
                x1, y1, x2, y2, score, label = result
                if score > score_thr:
                    w, h = int(x2 - x1), int(y2 - y1)
                    x1, y1 = int(x1), int(y1)
                    img_name_new = self.get_strfile(img_name)
                    image_id = self.imgname_map_id[img_name_new]
                    category_id = list(categories).index(label) + 1
                    detect_json = {
                        "area": w * h,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [
                            x1,
                            y1,
                            w,
                            h
                        ],
                        "category_id": category_id,
                        "id": image_id,
                        "ignore": 0,
                        "segmentation": [],
                        "score": score
                    }
                    self.predetect_json.append(detect_json)

    def write_json(self,out_dir):
        import os
        import json
        coco_json_path=os.path.join(out_dir,'coco_json_data.json')
        with open(coco_json_path, 'w') as f:
            json.dump(self.coco_json, f, indent=4)  # indent表示间隔长度
        predetect_json_path=os.path.join(out_dir,'predetect_json_data.json')
        with open(predetect_json_path, 'w') as f:
            json.dump(self.predetect_json, f, indent=4)  # indent表示间隔长度

    def read_yaml(self,yaml_path):
        import yaml
        f = open(yaml_path, 'rb')
        cfg = yaml.load(f, Loader=yaml.FullLoader)

        return cfg

    def yolov5txt2cocojson(self,img_roots_lst, out_dir=None,  save_img=False):
        '''
        json文件中的file_name包含文件夹/名字
        :param json_name: 保存json文件名字
        :param categories: 类别信息，为None则将self.root文件夹的名字作为类别信息
        :return:
        '''


        categories = {i: c["supercategory"] for i, c in enumerate(self.coco_json['categories'])}


        info_dict = {'label_int': []}

        for i, img_root in enumerate(img_roots_lst):
            img_name = self.get_strfile(img_root)

            img = cv2.imread(img_root)
            txt_root = img_root.replace('images','labels')[:-3]+'txt'

            image_id = self.imgname_map_id[img_name]
            height, width = img.shape[:2]


            image = {'file_name': img_name, 'height': height, 'width': width, 'id': image_id}

            txt_info = self.read_txt(txt_root)

            for info in txt_info:
                label_int = int(info[0])
                if label_int not in info_dict['label_int']:
                    info_dict['label_int'].append(label_int)
                label = categories[label_int]
                if label not in categories.values():
                    print('skip code for num is spare {}:{}'.format(label_int, label))
                    continue
                # opzealot debug
                if image not in self.coco_json['images']:
                    self.coco_json['images'].append(image)  # 将图像信息添加到json中
                x, y, w, h = float(info[1]) * width, float(info[2]) * height, float(info[3]) * width, float(
                    info[4]) * height

                img = cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                    (255, 0, 0), 2)
                img = self.chinese2img(img, label, coord=(int(x - w / 2), int(y - h / 2)))

                category_id = label_int + 1  # 给出box对应标签索引为类

                self.anation_id = self.anation_id + 1
                xmin, ymin, o_width, o_height = int(x - w / 2), int(y - h / 2), int(w), int(h)

                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                       'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': self.anation_id, 'ignore': 0,
                       'segmentation': []}
                self.coco_json['annotations'].append(ann)
            if save_img and out_dir is not None:
                save_img_path = self.build_dir(os.path.join(out_dir, 'save_txt_draw_img'))
                cv2.imwrite(os.path.join(save_img_path, img_name), img)

        if out_dir is not None:
            self.build_dir(out_dir)
            with open(os.path.join(out_dir,'coco.json'), 'w') as f:
                json.dump(self.coco_json, f, indent=4)  # indent表示间隔长度
            # print('[info_count]:{}'.format(info_dict))

    def get_root_lst(self,root, suffix='jpg', suffix_n=3):
        root_lst, name_lst = [], []
        import os
        for dir, file, names in os.walk(root):
            root_lst = root_lst + [os.path.join(dir, name) for name in names if name[-suffix_n:] == suffix]
            name_lst = name_lst + [name for name in names if name[-suffix_n:] == suffix]

        return root_lst, name_lst

    # 按行读取txt格式文件
    def read_txt(self,path):
        txt_info_lst = []
        with open(path, "r", encoding='utf-8') as f:
            for line in f:
                txt_info_lst.append(list(line.strip('\n').split()))
        txt_info_lst = np.array(txt_info_lst)
        return txt_info_lst

    def chinese2img(self,img, str, coord=(0, 0), label_size=20, label_color=(255, 0, 0)):
        # 将具有中文的字符打印到图上
        from PIL import Image, ImageDraw, ImageFont
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pilimg = Image.fromarray(cv2img)

        # PIL图片上打印汉字
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        font = ImageFont.truetype("/root/yolov5/SIMSUN.TTC", label_size, encoding="utf-8")
        # font = ImageFont.truetype("./simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text(tuple(coord), str, label_color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        # PIL图片转cv2 图片
        cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        return cv2charimg

    def build_dir(self,out_dir):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        return out_dir







