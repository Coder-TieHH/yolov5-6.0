
import os
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# print(ROOT)
from Map import Computer_map
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import torch
import argparse
import cv2
import numpy as np
from utils.general import non_max_suppression,scale_coords


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/root/VisDrone2019/VisDrone2019-DET-test-dev", help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/root/yolov5/runs/EMA_small_dimension/exp/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold，default=0.001')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='NMS IoU threshold，default=0.6')
    parser.add_argument('--imgsz', '--img', '--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--save_dir',  default='/root/yolov5/runs/EMA_small_dimension/val2', help='图像保存路径')
    parser.add_argument('--save_img', default=False, help='保存框图像查看')
    opt = parser.parse_args()

    return opt


def build_dir( out_dir):
    # 构建文件
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir
def draw_bbox( img, pred,
                  bbox_color='red',
                  text_color='red',
                  thickness=1,
                  font_scale=0.5
                  ):
    color = dict(red=(0, 0, 255),
                 green=(0, 255, 0),
                 blue=(255, 0, 0),
                 cyan=(255, 255, 0),
                 yellow=(0, 255, 255),
                 magenta=(255, 0, 255),
                 white=(255, 255, 255),
                 black=(0, 0, 0))


    for j, p in enumerate(pred):
        x1, y1, x2, y2 = np.array(p[:4]).astype(np.int32)
        score, cat =p[-2:]
        bbox_color_new = color[bbox_color]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color_new, thickness=thickness)
        score = round(score, 4)
        text_color_new = color[text_color]
        label_text = '{}:{}'.format(str(cat), str(score))
        cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color_new)
    return img

def init_model(weights):

    model = attempt_load(weights, map_location=device)
    model = model.eval()
    return model

def computer_main(opt, model):
    '''
    data_root:任何文件夹，但必须保证每个图片与对应xml必须放在同一个文件夹中
    model:模型，用于预测
    '''

    stride=32
    img_size=[opt.imgsz, opt.imgsz]

    C = Computer_map()
    img_root_lst = C.get_img_root_lst(opt.source)  # 获得图片绝对路径与图片产生image_id映射关系

    # 在self.coco_json中保存categories，便于产生coco_json和predetect_json
    categories = model.names  # 可以给txt路径读取，或直接给列表  #*********************得到classes,需要更改的地方***********##
    C.get_categories(categories)

    C.yolov5txt2cocojson(img_root_lst,out_dir=None,save_img=False)
    # 产生coco_json格式
    # xml_root_lst = [name[:-3] + 'xml' for name in img_root_lst]
    # for xml_root in xml_root_lst: C.xml2cocojson(xml_root)  # 产生coco json 并保存到self.coco_json中

    if opt.save_img:build_dir(opt.save_dir)
    # 产生预测的json
    for img_path in img_root_lst:
        img0 = cv2.imread(img_path)
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)
        # print("图片原始尺寸：{}\t模型预测尺寸：{}".format(img0.shape,im.shape))

        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im)[0]  ####**********************需要更改的地方***********************####

        result = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, multi_label=True)
        det = result[0]
        # result, classes = parse_result['result'], parse_result['classes']
        if len(det)>0:
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
        det = det.cpu().numpy() if det.is_cuda else det.numpy()  # 处理为cuda上的数据或cpu转numpy格式
        det = [[d[0],d[1],d[2],d[3],d[4], categories[int(d[5])] ] for d in det] # 给定名称name标签
        # det 格式为列表[x1,y1,x2,y2,score,label],若无结果为空
        img_name = C.get_strfile(img_path)
        C.detect2json(det, img_name)

        if opt.save_img:
            img=draw_bbox(img0,det)
            cv2.imwrite(os.path.join(opt.save_dir,img_name),img)
    map_value = C.computer_map()  # 计算map,返回 [mAP@0.5:0.95， mAP@0.5, mAP@0.75, ... ]
    yolo_best = 0.9*map_value[0]+0.1*map_value[1]


    return map_value, yolo_best

if __name__ == '__main__':
    opt = parse_opt()
    device = select_device(opt.device)
    model = init_model(opt.weights)
    map_value=computer_main(opt, model)

    print(map_value)
