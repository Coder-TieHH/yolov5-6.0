from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == "__main__":
    cocoGt = COCO('coco_format.json')        #标注文件的路径及文件名，json文件形式
    cocoDt = cocoGt.loadRes('predect_format.json')  #自己的生成的结果的路径及文件名，json文件形式

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


























