import json
import numpy as np
import pandas as pd
from pycocotools.coco import COCO


def get_multi_class_list(dataset_dir):
    ins_multi_class_dict = {}
    coco = COCO(dataset_dir)

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info['id']) 
        anns = coco.loadAnns(ann_ids)
        file_name = image_info['file_name']

        for ann in anns:
            if ann['image_id'] not in ins_multi_class_dict:
                ins_multi_class_dict[ann['image_id']] = []
            ins_multi_class_dict[ann['image_id']].append(ann['category_id'])
        
    ins_multi_class_list = []
    for key,list_ in ins_multi_class_dict.items():
        co = len(list_)
        if co==1:
            co =0
        elif co <6:
            co=10
        else:
            co=20
        
        for ele in list_:
            ins_multi_class_list.append(co+ele)

    return ins_multi_class_list