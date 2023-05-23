# get valication data
import csv
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
# check distribution
import pandas as pd
from collections import Counter
from torch.utils.data import Subset
import mean_average_precision_for_boxes #Object Det 미션-1 mAP Metric

def get_anotations_by_image(annotations, image_idx):
    anns = []
    for ann in annotations:
        if ann['image_id'] in image_idx:
            anns.append(ann)
    return anns


def get_incorrect_data(dir_true_json, dir_pred_csv, dir_save_json, iou_threshold, criterion):

    # load json: modify the path to your own ‘dir_true_json’ file
    with open(dir_true_json) as f:
        # print(data)
        data = json.load(f)
        info = data['info']
        licenses = data['licenses']
        images = data['images']
        categories = data['categories']
        annotations = data['annotations']

    solution = {}
    for i in images:
        solution[i["id"]] = []

    n=0
    for ann in annotations:
        #print(ann["image_id"])
        if not solution[ann["image_id"]]:
            n=0

        solution[ann["image_id"]].append( [
            '1',
            ann['category_id'],
            ann["bbox"][0], 
            ann["bbox"][0]+ann["bbox"][2],
            ann["bbox"][1],
            ann["bbox"][1]+ann["bbox"][3] 
            ] )
        
        n+=1

    #print(solution)

    # solution  = [  
    #               0: [[idx, Class,1, a,b,c,d], [idx, Class,1, a,b,c,d], [idx, Class,1, a,b,c,d]],
    #               1: [[idx, Class,1, a,b,c,d], [idx, Class,1, a,b,c,d]                        ]] 
    #              ]

    answer = {}
    f = open(dir_pred_csv, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    a =1
    for line in rdr:
        if a:
            a=0
            continue

        key = int(line[1][-8:-4])
        answer[key]=[]
        ln = list(map(float, line[0].split()))
        n=0
        for t in range(len(ln)//6):
            answer[key].append( ['1'] + [int(ln[6*t])] +[ln[6*t+1]] +[ln[6*t+2]] +[ln[6*t+4]] +[ln[6*t+3]] +[ln[6*t+5]])
            n+=1
    f.close()

    # answer  = [  
    #               0: [[idx, Class,p, a,b,c,d], [idx, Class,p, a,b,c,d], [idx, Class,p, a,b,c,d]],
    #               1: [[idx, Class,p, a,b,c,d], [idx, Class,p, a,b,c,d]                        ]] 
    #            ]

    #print(answer)

    incorrect_list = []
    for key in answer.keys():
        pred_boxes = answer[key]
        true_boxes = solution[key]
        n_class = len(categories)
        mean_ap, average_precisions = mean_average_precision_for_boxes(true_boxes, pred_boxes, iou_threshold)

        #print(mean_ap)
        if mean_ap < criterion:
            incorrect_list.append(key)

    #print(incorrect_list)
    #print(len(incorrect_list))


    new_images = []
    for idx in range(len(incorrect_list)):
        new_images.append(images[idx])

    with open(dir_save_json,'w') as train_writer:
            json.dump({
                'info' : info,
                'licenses' : licenses,
                'images' : new_images,
                'categories' : categories,
                'annotations' : get_anotations_by_image(annotations, incorrect_list)

            }, train_writer, indent=2)



    print('\nCreating %s... Done !' %dir_save_json)
    print(".")



if __name__ == '__main__':
    dir_true_json = './kfold3_val.json'
    dir_pred_csv = './submission_fold3.csv'
    dir_save_json = './incorrect.json'
    iou_threshold = 0.5
    criterion = 0.4
    get_incorrect_data(dir_true_json, dir_pred_csv, dir_save_json, iou_threshold, criterion)