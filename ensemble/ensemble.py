import pandas as pd
from ensemble_boxes import *
import numpy as np
import os
from tqdm import tqdm
import argparse

ensemble_folder='../ensemble/'
ensemble_files=os.listdir(ensemble_folder)
output=[]
for f in ensemble_files:
    output.append(pd.read_csv(os.path.join(ensemble_folder,f)))

def divide(pred):
    labels=[]
    scores=[]
    bboxes=[]
    if pred and type(pred)==str:
        d=pred.split(" ")[:-1]
        for i in range(int(len(d)/6)):
            labels.append(int(d[6*i]))
            scores.append(float(d[6*i+1]))
            x1=float(d[6*i+2])
            y1=float(d[6*i+3])
            x2=float(d[6*i+4])
            y2=float(d[6*i+5])
            bboxes.append([x1/1024,y1/1024,x2/1024,y2/1024])

    return labels,scores,bboxes

def change(bboxes, scores, labels):
    result=[]
    for i in range(len(labels)):
        result+=[int(labels[i]),scores[i],bboxes[i][0]*1024,bboxes[i][1]*1024,bboxes[i][2]*1024,bboxes[i][3]*1024]
    result=map(str,result)
    return ' '.join(result)

def main(args):
    submission=pd.DataFrame(index=range(len(output[0])),columns=['PredictionString','image_id'])
    submission['image_id']=output[0]['image_id']

    iou_thr = args.iou_thr
    skip_box_thr = args.skip_box_thr
    weights = args.weight

    print("="*25,"Ensemble start","="*25)

    for img in tqdm(range(len(output[0])),bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
        label_list=[]
        score_list=[]
        bbox_list=[]
        for models in output:
            label, score, bbox=divide(models['PredictionString'][img])
            label_list.append(label)
            score_list.append(score)
            bbox_list.append(bbox)
        bboxes,scores, labels=weighted_boxes_fusion(bbox_list, score_list, label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        #bboxes,scores, labels=weighted_boxes_fusion(bbox_list, score_list, label_list, weights=weights, iou_thr=iou_thr)
        out=change(bboxes,scores,labels)
        submission['PredictionString'][img]=out

    submission.to_csv(os.path.join(ensemble_folder,'ENSEMBLE.csv'),index=None)
    print("="*25,"Ensemble finish","="*25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for creating ensemble csv file')
    parser.add_argument('--iou_thr', type = float, default = 0.5)
    parser.add_argument('--skip_box_thr', type=float, default = 0.0001)
    parser.add_argument('--weight',  nargs='+',type = int, default = [1]*len(ensemble_files), help = 'List of info columns')


    args = parser.parse_args()

    main(args)