# get valication data
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# check distribution
import pandas as pd
from collections import Counter
from pycocotools.coco import COCO
from torch.utils.data import Subset



def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]

def get_anotations_by_image(annotations, image_idx):
    anns = []
    for ann in annotations:
        if ann['image_id'] in image_idx:
            anns.append(ann)
    return anns

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

def get_val_data(dataset_dir, base_dataset_dir):

    # load json: modify the path to your own �쁳rain.json�� file
    annotation = dataset_dir # dataset file 寃쎈줈
    print(annotation)

    with open(annotation) as f:
        # print(data)
        data = json.load(f)
        info = data['info']
        licenses = data['licenses']
        images = data['images']
        categories = data['categories']
        annotations = data['annotations']

    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']),1))
    Y= get_multi_class_list(dataset_dir)
    y = np.array(Y)
            
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        kfold_train_annotation = base_dataset_dir + '/kfold%s_train.json' %str(fold_ind)
        kfold_val_annotation = base_dataset_dir + '/kfold%s_val.json' %str(fold_ind)
        
        print("TRAIN:", groups[train_idx])
        print(" ", y[train_idx])
        print(" TEST:", groups[val_idx])
        print(" ", y[val_idx])

        # print(train_idx, val_idx)
        train_image_idxs =  list(set(groups[train_idx]))
        val_image_idxs =  list(set(groups[val_idx]))

        with open(kfold_train_annotation,'w') as train_writer:
            json.dump({
                'info' : info,
                'licenses' : licenses,
                'images' : [images[i] for i in train_image_idxs],
                'categories' : categories,
                'annotations' : get_anotations_by_image(annotations, train_image_idxs)
            }, train_writer, indent=2)
        print('\nCreating %s... Done !' %kfold_train_annotation)

        with open(kfold_val_annotation,'w') as val_writer:
            json.dump({
                'info' : info,
                'licenses' : licenses,
                'images' : [images[i] for i in val_image_idxs],
                'categories' : categories,
                'annotations' : get_anotations_by_image(annotations, val_image_idxs)
            }, val_writer, indent=2)
        print('Creating %s... Done !\n' %kfold_val_annotation)

    
    if __name__ == '__main__':
        print('\n if main ?')
        # check distribution
        distrs = [get_distribution(y)]
        index = ['training set']
        
        for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X,y, groups)):
            train_y, val_y = y[train_idx], y[val_idx]
            train_gr, val_gr = groups[train_idx], groups[val_idx]

            assert len(set(train_gr) & set(val_gr)) == 0 
            distrs.append(get_distribution(train_y))
            distrs.append(get_distribution(val_y))
            index.append(f'train - fold{fold_ind}')
            index.append(f'val - fold{fold_ind}')

            # train_set = Subset(train_gr, train_idx)
            # val_set = Subset(val_gr, val_idx)

            print(len(train_y), len(train_gr))
            print(len(val_y), len(val_gr))

        categories = [d['name'] for d in data['categories']]
        #print(pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)]))

        # print(train_y)
        # print(train_gr)
        # print(val_y)
        # print(val_gr)

    else:
        return groups, y


if __name__ == '__main__':
    dataset_dir = '../../dataset/train.json'
    base_dataset_dir = '../../dataset'
    get_val_data(dataset_dir, base_dataset_dir)