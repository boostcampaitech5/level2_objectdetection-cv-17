# get valication data
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# check distribution
import pandas as pd
from collections import Counter
from torch.utils.data import Subset

# multi class validation
from multi_class_validation import get_multi_class_list

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

def get_val_data(dataset_dir, base_dataset_dir):

    # load json: modify the path to your own ‘train.json’ file
    annotation = dataset_dir # dataset file 경로
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
    Y= get_multi_class_list(dataset_dir) # multi_class_validation
    y = np.array(Y) # multi_class_validation
    # y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        # kfold_train_annotation = base_dataset_dir + '/kfold%s_train.json' %str(fold_ind)
        # kfold_val_annotation = base_dataset_dir + '/kfold%s_val.json' %str(fold_ind)
        kfold_train_annotation = base_dataset_dir + 'strGroupKfold%s_train.json' %str(fold_ind)
        kfold_val_annotation = base_dataset_dir + 'strGroupKfold%s_val.json' %str(fold_ind)
        
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
        print(pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)]))

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