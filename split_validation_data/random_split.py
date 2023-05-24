# random split 1
from sklearn.model_selection import train_test_split
import json

with open('../../dataset/train.json', 'r') as f:
    dataset = json.load(f)

train_idx, val_idx = train_test_split([image['id'] for image in dataset['images']], test_size=0.2, random_state=42)
print(len(train_idx),len(val_idx))

train_set = {'images': [], 'annotations': [], 'categories': dataset['categories']}
val_set = {'images': [], 'annotations': [], 'categories': dataset['categories']}

for idx in train_idx:
    train_set['images'].append(dataset['images'][idx])
    
    for anns in dataset['annotations']:
        if anns['image_id']==dataset['images'][idx]['id']:
            train_set['annotations'].append(anns)

for idx in val_idx:
    val_set['images'].append(dataset['images'][idx])
    
    for anns in dataset['annotations']:
        if anns['image_id']==dataset['images'][idx]['id']:
            val_set['annotations'].append(anns)


# train set annotation 파일 저장
with open('train_split.json', 'w') as f:
    json.dump({'images': train_set['images'], 'annotations': train_set['annotations'], 'categories': train_set['categories']}, f)

# validation set annotation 파일 저장
with open('val_split.json', 'w') as f:
    json.dump({'images': val_set['images'], 'annotations': val_set['annotations'], 'categories': val_set['categories']}, f)


# random split 2
# from sklearn.model_selection import train_test_split
# import json

# with open('../dataset/train.json', 'r') as f:
#     dataset = json.load(f)

# train_data, val_data = train_test_split(dataset['annotations'], test_size=0.2, random_state=42)

# # train set annotation 파일 저장
# with open('train_split.json', 'w') as f:
#     json.dump({'images': dataset['images'], 'annotations': train_data, 'categories': dataset['categories']}, f)

# # validation set annotation 파일 저장
# with open('val_split.json', 'w') as f:
#     json.dump({'images': dataset['images'], 'annotations': val_data, 'categories': dataset['categories']}, f)