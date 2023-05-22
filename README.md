## level2_ObjectDetection_cv17_sixseg
### 재활용 품목 분류를 위한 Object Detection
#### Members

강대호
강정우
박혜나
서지훈
원유석
정대훈

#### 실험 진행 순서

- EDA
- select pretrained model (2 stage, 1 stage)
- data augmentation
- stratified group kfold
- TTA
- ensemble

#### 최종 활용 모델

1. faster R-CNN
2. retinaNet
3. yolo_v8

#### Wrap up Report
https://www.notion.so/_CV_-17-bf0ef1d1236144f59028f9c8a48da215

#### Dataset
- number of images : 9754
    - train: 4883
    - test: 4871
- number of class : 10 
- labels : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- image size : (1024, 1024)

#### annotation file
- coco format (images, annotations)
1. images
    - id: 파일 안에서 image 고유 id, ex) 1
    - height: 1024
    - width: 1024
    - filename: ex) train/002.jpg
2. annotations
    - id: 파일 안에 annotation 고유 id, ex) 1
    - bbox: 객체가 존재하는 박스의 좌표 (xmin, ymin, w, h)
    - area: 객체가 존재하는 박스의 크기
    - category_id: 객체가 해당하는 class의 id
    - image_id: annotation이 표시된 이미지 고유 id
- train.json: train image에 대한 annotation file (coco format)
- test.json: test image에 대한 annotation file (coco format)