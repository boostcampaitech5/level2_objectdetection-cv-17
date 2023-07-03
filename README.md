## level2_ObjectDetection_cv17_sixseg

### 재활용 품목 분류를 위한 Object Detection

<img width="80%" src="https://github.com/boostcampaitech5/level2_objectdetection-cv-17/assets/70469008/50714a41-c564-483e-9adf-cf1f8db7b90e"/>

사진에서 쓰레기를 Detection 하는 모델을 만들어 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제을 해결해보고자 합니다.
해당 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다.
부디 지구를 위기로부터 구해주세요! 🌎

#### Team Members

강대호
강정우
박혜나
서지훈
원유석
정대훈

#### 실험 진행 순서

1. EDA
2. select pretrained model (2 stage, 1 stage)
3. data augmentation
4. stratified group kfold
5. TTA
6. ensemble

#### 최종 활용 모델

1. faster R-CNN
2. retinaNet
3. yolo_v8

#### Wrap-up Report

https://drive.google.com/file/d/1ulYuJpxpaAYhkGpdKQsEPm2xAiv9fMH6/view?usp=sharing

#### 평가 Metric

- mAP50(Mean Average Precision)

#### Dataset

- number of images : 9754
  - train: 4883
  - test: 4871
- number of class : 10
- labels : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- image size : (1024, 1024)

#### annotation file

- coco format (images, annotations)
  - images
    - id: 파일 내 image 고유 id, ex) 1
    - height: 1024
    - width: 1024
    - filename: ex) train/002.jpg
  - annotations
    - id: 파일 안에 annotation 고유 id, ex) 1
    - bbox: 객체가 존재하는 박스의 좌표 (xmin, ymin, w, h)
    - area: 객체가 존재하는 박스의 크기
    - category_id: 객체가 해당하는 class의 id
    - image_id: annotation이 표시된 이미지 고유 id
- train.json: train image에 대한 annotation file (coco format)
- test.json: test image에 대한 annotation file (coco format)
