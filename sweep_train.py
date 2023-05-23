# 모듈 import
from pathlib import Path
import argparse
import glob
import os
import re
import copy

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import random

import wandb

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def train():
    
    wandb_default_config = {    "learning_rate": args.lr,
                                "optimizer": args.optimizer,
                                "batch_size": args.batch_size,
                                "anchor_ratio": args.anchor_ratio,
                                }
    wandb.init(                 project= "project",
                                entity= "entity",
                                config = wandb_default_config)
    
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py')

    root='/opt/ml/dataset/'
    
    save_dir = increment_path(os.path.join(args.work_dir, args.name))

    img_scale = (512, 512) 
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    # dataset config 수정
    train_pipeline = [
        dict(type='Resize', img_scale=img_scale, keep_ratio=True),
        dict(type='RandomFlip',direction='horizontal', flip_ratio=0.3),
        dict(type='RandomFlip',direction='vertical', flip_ratio=0.3),
        dict(type='Mosaic', img_scale=img_scale),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
    train_dataset = dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=root + 'train_split.json',
            img_prefix=root,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
        filter_empty_gt=False,
        ),
    pipeline=train_pipeline
    )
    cfg.data.train=train_dataset
    cfg.data.train['dataset']['classes'] = classes
    cfg.workflow = [('train', 1), ('val', 1)]
    
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + 'val_split.json' # test json 정보
    cfg.data.val.pipeline[2]['img_scale'] = img_scale # Resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = img_scale # Resize

    cfg.data.samples_per_gpu = wandb.config.batch_size # 학습시 배치크기 설정 default : 4

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs = {},
            interval = 50,
            log_checkpoint = True,
            log_checkpoint_metadata=True,
            num_eval_images=10)
    ]

    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    cfg.work_dir = save_dir # 학습결과 저장 경로설정

    cfg.model.roi_head.bbox_head.num_classes = 10 # bounding 박스 예측에 사용할 클래스수 설정
    cfg.model.rpn_head.anchor_generator.ratios = [0.25,0.5,1.0,2.0,4.0]

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    
    cfg.optimizer.type = wandb.config.optimizer      # default: Adam
    cfg.optimizer.lr = wandb.config.learning_rate    # default: 1e-3
    
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    cfg.runner.max_epochs = args.epochs # default: 12
    
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 8)')
    parser.add_argument("--resize", nargs="+", type=list, default=[512, 512], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 64)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--sweep', type=bool, default=False, help='using sweep (default: False)')
    parser.add_argument('--anchor_ratio', type=list , default=[0.5,1.0,2.0])

    # Container environment
    parser.add_argument('--work_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './work_dirs'))
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    args = parser.parse_args()
    print(args)

    if args.sweep:
        sweep_configuration = {
            'method': 'bayes',
            'name': 'experiment_name',
            'metric': {'name': "val/bbox_mAP_50", 'goal': 'maximize'},
            'parameters': 
            {
                'learning_rate': {'values' : [0.0001,0.00001,0.000001]},
                'optimizer': {'values': ['Adam','AdamW','SGD']},
                "batch_size": {'values': [2,4,8]},
                "anchor_ratio": {'values': [[0.5,1.0,2.0],[0.25,0.5,1.0,2.0,4.0]]}
            },
            'project': "project",
            'entity' : "entity"
    }
        
    if args.sweep:
        sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        )
        wandb.agent(sweep_id, function=train, count=12)
    else:
        train()
    