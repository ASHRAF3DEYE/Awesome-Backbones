# model settings

model_cfg = dict(
    backbone=dict(type='EfficientNet', arch='b7'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=9,
        in_channels=2560,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        ))

# dataloader pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=600,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomGrayscale'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=600,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# train
data_cfg = dict(
    batch_size = 2,
    num_workers = 4,
    train = dict(
        pretrained_flag = False,
        pretrained_weights = '',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = '',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# batch 32
# lr = 0.1 *4 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.1 * 2/256,
    momentum=0.9,
    weight_decay=1e-4)

# learning 
lr_config = dict(
    type='StepLrUpdater',
    step=[30, 60, 90]
)

