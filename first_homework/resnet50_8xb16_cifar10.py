_base_ = [
    '../_base_/models/resnet50_cifar.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]


train_dataloader = dict(
    batch_size=64,
    num_workers=2)

val_dataloader = dict(
    batch_size=64,
    num_workers=2)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))
# learning policy
# param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[5, 20, 50, 100], gamma=0.1)
param_scheduler = dict(_delete_=True,
                    type='CosineAnnealingLR', eta_min=1e-6, by_epoch=True, begin=0)





default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=-1, save_best='auto'),
    visualization=dict(type='VisualizationHook', enable=True, interval=4000)
)

val_evaluator = [
  dict(type='Accuracy', topk=(1, 5)),
  dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score']),
]
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend')  # can cancel wandb for debug
                ]
visualizer = dict(type='ClsVisualizer', vis_backends=vis_backends)

load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth'
resume = False
work_dir = '/root/autodl-tmp/mmcla_workdir/resnet50-1xb16-cifar10'  # debug can use local path