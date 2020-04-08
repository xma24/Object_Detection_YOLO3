import os
import glob
import numpy as np
import sys

def expr_setting(opt):

    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        mixed_precision = False  # not installed

    # Hyperparameters (results68: 59.9 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310
    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.225,  # iou training threshold
           'lr0': 0.00579,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.5,  # focal loss gamma
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98,  # image rotation (+/- deg)
           'translate': 0.05,  # image translation (+/- fraction)
           'scale': 0.05,  # image scale (+/- gain)
           'shear': 0.641}  # image shear (+/- deg)

    # Overwrite hyp with hyp*.txt (optional)
    f = glob.glob('hyp*.txt')
    if f:
        print('Using %s' % f[0])
        for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
            hyp[k] = v

    wdir = 'weights' + os.sep  # weights dir
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'

    opt.weights = last if opt.resume else opt.weights
    # device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if opt.device.type == 'cpu':
        mixed_precision = False

    return opt, hyp, last, best, mixed_precision