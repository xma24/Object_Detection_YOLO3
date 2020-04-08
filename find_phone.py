"""
This file is for inference.
In terminal: python find_phone.py PATH_TO_IMAGE
"""

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from utils.datasets import *
from parameter_setting import parameter_setting
from expr_setting import expr_setting
from utils.get_prediction_coordinates import detection_tuning

cuda_index = "1"
expr_index = "yolov3_find_phone_detection"
expr_sub_index = "1"

opt, argv = parameter_setting(cuda_index, expr_index, expr_sub_index)

opt, hyp, last, best, mixed_precision = expr_setting(opt)

print("sys.argv: ", sys.argv)

predicted_coordinates = detection_tuning(opt, sys.argv[1])
print("{0:.4f} {1:.4f}".format(predicted_coordinates[0, 0], predicted_coordinates[0, 1]))
