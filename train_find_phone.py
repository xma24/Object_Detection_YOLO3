"""
Train the phone coordinates detection model.
"""

import warnings
warnings.filterwarnings("ignore")

from parameter_setting import parameter_setting
from train_test import train
from expr_setting import expr_setting
from utils.print_results2file_with_filename import print_results2file_with_filename_clear, print_results2file_with_filename_create
from utils.final_results_print import final_results_print
from data_loader import create_find_phone_data


## parameter setting
cuda_index = "1"
expr_index = "yolov3_find_phone_detection"
expr_sub_index = "1"

opt, argv = parameter_setting(cuda_index, expr_index, expr_sub_index)
opt, hyp, last, best, mixed_precision = expr_setting(opt)

## generate yolo v3 labels
create_find_phone_data(opt)

## clean the results of previous running
print_results2file_with_filename_create(opt)
print_results2file_with_filename_clear(opt)

## Training process
final_results_filename = "evaluation_results_expr_" + str(opt.expr_index) + "_sub_expr_" + str(
    opt.expr_sub_index) + ".txt"

acc_folds = []
precision_folds = []
recall_folds = []
f1_folds = []

for fold_index in range(opt.folds):
    yolo3_results, fold_val_acc = train(opt, hyp, mixed_precision, fold_index)

    fold_val_precision = yolo3_results[0]
    fold_val_recall = yolo3_results[1]
    fold_val_f1 = yolo3_results[3]

    acc_folds.append(fold_val_acc)
    precision_folds.append(fold_val_precision)
    recall_folds.append(fold_val_recall)
    f1_folds.append(fold_val_f1)

## print the final results (detection accuracy, precison, recall, F1-score)
final_results_print(opt, final_results_filename, acc_folds, precision_folds, recall_folds, f1_folds)
