"""
This file is used to check the performance of the learned model on the who training dataset.
In the teminal: python checking_acc_train.py PATH_TO_THE_FIND_PHONE_FOLDER

Note: The FIND_PHONE_FOLDER should include the labels.txt file.

"""

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys


def list_filenames(inner_folder_name):
    """
    1. list the files in the folder "inner_folder_name"
    """
    filenames = [f for f in listdir(inner_folder_name) if isfile(join(inner_folder_name, f))]

    return filenames


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    from parameter_setting import parameter_setting
    from expr_setting import expr_setting
    from utils.get_prediction_coordinates import detection_tuning

    cuda_index = "1"
    expr_index = "yolov3_find_phone_detection"
    expr_sub_index = "1"

    opt, argv = parameter_setting(cuda_index, expr_index, expr_sub_index)

    opt, hyp, last, best, mixed_precision = expr_setting(opt)

    ## file names
    # train_data_folder = "./find_phone_images4checking/"
    train_data_folder = sys.argv[1]
    file_list = list_filenames(train_data_folder)

    ## label infomation
    labels_path = os.path.join(train_data_folder, "labels.txt")
    print("labels_path: ", labels_path)
    df = pd.read_csv(labels_path, sep=" ", names=["id", "x", "y"])
    phone_labels = np.array(df)
    phone_labels_dict = {}
    for i in range(phone_labels.shape[0]):
        phone_labels_dict[phone_labels[i, 0]] = (float(phone_labels[i, 1]), float(phone_labels[i, 2]))

    correct = 0.0
    for img_idx in range(len(file_list)):
        file_name = file_list[img_idx]
        end_content = file_name.split(".")[-1]
        if end_content == "txt":
            continue
        else:
            each_image_path = os.path.join(train_data_folder, file_name)
            prediction_coordinates = detection_tuning(opt, each_image_path)
            if prediction_coordinates is None:
                continue
            else:

                target_coordinates = phone_labels_dict[file_name]
                print("image: {}, Pred Coor: {}, Target Coor: {} ".format(file_name, prediction_coordinates,
                                                                      target_coordinates))

                dist_error = distance.euclidean(target_coordinates, prediction_coordinates)

                if dist_error <= 0.05:
                    correct += 1

    detection_acc = correct / (len(file_list) - 1)

    print("Detection accuracy on training dataset: ", detection_acc)
