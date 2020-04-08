import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold


def convert(width, height, box):
    dw = 1. / width
    dh = 1. / height
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return np.array([x, y, w, h])


def label_content(opt, raw_labels, train_index, val_index, fold_idx):
    ## train and val label folder
    train_labels_path = os.path.join(opt.labels_phone, "train_labels_fold_" + str(fold_idx))
    os.makedirs(train_labels_path, exist_ok=True)

    val_labels_path = os.path.join(opt.labels_phone, "val_labels_fold_" + str(fold_idx))
    os.makedirs(val_labels_path, exist_ok=True)

    for train_img_idx in train_index:
        train_img_name = raw_labels[train_img_idx][0]
        train_img_label = int(0)

        train_box_coordinates_init = raw_labels[train_img_idx][1:-1]
        train_box_coordinates = convert(opt.img_width, opt.img_height, train_box_coordinates_init)

        train_yolo_format_label = np.append(train_img_label, train_box_coordinates).reshape((1, -1))
        np.savetxt(os.path.join(train_labels_path, "find_phone_train_" + train_img_name.split(".")[0] + ".txt"),
                   train_yolo_format_label)

    for val_img_idx in val_index:
        val_img_name = raw_labels[val_img_idx][0]
        val_img_label = int(0)

        val_box_coordinates_init = raw_labels[val_img_idx][1:-1]
        val_box_coordinates = convert(opt.img_width, opt.img_height, val_box_coordinates_init)

        val_yolo_format_label = np.append(val_img_label, val_box_coordinates).reshape((1, -1))
        np.savetxt(os.path.join(val_labels_path, "find_phone_val_" + val_img_name.split(".")[0] + ".txt"),
                   val_yolo_format_label)


def label_names(opt, raw_labels, train_index, val_index, fold_idx):
    train_labels_names = np.array(raw_labels[train_index][0][-1]).reshape(-1, )
    np.savetxt(os.path.join(opt.labels_phone, "train_labels_names_fold_" + str(fold_idx) + ".names"),
               train_labels_names, fmt="%s")
    val_labels_names = np.array(raw_labels[val_index][0][-1]).reshape(-1, )
    np.savetxt(os.path.join(opt.labels_phone, "val_labels_names_fold_" + str(fold_idx) + ".names"),
               val_labels_names, fmt="%s")


def label_config(opt, fold_idx):
    ## train and val dataset label name
    train_path_config = []
    train_path_config.append(["classes=1"])
    train_path_config.append(
        ["train=" + os.path.join(opt.labels_phone, "train_labels_config_fold_" + str(fold_idx) + ".txt")])
    train_path_config.append(
        ["valid=" + os.path.join(opt.labels_phone, "val_labels_config_fold_" + str(fold_idx) + ".txt")])
    train_path_config.append(
        ["names=" + os.path.join(opt.labels_phone, "train_labels_names_fold_" + str(fold_idx) + ".names")])

    train_path_config_np = np.array(train_path_config).reshape(-1, 1)
    np.savetxt(os.path.join(opt.labels_phone, "train_labels_config_fold_" + str(fold_idx) + ".data"),
               train_path_config_np, fmt="%s")


def image_path_file(opt, raw_labels, train_index, val_index, fold_idx):
    ## train and val label folder
    train_labels_path = os.path.join(opt.labels_phone, "train_labels_fold_" + str(fold_idx))
    os.makedirs(train_labels_path, exist_ok=True)

    val_labels_path = os.path.join(opt.labels_phone, "val_labels_fold_" + str(fold_idx))
    os.makedirs(val_labels_path, exist_ok=True)

    ## generate train labels txt file
    train_labels_names = raw_labels[train_index][:, 0]
    train_labels_config = [os.path.join(opt.labels_phone, "vott-csv-export", each_train_label) for each_train_label
                           in
                           train_labels_names]
    train_labels_config_np = np.array(train_labels_config).reshape(-1, 1)
    np.savetxt(os.path.join(opt.labels_phone, "train_labels_config_fold_" + str(fold_idx) + ".txt"),
               train_labels_config_np, fmt="%s")

    ## generate val labels txt file
    val_labels_names = raw_labels[val_index][:, 0]
    val_labels_config = [os.path.join(opt.labels_phone, "vott-csv-export", each_val_label) for each_val_label
                         in
                         val_labels_names]
    val_labels_config_np = np.array(val_labels_config).reshape(-1, 1)
    np.savetxt(os.path.join(opt.labels_phone, "val_labels_config_fold_" + str(fold_idx) + ".txt"),
               val_labels_config_np, fmt="%s")


def full_labels_file(opt, raw_labels, train_index, val_index):
    for train_img_idx in train_index:
        train_img_name = raw_labels[train_img_idx][0]
        train_img_label = int(0)

        train_box_coordinates_init = raw_labels[train_img_idx][1:-1]
        train_box_coordinates = convert(opt.img_width, opt.img_height, train_box_coordinates_init)

        train_yolo_format_label = np.append(train_img_label, train_box_coordinates).reshape((1, -1))
        np.savetxt(os.path.join(opt.labels_phone, "vott-csv-export", train_img_name.split(".")[0] + ".txt"),
                   train_yolo_format_label)

    for val_img_idx in val_index:
        val_img_name = raw_labels[val_img_idx][0]
        val_img_label = int(0)

        val_box_coordinates_init = raw_labels[val_img_idx][1:-1]
        val_box_coordinates = convert(opt.img_width, opt.img_height, val_box_coordinates_init)

        val_yolo_format_label = np.append(val_img_label, val_box_coordinates).reshape((1, -1))
        np.savetxt(os.path.join(opt.labels_phone, "vott-csv-export", val_img_name.split(".")[0] + ".txt"),
                   val_yolo_format_label)


def create_find_phone_data(opt):
    inner_df = pd.read_csv(opt.find_phone_vott_labels)
    inner_labels = np.array(inner_df)

    kf = KFold(n_splits=opt.folds)

    for fold_idx in range(opt.folds):
        for train_index, val_index in kf.split(inner_labels):
            label_content(opt, inner_labels, train_index, val_index, fold_idx)
            label_names(opt, inner_labels, train_index, val_index, fold_idx)
            label_config(opt, fold_idx)
            image_path_file(opt, inner_labels, train_index, val_index, fold_idx)
            full_labels_file(opt, inner_labels, train_index, val_index)
