import torch
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
import sys

sys.path.append("..")

from utils.utils import *
from utils.print_results2file_with_filename import print_results2file_with_filename
from utils.get_prediction_coordinates import detection_tuning


def detection_accuracy(opt, paths, targets, output, width, height, device, results_filename):
    phone_labels_path = os.path.join(opt.find_phone_folder, "labels.txt")

    df = pd.read_csv(phone_labels_path, sep=" ", names=["id", "x", "y"])

    phone_labels = np.array(df)

    phone_labels_dict = {}

    for i in range(phone_labels.shape[0]):
        phone_labels_dict[phone_labels[i, 0]] = (float(phone_labels[i, 1]), float(phone_labels[i, 2]))

    correct_prediction = 0
    target_count = 0

    for si, pred in enumerate(output):
        image_id = Path(paths[si]).stem.split('_')[-1] + ".jpg"

        target_count += 1

        print_results2file_with_filename(opt, results_filename, "     Input image index: {}".format(si))

        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)

        if pred is None:
            continue
        else:
            if pred.shape[0] == 1:
                pred_coordinates = pred[:, :4]

                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height, width, height]).to(device)

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices

                        pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                        # Search for detections

                        if len(pi):
                            # Prediction to target ious

                            pred_np = pred.cpu().numpy()
                            predicted_coordinates = ((pred_np[pi, 0] + pred_np[pi, 2]) / (2 * width),
                                                     (pred_np[pi, 1] + pred_np[pi, 3]) / (2 * height))

                            print_results2file_with_filename(opt, results_filename,
                                                             "         target    coordinates: {}".format(
                                                                 phone_labels_dict[image_id]))

                            print_results2file_with_filename(opt, results_filename,
                                                             "         predicted coordinates: {} ".format(
                                                                 predicted_coordinates))

                            dist = distance.euclidean(phone_labels_dict[image_id], predicted_coordinates)

                            print_results2file_with_filename(opt, results_filename, "         dist: {}".format(dist))

                            if dist <= 0.05:
                                correct_prediction += 1
            else:
                continue

    detection_acc = correct_prediction / target_count

    print_results2file_with_filename(opt, results_filename, "Detection Accuracy: {}\n".format(detection_acc))

    return detection_acc
