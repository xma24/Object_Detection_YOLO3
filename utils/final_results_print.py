import numpy as np
import sys
sys.path.append("..")

from utils.print_results2file_with_filename import print_results2file_with_filename


def final_results_print(opt, final_results_file, acc_folds, precision_folds, recall_folds, f1_folds):
    acc_str = '{}-fold CV Accuracy: {} ; avg acc (+- std): {} ({}) \n'.format(opt.folds,
                                                                     np.array(acc_folds),
                                                                     np.mean(acc_folds),
                                                                     np.std(acc_folds))

    precision_str = '{}-fold CV Precision: {} ; avg precision (+- std): {} ({}) \n'.format(opt.folds,np.array(precision_folds),
                                                                                 np.mean(precision_folds),
                                                                                 np.std(precision_folds))

    recall_str = '{}-fold CV Recall: {} ; avg recall (+- std): {} ({}) \n'.format(opt.folds,
                                                                           np.array(recall_folds),
                                                                           np.mean(recall_folds),
                                                                           np.std(recall_folds))

    f1_str = '{}-fold CV F1-score: {} ; avg f1 (+- std): {} ({}) \n'.format(opt.folds,
                                                                   np.array(f1_folds),
                                                                   np.mean(f1_folds),
                                                                   np.std(f1_folds))

    print_results2file_with_filename(opt, final_results_file, acc_str)
    print_results2file_with_filename(opt, final_results_file, precision_str)
    print_results2file_with_filename(opt, final_results_file, recall_str)
    print_results2file_with_filename(opt, final_results_file, f1_str)

    print(acc_str)
    print(precision_str)
    print(recall_str)
    print(f1_str)
