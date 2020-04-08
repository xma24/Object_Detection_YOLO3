import os
from os import listdir
from os.path import isfile, join


def list_filenames(inner_folder_name):
    """
    1. list the files in the folder "inner_folder_name"
    """
    filenames = [f for f in listdir(inner_folder_name) if isfile(join(inner_folder_name, f))]

    return filenames


def print_results2file_with_filename(args, filename, data2print):
    os.makedirs(args.log_folder, exist_ok=True)
    log_file_path = os.path.join(args.log_folder, args.expr_index, args.expr_sub_index, filename)

    with open(log_file_path, "a+") as text_file:
        text_file.write('{}\n'.format(data2print))


def print_results2file_with_filename_clear(args):
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(os.path.join(args.log_folder, args.expr_index), exist_ok=True)
    os.makedirs(os.path.join(args.log_folder, args.expr_index, args.expr_sub_index), exist_ok=True)
    filename_list = list_filenames(os.path.join(args.log_folder, args.expr_index, args.expr_sub_index))
    if len(filename_list):
        for i in range(len(filename_list)):
            log_file_path = os.path.join(args.log_folder, args.expr_index, args.expr_sub_index, filename_list[i])
            with open(log_file_path, "w") as text_file:
                print(" ", file=text_file)


def print_results2file_with_filename_create(args):
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(os.path.join(args.log_folder, args.expr_index), exist_ok=True)
    os.makedirs(os.path.join(args.log_folder, args.expr_index, args.expr_sub_index), exist_ok=True)
