import argparse
import torch

def parameter_setting(cuda_index, expr_index, expr_sub_index):
    parser = argparse.ArgumentParser()

    ## experiment info
    parser.add_argument('--cuda_index', default=cuda_index)
    parser.add_argument('--expr_index', default=expr_index)
    parser.add_argument('--expr_sub_index', default=expr_sub_index)
    parser.add_argument('--dataset_name', default="find_phone")
    parser.add_argument('--cfg', type=str, default='../cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416], help='train and test image-sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--single-cls', default=True, action='store_true', help='train as single-class dataset')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--folds', default=3)
    parser.add_argument('--img_width', default=490, help='train and test image-sizes width')
    parser.add_argument('--img_height', default=326, help='train and test image-sizes height')
    parser.add_argument('--conf-thres', type=float, default=0.004, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    ## training info
    parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    ## project root folder
    parser.add_argument('--root_folder', default="./")

    ## input folder
    parser.add_argument('--data_folder', default="./find_phone/")
    parser.add_argument('--find_phone_folder',
                        default="./find_phone/")
    parser.add_argument('--find_phone_vott_labels',
                        default="./labels_phone/vott-csv-export/find_phone-export.csv")
    parser.add_argument('--labels_phone',
                        default="./labels_phone/")
    parser.add_argument('--weights', type=str,
                        default='./init_weights/ultralytics68.pt',
                        help='initial weights')

    ## output folder
    parser.add_argument('--results_folder', default="./results/")
    parser.add_argument('--w_folder', default="./results/w_folder/")
    parser.add_argument('--inference_modified_w_folder', default="./results/w_folder/")
    parser.add_argument("--log_folder",
                        default="./results/log_files/")

    ## gpu info
    parser.add_argument('--cuda', default=True)
    # parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    # args = parser.parse_args()
    args, argv = parser.parse_known_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    args.device = torch.device("cuda:" + cuda_index if use_cuda else "cpu")
    args.kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    args.weights = args.w_folder + 'last.pt' if args.resume else args.weights

    return args, argv
