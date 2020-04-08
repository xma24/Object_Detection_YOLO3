import warnings
warnings.filterwarnings("ignore")

from models.yolo3 import *
from utils.utils import *
from utils.datasets import *



def get_prediction_coordinates(opt, image_path, conf_thres):
    predicted_coordinates = np.ones((1, 2))

    model = Darknet(opt.cfg, opt.img_size)
    model.load_state_dict(torch.load(opt.inference_modified_w_folder + "best.pt", map_location=opt.device)['model'])

    model.to(opt.device)

    image2detect = torch.from_numpy(np.expand_dims(cv2.imread(image_path), axis=0)).to(opt.device).float() / 255.0


    image2detect = image2detect.permute(0, 3, 1, 2)
    dataset = LoadImages(image_path)

    _, _, height, width = image2detect.shape

    model.eval()

    with torch.no_grad():
        for path, img, im0s, vid_cap in dataset:

            # Get detections
            img = torch.from_numpy(img).to(opt.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Run model
            pred = model(img)[0]  # inference and training outputs

            # Run NMS
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.5, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                # s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if det is not None and len(det):
                    for i in range(det.shape[0]):
                        pred_np = det.cpu().numpy()

                        each_coordinates = np.array([(pred_np[i, 0] + pred_np[i, 2]) / (2 * width),
                                                     (pred_np[i, 1] + pred_np[i, 3]) / (2 * height)]).reshape((1, 2))
                        predicted_coordinates = np.append(predicted_coordinates, each_coordinates, axis=0)

                else:
                    return None
    return predicted_coordinates[1:]


def detection_tuning(opt, inner_image_path):
    conf_thres = 0.1

    for i in range(100):
        coordinates_ret = get_prediction_coordinates(opt, inner_image_path, conf_thres)
        if coordinates_ret is None:
            return None
        else:
            if coordinates_ret.shape[0] == 1:
                return coordinates_ret
            elif coordinates_ret.shape[0] > 1:
                conf_thres += 0.02
                coordinates_ret = get_prediction_coordinates(opt, inner_image_path, conf_thres)
                if coordinates_ret is None:
                    return None
            else:
                conf_thres -= 0.01
                coordinates_ret = get_prediction_coordinates(opt, inner_image_path, conf_thres)
                if coordinates_ret is None:
                    return None

    return coordinates_ret[0].reshape((1, 2))

