import cv2
import torch
import numpy as np
import mmcv
import PIL.Image
import PIL.ImageDraw

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from tools.condlanenet.common import COLORS
from mmdet.models.detectors.condlanenet import CondLanePostProcessor


def adjust_result(lanes, crop_bbox, img_shape, tgt_shape=(590, 1640)):
    def in_range(pt, img_shape):
        if pt[0] >= 0 and pt[0] < img_shape[1] and pt[1] >= 0 and pt[
                1] <= img_shape[0]:
            return True
        else:
            return False

    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(pt)
            if len(pts) > 1:
                results.append(pts)
    return results


def normalize_coords(coords):
    res = []
    for coord in coords:
        res.append((int(coord[0] + 0.5), int(coord[1] + 0.5)))
    return res


def vis_one(results, img, width=9):
    img_pil = PIL.Image.fromarray(img)

    preds = [normalize_coords(coord) for coord in results]
    for idx, pred_lane in enumerate(preds):
        PIL.ImageDraw.Draw(img_pil).line(xy=pred_lane, fill=COLORS[idx + 1], width=width)

    img = np.array(img_pil, dtype=np.uint8)
    return img


def main(image_file_path, model_file_path, visualize=True):
    # Preprocess test image
    img_raw = cv2.imread(image_file_path)
    img_resize = cv2.resize(img_raw[270:, ...], (800, 320), cv2.INTER_LINEAR)
    mean, std = [75.3, 76.6, 77.6], [50.5, 53.8, 54.3]
    img_resize = (img_resize - mean) / std  # float64
    img_tensor = torch.from_numpy(img_resize).permute(2, 0, 1).float().cuda()
    img_tensor = img_tensor.unsqueeze(0)

    # Load checkpoint
    cfg = mmcv.Config.fromfile('configs/condlanenet/culane/culane_small_test.py')
    model = build_detector(cfg.model)
    load_checkpoint(model, model_file_path, map_location='cuda:0')
    model.eval()

    # Inference
    model.cuda()
    seeds, _ = model(img_tensor)

    # Postprocessing
    post_processor = CondLanePostProcessor(
        mask_size=(1, 40, 100), hm_thr=0.5, use_offset=True, nms_thr=4)
    lanes, seeds = post_processor(seeds, 8)
    if visualize:
        raw_size = (590, 1640)
        crop_bbox = [0, 270, 1640, 590]
        result = adjust_result(
            lanes=lanes, crop_bbox=crop_bbox, img_shape=(320, 800, 3), tgt_shape=raw_size)
        img_vis = vis_one(result, img_raw)
        cv2.imshow('condlanenet_inference', img_vis)
        cv2.waitKey(0)


if __name__ == '__main__':
    image_file = 'test.jpg'
    model_file = 'culane_small.pth'
    main(image_file, model_file)
