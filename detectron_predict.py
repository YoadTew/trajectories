# Some basic setup
# Setup detectron2 logger
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2
import torch
import numpy as np

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

torch.cuda.set_device(0)

cfg = get_cfg()
cfg.merge_from_file("/home/work/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)


def filter_by_classes(outputs, classes=['car', 'bus', 'truck']):
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    vehicle_ids = np.where(np.isin(metadata.thing_classes, classes))[0]

    x = outputs['instances'].pred_classes.cpu()
    filtered_ids = np.where(np.isin(x, vehicle_ids))[0]
    return outputs['instances'][filtered_ids]

def predict(frame):
    outputs = predictor(frame)

    vehicles_outputs = filter_by_classes(outputs, ['car', 'bus', 'truck'])
    people_outputs = filter_by_classes(outputs, ['person'])

    bbox_dict = {
        'vehicle': vehicles_outputs.pred_boxes.tensor.cpu(),
        'person': people_outputs.pred_boxes.tensor.cpu()
    }

    return outputs, bbox_dict

def main():
    cap = cv2.VideoCapture('/home/work/allVid/ALENBI_11_09.10-09.30.avi')
    count = 0

    while(cap.isOpened()):
        count += 1
        print('frame number: ', count)
        ret, frame = cap.read()

        outputs, bbox_dict = predict(frame)

        if count % 10 == 1:
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            final_img = v.get_image()[:, :, ::-1]

            cv2.imwrite('/home/work/output/' + str(count) + '.jpg', final_img)

    print('Done')