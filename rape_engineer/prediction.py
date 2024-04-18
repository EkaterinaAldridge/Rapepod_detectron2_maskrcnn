# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


#Register a Dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("rape_train", {}, "train_rape.json", "rape_img/train")
register_coco_instances("rape_val", {}, "val_rape.json", "rape_img/val")

from detectron2.engine import DefaultTrainer


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("rape_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 100
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
#cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)



from detectron2.utils.visualizer import ColorMode    

#lw_ratio = np.zeros(outputs["instances"].pred_masks.shape[0])

for filename in os.listdir("./rape_img/test/"):

    #import pdb; pdb.set_trace()
    
    if filename[-3:] != "jpg":
        continue 
    
    im = cv2.imread("./rape_img/test/" + filename)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    
    
    v = Visualizer(im[:, :, ::-1],
                   metadata={'thing_classes': ['rapepod']},
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )

    for i in range(outputs["instances"].pred_masks.shape[0]):
        mask = outputs["instances"].pred_masks.to('cpu').numpy()[i].astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print("当前是第几个：", i)

        rect = None
        c = None
        if len(contours) == 1:
            c = contours[0]
            rect = cv2.minAreaRect(c)
        else:
            temp_area = 0
            temp_ract = None
            temp_c = None
            for c in contours:
                if cv2.contourArea(c) > temp_area:
                    temp_ract = cv2.minAreaRect(c)
                    temp_c = c
            rect = temp_ract
            c = temp_c

        length = cv2.arcLength(c, True)/2
        area = cv2.contourArea(c)
        # print("ract", rect)
        # print("length", length)
        # print("area", area)

        if length == 0:
            continue
        else:
            if area == 0:
                continue
            else:
                width = area / length
                lw_ratio = length / width
                area_ratio = area / length
                # v.draw_text("ratio %.2f" % lw_ratio, rect[0])
                #v.draw_text(" %1f " % i, rect[0])
                #v.draw_text(" %d ratio1: %.2f\n   ratio2: %.2f" % (i+1, lw_ratio, area_ratio), rect[0])
                # print("计算所得长宽比:", lw_ratio)
                # print("计算所得面积和长比：", area_ratio)



    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    cv2.imwrite("./results/"+filename,out.get_image()[:, :, ::-1])
    
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.savefig
    #plt.show()
        
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("rape_val", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "rape_val")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# # evaluator = COCOEvaluator("rape_val", ("bbox", "segm"), False, output_dir="./output")
# evaluator = COCOEvaluator("rape_val", False, output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "rape_val")
# # print(inference_on_dataset(predictor.model, val_loader, evaluator))
# inference_on_dataset(predictor.model, val_loader, evaluator)
# # another equivalent way to evaluate the model is to use `trainer.test`
