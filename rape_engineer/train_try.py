import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import random
from detectron2.engine import DefaultPredictor


#Register a Dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

register_coco_instances("rape_train", {}, "train_rape.json", "./rape_img/train")
register_coco_instances("rape_val", {}, "val_rape.json", "./rape_img/val")

#查看一下自己的元数据
##运行后可以看到thing_classes=["自己的数据类别"]
coco_val_metadata = MetadataCatalog.get("rape_val")
dataset_dicts = DatasetCatalog.get("rape_val")
print(coco_val_metadata)


# for d in random.sample(dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=coco_val_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     plt.imshow(vis.get_image()[:, :, ::-1])
#     plt.show()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("rape_train",)
cfg.DATASETS.TEST = ("rape_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#cfg.MODEL.DEVICE = 'cpu'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)    #传入output文件夹 存放训练好的权重等
trainer = DefaultTrainer(cfg)
#DefaultTrainer.test(evaluators='coco')
trainer.resume_or_load(resume=False)
trainer.train()


import os
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # 我们进行训练的 权值文件  存放处
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # 设置一个阈值
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(dataset_dicts, 1):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im,
                   metadata=coco_val_metadata,
                   scale=0.9,
                   instance_mode=ColorMode.IMAGE
                   )
    # remove the colors of unsegmented pixels
    print(outputs['instances'].pred_classes)
    print(outputs["instances"].pred_boxes)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()

