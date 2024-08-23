import torch
from super_gradients.training import models
from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train, coco_detection_yolo_format_val)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

import argparse

from generator import _GN_ROOT_PATH, __include_packages
__include_packages()

import utils

def train_yolo_nas(model_arch: str = "yolo_nas_l",
                   batch: int = 8,
                   epochs: int = 25,
                   workers: int = 2,
                   checkpoint_out_dir: str = f"{_GN_ROOT_PATH}/dataset_generator/yolonas_trainings",
                   experiment_name: str = "yolonas_train",
                   dataset_dir: str = f"{_GN_ROOT_PATH}/dataset_generator/gn_sam2segdataset"):

    trainer = Trainer(experiment_name=experiment_name,
                      ckpt_root_dir=checkpoint_out_dir)


    dataset = utils.read_yaml(f"{dataset_dir}/data.yaml")
    classes = dataset["names"]
    
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": dataset_dir,
            "images_dir": "train/images",
            "labels_dir": "train/labels",
            "classes": classes
        },
        dataloader_params={
            "batch_size": batch,
            "num_workers": workers
        }
    )

    val_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": dataset_dir,
            "images_dir": "valid/images",
            "labels_dir": "valid/labels",
            "classes": classes
        },
        dataloader_params={
            "batch_size": batch,
            "num_workers": workers
        }
    )

    test_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": dataset_dir,
            "images_dir": "test/images",
            "labels_dir": "test/labels",
            "classes": classes
        },
        dataloader_params={
            "batch_size": batch,
            "num_workers": workers
        }
    )

    model = models.get(model_arch,
                       num_classes=len(classes),
                       pretrained_weights="coco")

    train_params = {
            "silent_mode": False,
            "average_best_models": True,
            "warmup_mode": "linear_epoch_step",
            "warmup_initial_lr": 1e-6,
            "lr_warmup_epochs": 3,
            "initial_lr": 5e-4,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.1,
            "optimizer": "Adam",
            "optimizer_params": {"weight_decay": 0.0001},
            "zero_weight_decay_on_bias_and_bn": True,
            "ema": True,
            "ema_params": {"decay": 0.9, "decay_type": "threshold"},
            "max_epochs": epochs,
            "mixed_precision": True,
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=len(dataset_params['classes']),
                reg_max=16
                ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=len(dataset_params['classes']),
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7
                        )
                    )
                ],
            "metric_to_watch": 'mAP@0.50'
    }

    trainer.train(model=model,
                  training_params=train_params,
                  train_loader=train_data,
                  valid_loader=val_data)

    # TODO: implement YOLO-NAS model usage in "modelloader.py"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", help="Directory containing datset")

    parser.add_argument("--batch", type=int, default=8, help="Batch size. Default is 8")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs. Default is 25")
    parser.add_argument("--workers", type=int, default=2, help="Workers. Default is 2")

    parser.add_argument("--yolonas-arch", dest="model_arch", type=str, default="yolo_nas_l", help="YOLO-NAS architecture (yolo_nas_{s,m,l}). Default is yolo_nas_l")

    parser.add_argument("-d", "--out-dir", dest="out_dir", type=str, default="yolonas_trainings", help="Directory to save model file")
    parser.add_argument("-n", "--name", type=str, default="yolonas_train", help="Model name to save")
    
    return parser.parse_args()


def main():
    args = parse_args()

    train = train_yolo_nas(model_arch=args.model_arch,
                           batch=args.batch,
                           epochs=args.epochs,
                           workers=args.workers,
                           checkpoint_out_dir=args.out_dir,
                           experiment_name=args.name,
                           dataset_dir=args.dataset_dir)

    print("Finish")



if __name__ == "__main__":
    main()

