import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

from scg_detection_tools.models import BaseDetectionModel
from scg_detection_tools.detect import Detector

def count_per_image(imgs: List[str], 
                    detections: List[sv.Detections],
                    x_label: str = "image",
                    y_label: str = "count",
                    save = False,
                    show = True):
    count = np.array([len(det.xyxy) for det in detections])
    img_id = np.arange(1, len(imgs)+1)

    if save or show:
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(img_id, count, marker='o')

        ax.set(xlabel=x_label,
            ylabel=y_label)

        if save:
            fig.savefig(f"exp_analysis/plots/count_per_image.png")
        if show:
            plt.show()

    return count


def model_count_valid_metric(model: BaseDetectionModel,
                            imgs: List[str],
                            true_counts: Union[List[int], np.ndarray],
                            confidence=60.0):
    """ 
    Validade model based on counting objects from an image and comparing with true count.

    Returns the MAE, MSE, RMSE and Error STD from the detections of every image.
    """

    detector = Detector(model)
    detections = detector.detect_objects(imgs, confidence=confidence)
    pred_count = count_per_image(imgs, detections, show=False, save=False)

    if not isinstance(true_counts, np.ndarray):
        true_counts = np.array(true_counts)

    if len(true_counts) != len(pred_count):
        raise RuntimeError(f"'true_counts' (shape {true_counts.shape}) and 'pred_counts' (shape {pred_count.shape}) must have same length")

    errors = true_counts - pred_count
    relative = np.divide(
        errors.astype(np.float64), 
        true_counts.astype(np.float64), 
        out=np.zeros_like(errors.astype(np.float64)), 
        where=true_counts != 0
    )
    
    mae = np.absolute(errors).mean()
    mse = (errors**2).mean()
    rmse = np.sqrt(mse)
    stderror = np.sqrt( ((errors - errors.mean())**2).mean() )

    rel_mae = np.absolute(relative).mean()
    rel_mse = (relative**2).mean()
    rel_rmse = np.sqrt(rel_mse)
    rel_stderror = np.sqrt( ((relative - relative.mean())**2).mean() )

    return { 
        "mae": mae, 
        "mse": mse, 
        "rmse": rmse, 
        "stderror": stderror,

        "relative_mae": rel_mae,
        "relative_mse": rel_mse,
        "relative_rmse": rel_rmse,
        "relative_stderror": rel_stderror,
    }
