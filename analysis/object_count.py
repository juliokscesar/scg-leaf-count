import supervision as sv
import numpy as np
from typing import List

def count_per_image(imgs: List[str], 
                    detections: List[sv.Detections],
                    x_label: str = "image",
                    y_label: str = "count",
                    save = False,
                    show = True):
    count = np.array([len(det.xyxy) for det in detections])
    img_id = np.arange(1, len(imgs)+1)
    save_to_csv(out_file="count_data.csv", img=img_id, count=count)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(img_id, count, marker='o')

    ax.set(xlabel=x_label,
           ylabel=y_label)

    if save:
        fig.savefig(f"exp_analysis/plots/count_per_image.png")
    if show:
        plt.show()

