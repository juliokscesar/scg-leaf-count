import numpy as np

def box_area(box: np.ndarray):
    x1, y1, x2, y2 = box
    return (x2-x1) * (y2-y1)

def box_diagonal(box: np.ndarray):
    x1, y1, x2, y2 = box
    return ( (x2-x1)**2 + +(y2-y1)**2 ) ** 0.5

def mask_pixels(mask: np.ndarray):
    pixels = int(np.sum(mask == 1))
    return pixels


