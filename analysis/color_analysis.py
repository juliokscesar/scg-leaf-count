from typing import List, Union, Tuple
import cv2
import numpy as np
import warnings

from scg_detection_tools.utils.cvt import contours_to_masks, boxes_to_masks

def calc_img_hist(img: np.ndarray, channels: List[int], mask: np.ndarray = None):
    hists = []
    for c in channels:
        h = cv2.calcHist([img], [c], mask, [256], [0,256])
        hists.append(h.reshape(256))
    return hists

def color_hist(img: Union[np.ndarray, str],
               imgcspace: str = "RGB",
               cspaces: List[str] = ["RGB"],
               masks: np.ndarray = None,
               masks_contours: np.ndarray = None,
               mask_classes: List[int] = None,
               boxes: np.ndarray = None):
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgcspace = "RGB"
    
    if masks and masks_contours:
        warnings.warn("func color_hist: passed masks and masks_contours, but only masks will be used")
        masks_contours = None
    if masks and boxes:
        warnings.warn("func color_hist: passed masks and boxes, but only masks will be used")
        boxes = None
    if masks_contours and boxes:
        warnings.warn("func color_hist: passed masks_contours and boxes, but only masks_contours will be used")
        boxes = None


    if masks is None:
        if masks_contours is not None:
            masks = contours_to_masks(masks_contours, imgsz=img.shape[:2])
        elif boxes is not None:
            masks = boxes_to_masks(boxes, imgsz=img.shape[:2])

    hists = {}
    for cspace in cspaces:
        cspace = cspace.strip().upper()

        hists[cspace] = { "full": None, "masks": None }

        if cspace == "RGB" or cspace == "HSV":
            if imgcspace == "GRAY":
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                imgcspace = "RGB"

            channels = [0,1,2]
            if cspace == "HSV" and imgcspace != "HSV":
                imgcspace = "HSV"
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == "RGB" and imgcspace != "RGB":
                imgcspace = "RGB"
                if cspace == "BGR":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif cspace == "HSV":
                    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        elif cspace == "GRAY":
            channels = [0]
            if img.ndim != 2:
                if imgcspace == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif imgcspace == "HSV":
                    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif imgcspace == "BGR":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                imgcspace = "GRAY"

        else:
            raise ValueError(f"func color_hist: color space {cspace} not supported")

        if masks is not None:
            if mask_classes is None:
                mask_classes = np.zeros(len(masks))
            elif len(mask_classes) != len(masks):
                warnings.warn("func color_hist: len(mask_classes) != len(masks). Filling rest with 0")
                mask_classes = np.resize(np.array(mask_classes), len(masks))

            hists[cspace]["masks"] = []
            masks_hist = np.zeros(shape=(len(channels),256))
            for mask, nc in zip(masks, mask_classes):
                ch_hist = calc_img_hist(img, channels, mask)
                hists[cspace]["masks"].append({"class": nc, "hist": ch_hist})
                masks_hist += ch_hist
            hists[cspace]["full"] = masks_hist
        else:
            hists[cspace]["full"] = calc_img_hist(img, channels, None)

    return hists


# TODO: figure out a way to save each mask histogram and their correspondent class
def calc_masks_hist(img: np.ndarray, masks: np.ndarray, channels: List[int]):
    raise NotImplemented()
    MAX_THREADS = 500
    workers = [None] * MAX_THREADS
    
    def _conc_call(q, *args):
        res = calc_img_hist(*args)
        q.put(res)

    que = queue.Queue()
    mask_per_worker = (len(masks) // MAX_THREADS) + 1
    num_workers = len(masks) // mask_per_worker
    for i in range(num_workers):
        worker = threading.Thread(target=_conc_call, args=(que, img, channels, masks[i*mask_per_worker:
                                                                                (i+1)*mask_per_worker] ))
        worker.start()

    for worker in workers:
        if worker is None:
            break
        worker.join()

    masks_hists = np.zeros(shape=(len(channels), 256))
    # After all threads finished
    while not que.empty():
        masks_hists += np.array(que.get())

    return masks_hists
