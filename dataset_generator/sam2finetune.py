import numpy as np
import cv2
from pathlib import Path
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import argparse


from generator import _GN_ROOT_PATH, __include_packages
__include_packages()

import imagetools
import utils
import generator as gn


# Data should have images and annotations separated in different
# folders. Every image (like 'a.png') has an annotations TXT file
# with the same name (like 'a.txt')
def get_annotation_data(imgs_dir: str, annotations_dir: str):
    img_files = utils.get_all_files_from_paths(imgs_dir)
    ann_files = utils.get_all_files_from_paths(annotations_dir)

    data = []
    for img, ann in zip(img_files, ann_files):
        print(img)
        masks = yolo_annotation_to_sam2(ann_file=ann)

        imagetools.plot_image(masks)

        data.append({"image": img, "annotation": masks})

    return data


def yolo_annotation_to_sam2(ann_file: str, img_sz=(640,640)):
    contour_img = np.zeros(img_sz, dtype=np.uint8)

    with open(ann_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            
            points = list(map(float, data[1:]))

            # reshape list into pairs of (x,y) coordinates
            points = np.array(points).reshape(-1, 2)

            points = np.int32(points * np.array(img_sz[::-1]))
            cv2.fillPoly(contour_img, [points], color=255)

    return contour_img


def read_batch(data):
    # select image
    ent = data[np.random.randint(len(data))]
    img = cv2.imread(ent["image"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ann_map = cv2.imread(ent["annotation"])

    if img.shape[0] > 1024 or img.shape[1] > 1024:
        r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        img = cv2.resize(img, (img.shape[::-1]*r).astype(np.int32))

    
    inds = np.unique(ann_map)[1:]
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([ yx[::-1] ])

    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])


def finetune_sam2(imgs_dir: str,
                  annotations_dir: str,
                  sam2_checkpoint: str = f"{_GN_ROOT_PATH}/dataset_generator/sam2chkpts/sam2_hiera_small.pt",
                  model_cfg: str = "sam2_hiera_s.yaml"):

    if not torch.cuda.is_available():
        raise Exception("Fine tuning SAM2 is currentyl available only with a CUDA device")

    device = "cuda"
    s = torch.cuda.amp.GradScaler()
    torch.cuda.amp.autocast().__enter__()

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    optimizer = torch.optim.AdamW(params=predictor.model_parameters(),
                                  lr=1e-5,
                                  weight_decay=4e-5)


    data = get_annotation_data(imgs_dir, annotations_dir)

    ITERATIONS = 100_000
    mean_iou = 0
    for _ in range(ITERATIONS):
        img, masks, input_point, input_label = read_batch(data)
        if mask.shape[0] == 0: continue # ignore empty batches
        predictor.set_image(img)

    
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point,
                                                                                input_label,
                                                                                box=None,
                                                                                mask_logits=None,
                                                                                normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                 boxes=None,
                                                                                 masks=None)


        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                                                                           mage_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                                           dense_prompt_embeddings=dense_embeddings,
                                                                           multimask_output=False,
                                                                           repeat_image=batched_mode,
                                                                           high_res_features=high_res_features)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        prd_masks = torch.sigmoid(prd_masks[:,0])
        gt_mask = torch.tensor(masks.astype(np.float32)).cuda()

        # segmentation loss
        seg_loss = (-gt_mask * torch.log(prd_masks + 0.00001) - (1-gt_mask) * torch.log((1-prd_masks) + 0.00001)).mean()

        # score loss
        inter = (gt_mask * (prd_masks>0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_masks>0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:,0] - iou).mean()

        loss = seg_loss+score_loss*0.5

        # backpropagation
        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if itr%1000==0: torch.save(predictor.model.state_dict(), "last_statedict.torch")
        
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().deatch().numpy())
        print(f"Step: {step} | Accuracy(IOU): {mean_iou}") 


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("images_dir", help="Path to directory containing images to use.")
    parser.add_argument("annotations_dir", help="Path to directory containing the segmentation labels in .TXT files with same name as corresponding image.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    finetune_sam2(imgs_dir=args.images_dir, 
                  annotations_dir=args.annotations_dir)

    print("Finish")
    

if __name__ == "__main__":
    main()

