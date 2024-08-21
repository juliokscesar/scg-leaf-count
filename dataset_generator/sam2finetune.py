import utils
from pathlib import Path

# Data should have images and annotations separated in different
# folders. Every image (like 'a.png') has an annotations TXT file
# with the same name (like 'a.txt')
def get_annotation_data(imgs_dir: str, annotations_dir: str):
    img_files = utils.get_all_files_from_paths(imgs_dir)
    ann_files = utils.get_all_files_from_paths(annotations_dir)

    data = []
    for img, ann in zip(img_files, ann_files):
        print(img, ann)
        data.append({"image": img, "annotation": ann})

    return data

