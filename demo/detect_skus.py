from pathlib import Path
from tqdm import tqdm
import matplotlib

matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
from collections import namedtuple
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import fire


LabelRow = namedtuple("LabelRow", "im_path sku_index xmin ymin xmax ymax")


def flatten_list_of_list(l):
    return [c for row in l for c in row]


def dbscan(y, esp=200):
    Y = np.vstack((np.zeros_like(y), y)).transpose()
    db = DBSCAN(eps=esp, min_samples=1).fit(Y)
    return db.labels_


def sort_bboxes(bboxes, im_size, empirical_divisor=10):
    center_x = (bboxes[:, 2] + bboxes[:, 0]) // 2
    ymax = bboxes[:, 3]
    clust_labels = dbscan(ymax, esp=im_size[1] / empirical_divisor)
    order = []
    for row_id in range(clust_labels.max() + 1):
        idx = (clust_labels == row_id).nonzero()[0]
        inner_row_order = np.argsort(center_x[idx])  # order x positions within a row
        order.append(idx[inner_row_order].tolist())
    order = sorted(order, key=lambda _idx: ymax[_idx].mean())  # order rows according to their ymax.mean()
    order = flatten_list_of_list(order)
    return bboxes[order]


def to_pct_repr(bbox, im_size):
    return [c / sz for c, sz in zip(bbox, im_size * 2)]


def crop(
    src_img_dir,
    result_save_dir,
    detector_conf,
    detector_model_path,
    conf_th=0.3,
    save_to_sub_folders=False,
    append_img_name=False,
):
    # build the model from a config file and a checkpoint file
    model = init_detector(detector_conf, detector_model_path, device="cuda:0")

    src_img_dir = Path(src_img_dir)
    result_save_dir = Path(result_save_dir)
    result_save_dir.mkdir(parents=True, exist_ok=True)

    label_data = []
    for i, im_path in enumerate(sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.JPG")))):
        result = inference_detector(model, str(im_path))

        # show the results
        # show_result_pyplot(str(im_path), result, model.CLASSES)
        # plt.show()

        im = cv2.imread(str(im_path))
        im_size = im.shape[:2][::-1]
        bboxes = result[0][result[0][:, 4] >= conf_th, :4]
        bboxes = sort_bboxes(bboxes, im_size)

        for j, bbox in tqdm(enumerate(bboxes)):
            _bx = [int(round(c)) for c in bbox]
            fname = f"{j:0>2}#{im_path.stem}.jpg" if append_img_name else f"{j:0>2}_{i + 1}.jpg"
            if save_to_sub_folders:
                fname = Path(im_path.stem) / fname
            save_path = result_save_dir / fname
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), im[_bx[1] : _bx[3], _bx[0] : _bx[2], :])
            label_data.append(LabelRow(str(im_path), str(fname), *to_pct_repr(bbox, im_size)))

    pd.DataFrame(label_data).to_csv(result_save_dir / "detection-info.csv", index=False)


if __name__ == "__main__":
    fire.Fire(crop)
