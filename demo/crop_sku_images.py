from pathlib import Path
from tqdm import tqdm
import matplotlib; matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN


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


config_file = '../configs/faster_rcnn_hrnetv2_w48_2x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/faster_rcnn_hrnetv2p_w48_2x_20190820-79fb8bfc.pth'
checkpoint_file = '/home/rlan/deploy/mmdetection/work_dirs/faster_rcnn_hrnetv2p_w48_2x/epoch_3.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

conf_th = 0.3
sku_source_img_dir = Path("/tmp/sku-source")
sku_img_save_dir = Path("/data2/datasets/clobotics/chinadrink/labgen/batch1/sku")
sku_img_save_dir.mkdir(parents=True, exist_ok=True)

for i, im_path in enumerate(sorted(sku_source_img_dir.glob("*.jpg"))):
    result = inference_detector(model, str(im_path))

    # show the results
    # show_result_pyplot(str(im_path), result, model.CLASSES)
    # plt.show()

    im = cv2.imread(str(im_path))
    bboxes = result[0][result[0][:, 4] >= conf_th, :4]
    bboxes = sort_bboxes(bboxes, im.shape[:2][::-1])
    for j, bbox in tqdm(enumerate(bboxes)):
        _bx = [int(round(c)) for c in bbox]
        cv2.imwrite(str(sku_img_save_dir / f"{j:0>2}_{i+1}.jpg"), im[_bx[1]:_bx[3], _bx[0]:_bx[2], :])
