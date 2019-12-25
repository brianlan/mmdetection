from pathlib import Path
from tqdm import tqdm
import matplotlib; matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
from mmdet.apis.inference import inference_template_matcher
import cv2


config_file = '../configs/faster_rcnn_qatm_hrnetv2_w48_2x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/faster_rcnn_hrnetv2p_w48_2x_20190820-79fb8bfc.pth'
checkpoint_file = '/home/rlan/deploy/mmdetection/work_dirs/faster_rcnn_hrnetv2p_w48_2x/epoch_3.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
# img = '/tmp/rare-5sku/1037979/IMG_7790.JPG'
# template = '/tmp/rare-5sku/1037979.jpg'
# result = inference_template_matcher(model, img, template)
#
# # show the results
# show_result_pyplot(img, result, model.CLASSES)
# plt.show()

# Inference Multiple Images
# save_dir = Path("/tmp/vis/onboard-demo")
# img_dir = Path("/tmp/onboard-demo")
# template_fname = "target_sku.jpg"
# img_fnames = ["00.jpg", "01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg",
#              "10.jpg", "11.jpg", "12.jpg"]
# for img_fn in tqdm(img_fnames):
#     img_path = img_dir / img_fn
#     template_path = img_dir / template_fname
#     result = inference_template_matcher(model, str(img_path), str(template_path))
#     _im = show_result(str(img_path), result, model.CLASSES, show=False)
#     save_path = save_dir / img_fn
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(save_path), _im)

# Inference Multiple Images
# save_dir = Path("/tmp/vis/onboard-demo/t0")
# img_dir = Path("/tmp/onboard-demo/search")
# template_path = "/tmp/onboard-demo/template/t0.jpg"
# paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG"))
# paths = [p for p in paths if not p.name.startswith('.')]
# for img_path in tqdm(paths):
#     result = inference_template_matcher(model, str(img_path), str(template_path))
#     _im = show_result(str(img_path), result, model.CLASSES, show=False)
#     save_path = save_dir / img_path.relative_to(img_dir)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(save_path), _im)


save_dir = Path("/tmp/vis/batch1")
img_dir = Path("/data2/datasets/clobotics/chinadrink/labgen/batch1/images")
template_dir = Path("/data2/datasets/clobotics/chinadrink/labgen/batch1/sku")
paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG"))
paths = [p for p in paths if not p.name.startswith('.')]
downscale = 4
for template_path in tqdm(template_dir.glob("*.jpg")):
    for img_path in tqdm(paths):
        result = inference_template_matcher(model, str(img_path), str(template_path))
        _im = show_result(str(img_path), result, model.CLASSES, show=False)
        save_path = save_dir / template_path.stem / img_path.relative_to(img_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _im = cv2.resize(_im, (_im.shape[1] // downscale, _im.shape[0] // downscale))
        cv2.imwrite(str(save_path), _im)
