from __future__ import print_function, division

import copy
import logging
import math
from abc import ABCMeta, abstractmethod

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, models, transforms as T

from mmdet.core import auto_fp16, get_classes, tensor2imgs
from mmdet.core import bbox2result
from mmdet.models import builder
from mmdet.models.registry import DETECTORS
from mmdet.models.qatm.qatm import CreateModel, run_one_sample, run_one_sample_multi_scale

logger = logging.getLogger(__name__)


int_ = lambda x: int(round(x))


def IoU( r1, r2 ):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1; y12 = y11 + h1
    x22 = x21 + w2; y22 = y21 + h2
    x_overlap = max(0, min(x12,x22) - max(x11,x21) )
    y_overlap = max(0, min(y12,y22) - max(y11,y21) )
    I = 1. * x_overlap * y_overlap
    U = (y12-y11)*(x12-x11) + (y22-y21)*(x22-x21) - I
    J = I/U
    return J


def evaluate_iou( rect_gt, rect_pred ):
    # score of iou
    score = [ IoU(i, j) for i, j in zip(rect_gt, rect_pred) ]
    return score


def compute_score( x, w, h ):
    # score of response strength
    k = np.ones( (h, w) )
    score = cv2.filter2D(x, -1, k)
    score[:, :w//2] = 0
    score[:, math.ceil(-w/2):] = 0
    score[:h//2, :] = 0
    score[math.ceil(-h/2):, :] = 0
    return score


def locate_bbox( a, w, h ):
    row = np.argmax( np.max(a, axis=1) )
    col = np.argmax( np.max(a, axis=0) )
    x = col - 1. * w / 2
    y = row - 1. * h / 2
    return x, y, w, h


def score2curve( score, thres_delta = 0.01 ):
    thres = np.linspace( 0, 1, int(1./thres_delta)+1 )
    success_num = []
    for th in thres:
        success_num.append( np.sum(score >= (th+1e-6)) )
    success_rate = np.array(success_num) / len(score)
    return thres, success_rate


def all_sample_iou( score_list, gt_list):
    num_samples = len(score_list)
    iou_list = []
    for idx in range(num_samples):
        score, image_gt = score_list[idx], gt_list[idx]
        w, h = image_gt[2:]
        pred_rect = locate_bbox( score, w, h )
        iou = IoU( image_gt, pred_rect )
        iou_list.append( iou )
    return iou_list


def plot_success_curve( iou_score, title='' ):
    thres, success_rate = score2curve( iou_score, thres_delta = 0.05 )
    auc_ = np.mean( success_rate[:-1] ) # this is same auc protocol as used in previous template matching papers #auc_ = auc( thres, success_rate ) # this is the actual auc
    plt.figure()
    plt.grid(True)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0, 1)
    plt.title(title + 'auc={}'.format(auc_))
    plt.plot( thres, success_rate )
    plt.show()


def nms(score, w_ini, h_ini, thresh=0.7):
    dots = np.array(np.where(score > thresh * score.max()))

    x1 = dots[1] - w_ini // 2
    x2 = x1 + w_ini
    y1 = dots[0] - h_ini // 2
    y2 = y1 + h_ini

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = score[dots[0], dots[1]]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes


class BaseTemplateMatcher(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseTemplateMatcher, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            template (list[Tensor]): list of template of shape (1, C, H, W)
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    # @abstractmethod
    # async def async_simple_test(self, img, img_meta, **kwargs):
    #     pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    # async def aforward_test(self, *, img, img_meta, **kwargs):
    #     for var, name in [(img, 'img'), (img_meta, 'img_meta')]:
    #         if not isinstance(var, list):
    #             raise TypeError('{} must be a list, but got {}'.format(
    #                 name, type(var)))
    #
    #     num_augs = len(img)
    #     if num_augs != len(img_meta):
    #         raise ValueError(
    #             'num of augmentations ({}) != num of image meta ({})'.format(
    #                 len(img), len(img_meta)))
    #     # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
    #     imgs_per_gpu = img[0].size(0)
    #     assert imgs_per_gpu == 1
    #
    #     if num_augs == 1:
    #         return await self.async_simple_test(img[0], img_meta[0], **kwargs)
    #     else:
    #         raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            template (List[Tensor]):
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=False`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=True`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self, data, result, dataset=None, score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr)


@DETECTORS.register_module
class FasterRCNNTemplateMatcher(BaseTemplateMatcher):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 alpha=25,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(BaseTemplateMatcher, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        self.qatm_model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=alpha, use_cuda=True)
        self.qatm_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ]
        )

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def extract_feat(self, img, use_neck=True):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if use_neck and self.with_neck:
            x = self.neck(x)
        return x

    def init_weights(self, pretrained=None):
        super(FasterRCNNTemplateMatcher, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    # @staticmethod
    # def unpad(fmap, img_shape, pad_shape):
    #     h_crop = int(round(fmap.shape[2] * (pad_shape[0] - img_shape[0]) / img_shape[0]))
    #     w_crop = int(round(fmap.shape[3] * (pad_shape[1] - img_shape[1]) / img_shape[1]))
    #     return fmap[:, :, :-h_crop, :-w_crop].contiguous()

    # @staticmethod
    # def run_one_sample(model, template, image, image_name):
    #     val = model(template, image, image_name)
    #     if val.is_cuda:
    #         val = val.cpu()

    @staticmethod
    def scale_rois(rois, scales=[0.9, 1.1]):
        new_rois = []
        w = rois[:, 2] - rois[:, 0] + 1
        h = rois[:, 3] - rois[:, 1] + 1
        center_x = (rois[:, 0] + rois[:, 2]) / 2
        center_y = (rois[:, 1] + rois[:, 3]) / 2
        for s in scales:
            new_w = w * s
            new_h = h * s
            xmin = center_x - (new_w - 1) / 2
            xmax = center_x + (new_w - 1) / 2
            ymin = center_y - (new_h - 1) / 2
            ymax = center_y + (new_h - 1) / 2
            new_rois.append(torch.cat((xmin.reshape(-1, 1), ymin.reshape(-1, 1), xmax.reshape(-1, 1), ymax.reshape(-1, 1)), dim=1))
        return torch.cat(new_rois, dim=0)

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        # alpha = 25
        #
        # i = self.extract_feat(img, use_neck=False)
        # I_feat = torch.cat([
        #     i[0],
        #     F.interpolate(i[1], size=(i[0].shape[2], i[0].shape[3]), mode='bilinear', align_corners=True),
        #     F.interpolate(i[2], size=(i[0].shape[2], i[0].shape[3]), mode='bilinear', align_corners=True),
        #     F.interpolate(i[3], size=(i[0].shape[2], i[0].shape[3]), mode='bilinear', align_corners=True),
        # ], dim=1)
        # I_feat = self.unpad(I_feat, img_meta[0]['img_shape'], img_meta[0]['pad_shape'])
        #
        # t = self.extract_feat(template, use_neck=False)
        # T_feat = torch.cat([
        #     t[0],
        #     F.interpolate(t[1], size=(t[0].shape[2], t[0].shape[3]), mode='bilinear', align_corners=True),
        #     F.interpolate(t[2], size=(t[0].shape[2], t[0].shape[3]), mode='bilinear', align_corners=True),
        #     F.interpolate(t[3], size=(t[0].shape[2], t[0].shape[3]), mode='bilinear', align_corners=True),
        # ], dim=1)
        # T_feat = self.unpad(T_feat, template_meta[0]['img_shape'], template_meta[0]['pad_shape'])
        # ####################################
        # #  Compute ROIs using QATM on FPN
        # ####################################
        # conf_maps = None
        # batchsize_T = T_feat.size()[0]
        # for j in range(batchsize_T):
        #     T_feat_j = T_feat[j].unsqueeze(0)
        #     I_feat_norm, T_feat_j = MyNormLayer()(I_feat, T_feat_j)
        #     dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
        #                         T_feat_j / torch.norm(T_feat_j, dim=1, keepdim=True))
        #     conf_map = QATM(alpha)(dist)
        #     if conf_maps is None:
        #         conf_maps = conf_map
        #     else:
        #         conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        #
        # val = conf_maps.cpu().numpy()
        # val = np.log(val)
        #
        # batch_size = val.shape[0]
        # scores = []
        # for j in range(batch_size):
        #     # compute geometry average on score map
        #     gray = val[j, :, :, 0]
        #     gray = cv2.resize(gray, img_meta[0]['img_shape'][:2][::-1])
        #     h, w = template_meta[0]['img_shape'][:2]
        #     # h = template.size()[-2]
        #     # w = template.size()[-1]
        #     score = compute_score(gray, w, h)
        #     score[score > -1e-7] = score.min()
        #     score = np.exp(score / (h * w))  # reverse number range back after computing geometry average
        #     scores.append(score)
        # scores = np.array(scores)

        # T = self.qatm_transforms(img)[None].cuda()
        scale_factor = img_meta[0]['scale_factor']
        _im = cv2.imread(img_meta[0]['filename'])
        _im = cv2.resize(_im, tuple(int(s * scale_factor) for s in _im.shape[:2][::-1]))
        _im = self.qatm_transforms(_im)[None].cuda()

        _template = cv2.imread(img_meta[0]['template_path'])
        _template = cv2.resize(_template, tuple(int(s * scale_factor) for s in _template.shape[:2][::-1]))
        _template = self.qatm_transforms(_template)[None].cuda()

        template_scales = (0.6, 0.8, 1.0, 1.25, 1.5)
        scores = run_one_sample_multi_scale(self.qatm_model, _template, _im, "test", scales=template_scales)

        boxes = []
        for i, s in enumerate(template_scales):
            h = int(_template.shape[2] * s)
            w = int(_template.shape[3] * s)
            _bx = nms(scores[i], w, h, thresh=0.99)
            boxes.append(_bx)

        rois = torch.tensor(np.concatenate(boxes).reshape(-1, 4)).float()
        # rois = torch.cat((rois, self.scale_rois(rois, scales=[0.9, 1.1])), dim=0)
        rois = torch.cat((torch.tensor([0.0] * len(rois)).reshape(-1, 1), rois), dim=1).to(img.device)

        # s = scores[0]
        # s = ((s - s.min()) / (s.max() - s.min()) * 255).astype(np.uint8)
        _tmp_im = img.cpu().numpy()[0].transpose(1, 2, 0)
        _tmp_im = np.clip((_tmp_im * .226 + .44) * 255, 0, 255).astype(np.uint8)
        _tmp_im = cv2.cvtColor(_tmp_im, cv2.COLOR_RGB2GRAY)
        for r in rois:
            # s = cv2.rectangle(s, (int(r[1]), int(r[2])), (int(r[3]), int(r[4])), (255, 0, 0), 3)
            _tmp_im = cv2.rectangle(_tmp_im, (int(r[1]), int(r[2])), (int(r[3]), int(r[4])), (0, 255, 0), 1)

        x = self.extract_feat(img)

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, rois, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           rois,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.
        :param rois: of shape N x 5, with each row denotes [batch_id, xmin, ymin, xmax, ymax]
        """
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels


# class QATM():
#     def __init__(self, alpha):
#         self.alpha = alpha
#
#     def __call__(self, x):
#         batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
#         x = x.view(batch_size, ref_row * ref_col, qry_row * qry_col)
#         xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
#         xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
#         confidence = torch.sqrt(F.softmax(self.alpha * xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
#         conf_values, ind3 = torch.topk(confidence, 1)
#         ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row * ref_col))
#         ind1 = ind1.flatten()
#         ind2 = ind2.flatten()
#         ind3 = ind3.flatten()
#         if x.is_cuda:
#             ind1 = ind1.cuda()
#             ind2 = ind2.cuda()
#
#         values = confidence[ind1, ind2, ind3]
#         values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
#         return values
#
#     def compute_output_shape(self, input_shape):
#         bs, H, W, _, _ = input_shape
#         return (bs, H, W, 1)
#
#
# class MyNormLayer():
#     def __call__(self, x1, x2):
#         bs, _ , H, W = x1.size()
#         _, _, h, w = x2.size()
#         x1 = x1.view(bs, -1, H*W)
#         x2 = x2.view(bs, -1, h*w)
#         concat = torch.cat((x1, x2), dim=2)
#         x_mean = torch.mean(concat, dim=2, keepdim=True)
#         x_std = torch.std(concat, dim=2, keepdim=True)
#         x1 = (x1 - x_mean) / x_std
#         x2 = (x2 - x_mean) / x_std
#         x1 = x1.view(bs, -1, H, W)
#         x2 = x2.view(bs, -1, h, w)
#         return [x1, x2]
#
#
# class Featex():
#     def __init__(self, model, use_cuda):
#         self.use_cuda = use_cuda
#         self.feature1 = None
#         self.feature2 = None
#         self.model = copy.deepcopy(model.eval())
#         self.model = self.model[:17]
#         for param in self.model.parameters():
#             param.requires_grad = False
#         if self.use_cuda:
#             self.model = self.model.cuda()
#         self.model[2].register_forward_hook(self.save_feature1)
#         self.model[16].register_forward_hook(self.save_feature2)
#
#     def save_feature1(self, module, input, output):
#         self.feature1 = output.detach()
#
#     def save_feature2(self, module, input, output):
#         self.feature2 = output.detach()
#
#     def __call__(self, input, mode='big'):
#         if self.use_cuda:
#             input = input.cuda()
#         _ = self.model(input)
#         if mode == 'big':
#             # resize feature1 to the same size of feature2
#             self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]),
#                                           mode='bilinear', align_corners=True)
#         else:
#             # resize feature2 to the same size of feature1
#             self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]),
#                                           mode='bilinear', align_corners=True)
#         return torch.cat((self.feature1, self.feature2), dim=1)