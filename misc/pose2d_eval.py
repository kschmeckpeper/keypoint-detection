import torch
import numpy as np
from skimage.draw import circle
from IPython.core.debugger import Pdb

class Pose2DEval:

    def __init__(self, detection_thresh=0.1, dist_thresh=10):
        self.detection_thresh = detection_thresh
        self.dist_thresh = dist_thresh

    def heatmaps_to_locs(self, heatmaps, no_thresh=False):
        vals, uv = torch.max(heatmaps.view(heatmaps.shape[0], 
                                        heatmaps.shape[1], 
                                        heatmaps.shape[2]*heatmaps.shape[3]), 2)
        # zero out entries below the detection threshold
        thresh = self.detection_thresh
        if no_thresh:
            self.detection_thresh = 0
        uv *= (vals > self.detection_thresh).type(torch.long) 
        rows = uv / heatmaps.shape[3]
        cols = uv % heatmaps.shape[3]
        return torch.stack([cols, rows], 2).cpu().type(torch.float)

    def pck(self, gt_heatmaps, pred_heatmaps):
        gt_locs = self.heatmaps_to_locs(gt_heatmaps)
        pred_locs = self.heatmaps_to_locs(pred_heatmaps)
        visible_keypoints = (gt_locs[:,:,0] > 0)
        return 100 * torch.mean((torch.sqrt(torch.sum((gt_locs - pred_locs) ** 2, dim=-1))[visible_keypoints] < self.dist_thresh).type(torch.float))

    def draw_keypoints_with_labels(self, images, gt_heatmaps, pred_heatmaps):
        gt_images, pred_images  = images.clone(), images.clone()
        rescale = images.shape[2]/gt_heatmaps.shape[2]
        gt_keypoints = self.heatmaps_to_locs(gt_heatmaps)*rescale
        pred_keypoints = self.heatmaps_to_locs(pred_heatmaps)*rescale
        for i in range(images.shape[0]):
            for gt_keypoint, pred_keypoint in zip(gt_keypoints[i,:,:], pred_keypoints[i,:,:]):
                if gt_keypoint[0] != 0 and gt_keypoint[1] != 0:
                    r,c = circle(gt_keypoint[1], gt_keypoint[0], 3, shape=images.shape[-2:])
                    # blue color for the ground truth keypoints
                    gt_images[i,0,r,c] = 0
                    gt_images[i,1,r,c] = 0
                    gt_images[i,2,r,c] = 1
                if pred_keypoint[0] != 0 and pred_keypoint[1] != 0:
                    r,c = circle(pred_keypoint[1], pred_keypoint[0], 3, shape=images.shape[-2:])
                    correct_prediction = torch.sqrt(torch.sum((gt_keypoint - pred_keypoint) ** 2)) < self.dist_thresh
                    # blue color if predicted keypoint is within the margin, else red
                    val = [0,0,1] if correct_prediction else [1,0,0]
                    pred_images[i,0,r,c] = val[0]
                    pred_images[i,1,r,c] = val[1]
                    pred_images[i,2,r,c] = val[2]
        return gt_images, pred_images

    def draw_keypoints_unlabeled(self, images, pred_heatmaps):
        pred_images  = images.clone()
        rescale = images.shape[2]/pred_heatmaps.shape[2]
        pred_keypoints = self.heatmaps_to_locs(pred_heatmaps)*rescale
        for i in range(images.shape[0]):
            for pred_keypoint in pred_keypoints[i,:,:]:
                if pred_keypoint[0] != 0 and pred_keypoint[1] != 0:
                    r,c = circle(pred_keypoint[1], pred_keypoint[0], 3, shape=images.shape[-2:])
                    # blue color for the predicted keypoints
                    pred_images[i,0,r,c] = 0
                    pred_images[i,1,r,c] = 0
                    pred_images[i,2,r,c] = 1
        return pred_images
