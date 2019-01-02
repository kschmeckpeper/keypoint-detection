import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

import time
from tqdm import tqdm
tqdm.monitor_interval = 0

from dataset import RctaDataset
from base_trainer import BaseTrainer
from transforms import RandomFlipLR, RandomRescaleBB, RandomGrayscale, RandomRotation,\
                                RandomBlur, ColorJitter, CropAndResize, LocsToHeatmaps,\
                                ToTensor, Normalize, Denormalize, Select
from models import StackedHourglass
from misc import Pose2DEval

class KeypointTrainer(BaseTrainer):

    def _init_fn(self):
        transform_list = [Select(['image', 'bb', 'keypoints'])]
        transform_list.append(RandomRescaleBB(0.9, 1.4))
        # transform_list.append(RandomFlipLR())
        transform_list.append(RandomBlur())
        transform_list.append(RandomGrayscale())
        transform_list.append(ColorJitter(brightness=self.options.jitter, contrast=self.options.jitter, saturation=self.options.jitter, hue=self.options.jitter/4))
        transform_list.append(RandomRotation(degrees=self.options.degrees))
        transform_list.append(CropAndResize(out_size=(self.options.crop_size, self.options.crop_size)))
        transform_list.append(LocsToHeatmaps(out_size=(self.options.heatmap_size, self.options.heatmap_size)))
        transform_list.append(ToTensor())
        transform_list.append(Normalize())

        test_transform_list = [CropAndResize(out_size=(self.options.crop_size, self.options.crop_size))]
        test_transform_list.append(LocsToHeatmaps(out_size=(self.options.heatmap_size, self.options.heatmap_size)))
        test_transform_list.append(ToTensor())
        test_transform_list.append(Normalize())

        self.train_ds = RctaDataset(root_dir=self.options.dataset_dir, is_train=True,
                                   transform = transforms.Compose(transform_list))
        print("Keypoints in trainer:", self.train_ds.keypoints[74*6])
        print("Bounding boxes:", self.train_ds.bounding_boxes[74*6])
        self.test_ds = RctaDataset(root_dir=self.options.dataset_dir, is_train=False,
                                  transform = transforms.Compose(test_transform_list))
        self.model = StackedHourglass(num_hg=self.options.num_hg, hg_channels=self.options.hg_channels, out_channels=102).to(self.device)
        print('Total number of model parameters:', self.model.num_trainable_parameters())
        # create optimizer
        # if self.options.optimizer == 'sgd':
        #     self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.options.lr, momentum=self.options.sgd_momentum, weight_decay=self.options.wd)
        # elif self.options.optimizer == 'rmsprop':
        self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=self.options.lr, momentum=0, weight_decay=self.options.wd)
        # else:
        #     self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.options.lr, betas=(self.options.adam_beta1, 0.999), weight_decay=self.options.wd)

        # pack all models and optimizers in dictionaries to interact with the checkpoint saver
        self.models_dict = {'stacked_hg': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        self.criterion = nn.MSELoss(size_average=True).to(self.device)
        self.pose = Pose2DEval(detection_thresh=self.options.detection_thresh, dist_thresh=self.options.dist_thresh)

    def _train_step(self, input_batch):
        self.model.train()
        images = input_batch['image']
        print "Image_shape:", images.shape
        gt_keypoints = input_batch['keypoint_heatmaps']
        pred_keypoints = self.model(images)
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(pred_keypoints)):
            loss += self.criterion(pred_keypoints[i], gt_keypoints)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return [pk.detach() for pk in pred_keypoints], loss.detach()
    
    def _train_summaries(self, batch, pred_keypoints, loss):
        pck = self.pose.pck(batch['keypoint_heatmaps'], pred_keypoints[-1])
        self._save_summaries(batch, pred_keypoints, loss, pck, self.step_count, is_train=True) 

    def test(self):
        test_data_loader = DataLoader(self.test_ds, batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test)
        test_loss = torch.tensor(0.0, device=self.device)
        mean_pck = 0.0
        for tstep, batch in enumerate(tqdm(test_data_loader, desc='Testing')):
            if time.time() < self.endtime:
                batch = {k: v.to(self.device) for k,v in batch.items()}
                pred_keypoints, loss = self._test_step(batch)
                test_loss += loss.data
                mean_pck += self.pose.pck(batch['keypoint_heatmaps'], pred_keypoints[-1])
            else:
                tqdm.write('Testing interrupted at step ' + str(tstep))
                break
        test_loss /= (tstep+1)
        mean_pck /= (tstep+1)
        self._save_summaries(batch, pred_keypoints, test_loss, mean_pck, self.step_count, is_train=False) 

    def _test_step(self, input_batch):
        self.model.eval()
        images = input_batch['image']
        gt_keypoints = input_batch['keypoint_heatmaps']
        with torch.no_grad():
            pred_keypoints = self.model(images)
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(pred_keypoints)):
            loss += self.criterion(pred_keypoints[i], gt_keypoints)
        return pred_keypoints, loss

    def _save_summaries(self, input_batch, pred_keypoints, loss, pck, step, is_train=True):
        prefix = 'train/' if is_train else 'test/'
        input_batch = Denormalize()(input_batch)
        images = input_batch['image']
        gt_keypoints = input_batch['keypoint_heatmaps']

        gt_image_keypoints = []
        pred_image_keypoints = []
        gt_image_keypoints, pred_image_keypoints = self.pose.draw_keypoints_with_labels(images, gt_keypoints, pred_keypoints[-1])

        gt_image_keypoints_grid = make_grid(gt_image_keypoints, pad_value=1, nrow=3)
        pred_image_keypoints_grid = make_grid(pred_image_keypoints, pad_value=1, nrow=3)

        pred_heatmaps_grid = make_grid(pred_keypoints[-1][0,:,:,:].unsqueeze(0).transpose(0,1), pad_value=1, nrow=5)
        pred_heatmaps_grid[pred_heatmaps_grid > 1] = 1
        pred_heatmaps_grid[pred_heatmaps_grid < 0] = 0

        self.summary_writer.add_scalar(prefix + 'loss', loss, step)
        self.summary_writer.add_scalar(prefix + 'PCK', pck, step)
        self.summary_writer.add_image(prefix + 'gt_image_keypoints', gt_image_keypoints_grid, step)
        self.summary_writer.add_image(prefix + 'pred_image_keypoints', pred_image_keypoints_grid, step)
        self.summary_writer.add_image(prefix + 'pred_heatmaps_image1', pred_heatmaps_grid, step)
        if is_train:
            self.summary_writer.add_scalar('lr', self._get_lr(), step)
        return
