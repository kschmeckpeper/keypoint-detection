import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np


from os.path import join, isdir
from os import makedirs

from tqdm import tqdm

from transforms import RandomFlipLR, RandomRescaleBB, RandomGrayscale, RandomRotation,\
                                RandomBlur, ColorJitter, CropAndResize, LocsToHeatmaps,\
                                ToTensor, Normalize, Denormalize, Select
from models import StackedHourglass
from dataset import RctaDataset
from misc import Pose2DEval
from test_options import TestOptions
from utils import CheckpointSaver
from torch.utils.data import DataLoader

from PIL import Image
class KeypointTester():
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        test_transform_list = []
        if self.options.max_scale > 1:
            test_transform_list.append(RandomRescaleBB(1.0, self.options.max_scale))
        test_transform_list.append(CropAndResize(out_size=(self.options.crop_size, self.options.crop_size)))
        test_transform_list.append(LocsToHeatmaps(out_size=(self.options.heatmap_size, self.options.heatmap_size)))
        test_transform_list.append(ToTensor())
        test_transform_list.append(Normalize())


        self.test_ds = RctaDataset(root_dir=self.options.dataset_dir,
                                   is_train=False,
                                   transform=transforms.Compose(test_transform_list))

        self.model = StackedHourglass(self.options.num_keypoints).to(self.device)
        # Only create optimizer because it is required to restore from checkpoint
        self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=0, momentum=0, weight_decay=0)
        self.models_dict = {'stacked_hg': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        print("log dir:", options.log_dir)
        print("checkpoint dir:", options.checkpoint_dir)
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        print("checkpoint:", self.options.checkpoint)
        self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)


        self.criterion = nn.MSELoss().to(self.device)
        self.pose = Pose2DEval(detection_thresh=self.options.detection_thresh, dist_thresh=self.options.dist_thresh)


    def test(self):
        test_data_loader = DataLoader(self.test_ds,
                                      batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=False)

        pcks = []
        pcks2 = []
        losses = []
        distances = []
        all_object_classes = {}
        for tstep, batch in enumerate(tqdm(test_data_loader, desc='Testing')):
            batch = {k: v.to(self.device) for k,v in batch.items()}

            object_classes = []
            used_keypoints = torch.sum(batch['keypoint_heatmaps'], axis=[2, 3])
            for i in range(batch['keypoint_heatmaps'].shape[0]):
                nonzero = torch.nonzero(used_keypoints[i, :])
                limits = (nonzero[0][0].item(), nonzero[-1][0].item())
                curr_class = None
                for c in all_object_classes:
                    if limits[0] <= c and c <= limits[1]:
                        curr_class = c
                        all_object_classes[c]['limits'] = (min(limits[0], all_object_classes[c]['limits'][0]),
                                                           max(limits[1], all_object_classes[c]['limits'][1]))
                if curr_class is None:
                    curr_class = int((limits[0] + limits[1]) // 2)
                    all_object_classes[curr_class] = {'limits': limits,
                                                      'index':len(all_object_classes),
                                                      'pcks':[],
                                                      'distances': [],
                                                      'dir': join(self.options.log_dir, "test_"+self.options.name, "class_{}".format(len(all_object_classes)))}
                    if not isdir(all_object_classes[curr_class]['dir']):
                        makedirs(all_object_classes[curr_class]['dir'])
                object_classes.append(curr_class)

            print("object classes:", object_classes)
        
            pred_keypoints, loss = self._test_step(batch)
            print("input images:", batch['image'].shape)

            denormed_batch = Denormalize()(batch)

            losses.append(loss.data.cpu().item())
            shape = pred_keypoints[-1].shape
            for i in range(shape[0]):
                pcks.append(self.pose.pck(batch['keypoint_heatmaps'][i].reshape(1, shape[1], shape[2], shape[3]),
                                          pred_keypoints[-1][i].reshape(1, shape[1], shape[2], shape[3])))
                locs = self.pose.heatmaps_to_locs(pred_keypoints[-1][i].reshape(1, shape[1], shape[2], shape[3]), no_thresh=True)
                gt_locs = self.pose.heatmaps_to_locs(batch['keypoint_heatmaps'][i].reshape(1, shape[1], shape[2], shape[3]))

                for k in range(gt_locs.shape[1]):
                    if gt_locs[0][k][0] == 0 and gt_locs[0][k][1] == 0:
                        continue
                    dist = np.sqrt((gt_locs[0][k][0] - locs[0][k][0])**2 + (gt_locs[0][k][1] - locs[0][k][1])**2)
                    all_object_classes[object_classes[i]]['distances'].append(dist)
                    distances.append(dist)
                all_object_classes[object_classes[i]]['pcks'].append(pcks[-1])
                input_image = transforms.ToPILImage()(denormed_batch['image'][i].cpu())
                input_image.save(join(all_object_classes[object_classes[i]]['dir'], "input_im_{:04d}.png".format(len(pcks))))

                print("before shape:", batch['keypoint_heatmaps'][i].shape)
                print("before heatmap:", (torch.sum(batch['keypoint_heatmaps'][i], axis=[0]).cpu()).min(), (torch.sum(batch['keypoint_heatmaps'][i], axis=[0]).cpu()).max())
                print("input_im:", input_image.size, np.array(input_image).max(), np.array(input_image).min())
                gt_heatmap_im = transforms.ToPILImage()(torch.sum(batch['keypoint_heatmaps'][i], axis=[0]).cpu())
                print("gt_hearmap:", gt_heatmap_im.size, np.array(gt_heatmap_im).max(), np.array(gt_heatmap_im).min())
                print("numpy gt:", np.array(gt_heatmap_im).max(), np.array(gt_heatmap_im).min())
                gt_heatmap_with_im = Image.fromarray(np.array(input_image) // 2 + np.array(gt_heatmap_im.resize(input_image.size, Image.ANTIALIAS)).reshape(input_image.size[0], input_image.size[1], 1) // 2)
                gt_heatmap_with_im.save(join(all_object_classes[object_classes[i]]['dir'], "gt_im_with_heatmap_{:04d}.png".format(len(pcks))))

                print("gt_hearmap with:", gt_heatmap_with_im.size, np.array(gt_heatmap_with_im).max(), np.array(gt_heatmap_with_im).min())
                print("before image:", (torch.sum(pred_keypoints[-1][i], axis=[0]).cpu()).min(), (torch.sum(pred_keypoints[-1][i], axis=[0]).cpu()).max(), (torch.sum(pred_keypoints[-1][i], axis=[0]).cpu()).shape)
                pred_heatmap_im = transforms.ToPILImage()(torch.clamp(torch.sum(pred_keypoints[-1][i], axis=[0]).cpu(), 0.0, 1.0))
                pred_heatmap_with_im = Image.fromarray(np.array(input_image) // 2 + np.array(pred_heatmap_im.resize(input_image.size, Image.ANTIALIAS)).reshape(input_image.size[0], input_image.size[1], 1) // 2)
                pred_heatmap_with_im.save(join(all_object_classes[object_classes[i]]['dir'], "pred_im_with_heatmap_{:04d}.png".format(len(pcks))))


            pcks2.append(self.pose.pck(batch['keypoint_heatmaps'], pred_keypoints[-1]))
            print("heatmaps:", pred_keypoints[0].shape, pred_keypoints[-1].shape)
        print("pcks:", pcks)
        print("Means:", np.mean(pcks), "Std error:", np.std(pcks) / np.sqrt(len(pcks)))
        print("Means2:", np.mean(pcks2), "Std error:", np.std(pcks2) / np.sqrt(len(pcks)))
        print("means 1 and means 2 should be equal")
        print("mean loss:", np.mean(losses))

        for c in all_object_classes:
            print("PCK for class:", c, "Mean:", np.mean(all_object_classes[c]['pcks']), "std error:", np.std(all_object_classes[c]['pcks']) / np.sqrt(len(all_object_classes[c]['pcks'])))
            print("Dist for class:", c, "Mean:", np.mean(all_object_classes[c]['distances']), "std error:", np.std(all_object_classes[c]['distances']) / np.sqrt(len(all_object_classes[c]['distances'])))



    def _test_step(self, input_batch):
        self.model.eval()
        images = input_batch['image']
        gt_keypoints = input_batch['keypoint_heatmaps']
        summed = torch.sum(gt_keypoints, axis=[2, 3])
        with torch.no_grad():
            pred_keypoints = self.model(images)
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(pred_keypoints)):
            loss += self.criterion(pred_keypoints[i], gt_keypoints)
        return pred_keypoints, loss


if __name__ == '__main__':
    options = TestOptions().parse_args()

    tester = KeypointTester(options)
    tester.test()
