import argparse
from base_options import BaseTrainOptions

class TestOptions(BaseTrainOptions):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--num_workers', type=int, default=4, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)


        io = self.parser.add_argument_group('io')
        io.add_argument('--dataset_dir', default='../pose-hg-train/data/test_manual', help='Path to the desired dataset')
#        io.add_argument('--checkpoint_dir', help='Path to the desired checkpoint')
        io.add_argument('--checkpoint', help='Path to the desired checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--log_dir', default='/NAS/scratch/karls/logs_rcta', help='Directory to store logs')




        data_proc = self.parser.add_argument_group('Data Preprocessing')
        data_proc.add_argument('--degrees', type=float, default=0, help='Random rotation angle in the range [-degrees, degrees]')
        data_proc.add_argument('--max_scale', type=float, default=1.0)
        data_proc.add_argument('--crop_size', type=int, default=256, help='Size of cropped image to feed to the network')
        fliplr = data_proc.add_mutually_exclusive_group()
        fliplr.add_argument('--flip_lr', dest='flip_lr', action='store_true', help='Flip training images')
        fliplr.add_argument('--no_flip_lr', dest='flip_lr', action='store_false', help='Flip training images')
        apriltag = data_proc.add_mutually_exclusive_group()
        apriltag.add_argument('--apriltag', dest='apriltag', action='store_true', help='Flip training images')
        apriltag.add_argument('--no_apriltag', dest='apriltag', action='store_false', help='Flip training images')
        rr = data_proc.add_mutually_exclusive_group()
        rr.add_argument('--random_rescale', dest='random_rescale', action='store_true', help='Randomly rescale bounding boxes')
        rr.add_argument('--no_random_rescale', dest='random_rescale', action='store_false', help='Do not rescale bounding boxes')
        data_proc.add_argument('--heatmap_size', type=int, default=64, help='Size of output heatmaps')
        data_proc.add_argument('--detection_thresh', type=float, default=1e-1, help='Size of output heatmaps')
        data_proc.add_argument('--dist_thresh', type=float, default=10, help='Size of output heatmaps')
        data_proc.add_argument('--jitter', type=float, default=0.25, help='Amount of image jitter to apply [0, 1]')
        data_proc.set_defaults(flip_lr=True, random_rescale=True, apriltag=True)


        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_keypoints', type=int, default=102, help='Number of distinct keypoint classes')
        train.add_argument('--test_batch_size', type=int, default=8, help='Batch size')

