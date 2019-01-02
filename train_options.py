import argparse
from base_options import BaseTrainOptions

class TrainOptions(BaseTrainOptions):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')
        task  = req.add_mutually_exclusive_group(required=True)
        task.add_argument('--segmentation', dest='task', action='store_const', const='segmentation')
        task.add_argument('--keypoints', dest='task', action='store_const', const='keypoints')
        task.add_argument('--joint', dest='task', action='store_const', const='joint')
        task.add_argument('--joint_gan', dest='task', action='store_const', const='joint_gan')
        task.add_argument('--joint_ref', dest='task', action='store_const', const='joint_ref')
        task.add_argument('--autoencoder', dest='task', action='store_const', const='autoencoder')
        task.add_argument('--k2m', dest='task', action='store_const', const='k2m')
        task.add_argument('--k2m_gan', dest='task', action='store_const', const='k2m_gan')
        req.set_defaults(task='keypoints')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=3600, help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=4, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--dataset_dir', default='~/pose-hg-train/data/combined', help='Path to the desired dataset')
        io.add_argument('--log_dir', default='../logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')

        data_proc = self.parser.add_argument_group('Data Preprocessing')
        data_proc.add_argument('--degrees', type=float, default=45, help='Random rotation angle in the range [-degrees, degrees]')
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
        arch_hg = self.parser.add_argument_group('Hourglass Architecture')
        arch_hg.add_argument('--hg_channels', type=int, default=256, help='Number of channels for the Hourglass') 
        arch_hg.add_argument('--num_hg', type=int, default=2, help='Number of stacked Hourglasses') 
        arch_hg.add_argument('--num_resblocks', type=int, default=1, help='Number of stacked residual blocks') 

        arch_unet = self.parser.add_argument_group('UNet Architecture')
        arch_unet.add_argument('--num_filters', type=int, default=64, help='Number of filters in conv1') 
        arch_unet.add_argument('--num_blocks', type=int, default=5, help='Number of blocks') 
        arch_unet.add_argument('--unet_type', default='v2', help='Number of blocks') 
        arch_unet.add_argument('--mask_only', dest='mask_only', default=False, action='store_true', help='Number of blocks') 

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=100, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=6, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=8, help='Batch size')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true', help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false', help='Don\'t shuffle testing data')
        train.set_defaults(shuffle_train=True, shuffle_test=True)
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Chekpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=500, help='Testing frequency')
        train.add_argument('--test_iters', type=int, default=200, help='Number of testing iterations')


        optim = self.parser.add_argument_group('Optimization')
        optim_type = optim.add_mutually_exclusive_group()
        optim_type.add_argument('--use_sgd', dest='optimizer', action='store_const', const='sgd',help='Use SGD (default Adam)')
        optim_type.add_argument('--use_rmsprop', dest='optimizer', action='store_const', const='rmsprop',help='Use  (default Adam)')
        optim_type.add_argument('--use_adam', dest='optimizer', action='store_const', const='adam',help='Use SGD (default Adam)')
        optim.add_argument('--adam_beta1', type=float, default=0.9, help='Value for Adam Beta 1')
        optim.add_argument('--sgd_momentum', type=float, default=0.0, help='Momentum for SGD')
        optim.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
        optim.add_argument("--lr_decay", type=float, default=0.98, help="Exponential decay rate")
        optim.add_argument("--wd", type=float, default=0, help="Weight decay weight")
        optim.add_argument('--keypoint_lw', type=float, default=100, help='Keypoint loss weight')
        optim.add_argument('--gan_mask_lw', type=float, default=10, help='Gan mask loss weight')
        optim.set_defaults(optimizer='rmsprop')

        return 
