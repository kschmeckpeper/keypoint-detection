#!/usr/bin/python3

# this is a hack to make it work in the cluster because
#import matplotlib
#matplotlib.use('Agg')

from train_options import TrainOptions
from keypoint_trainer import KeypointTrainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    if options.task == 'keypoints':
        trainer = KeypointTrainer(options)
    else:
        print("The requested option is not supported on this dataset")
        exit()
    trainer.train()
