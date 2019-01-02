import torch
import time
import sys
import math
from tqdm import tqdm
tqdm.monitor_interval = 0
from tensorboardX import SummaryWriter
from utils import CheckpointDataLoader, CheckpointSaver

class BaseTrainer:

    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self._init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']
        # self.lr_schedulers = {k: torch.optim.lr_scheduler.ReduceLROnPlateau(v, patience=5)
                                # for k,v in self.optimizers_dict.items()}
        self.lr_schedulers = {k: torch.optim.lr_scheduler.ExponentialLR(v, gamma=self.options.lr_decay, last_epoch=self.epoch_count-1)\
                              for k,v in self.optimizers_dict.items()}
        for opt in self.optimizers_dict:
            self.lr_schedulers[opt].step()

    def _init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    # @profile
    def train(self):
        self.endtime = time.time() + self.options.time_to_run
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=math.ceil(len(self.train_ds)/self.options.batch_size),
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                #if epoch == 1: #step == 74 or step == 73:
                    #from IPython.core.debugger import Pdb
                    #Pdb().set_trace()
                #    print("Epoch", epoch, "Step", step)
                #    print(batch['keypoint_locs'])

                
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) for k,v in batch.items()}
                    out = self._train_step(batch)

                    self.step_count += 1
                    if self.step_count % self.options.summary_steps == 0:
                        try:
                            self._train_summaries(batch, *out)
                        except:
                            from IPython.core.debugger import Pdb
                            Pdb().set_trace()

                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                        tqdm.write('Checkpoint saved')

                    if self.step_count % self.options.test_steps == 0:
                        val_loss = self.test()
                        # for opt in self.optimizers_dict:
                        #     self.lr_schedulers[opt].step(val_loss)
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # apply the learning rate scheduling policy
            for opt in self.optimizers_dict:
                self.lr_schedulers[opt].step()
            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count) 
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count) 
        return

    def _get_lr(self):
        return next(iter(self.optimizers_dict.values())).param_groups[0]['lr']
        # return next(iter(self.lr_schedulers.values())).get_lr()[0]

    def _train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def _train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _save_summaries method')

    def test(self, input_batch):
        raise NotImplementedError('You need to provide a _test_step method')

