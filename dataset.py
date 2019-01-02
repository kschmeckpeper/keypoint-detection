import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from os.path import join
import h5py
from IPython.core.debugger import Pdb

class RctaDataset(Dataset):
    def __init__(self, root_dir='~/pose-hg-train/data/combined/', is_train=True, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform

        file_type = 'train'
        if not is_train:
            file_type = 'valid'
        annot_file = h5py.File(join(root_dir, "annot", file_type+'.h5'), 'r')
        keypoints = annot_file['part']
        centers = annot_file['center']
        scales = np.array(annot_file['scale']) * 100


        self.keypoints = []
        self.bounding_boxes = []
        self.old_boxes = []

        for i in range(len(keypoints)):
            keypoints_with_visibility = np.zeros((keypoints[i].shape[0], 3))
            keypoints_with_visibility[:, :2] = keypoints[i]
            keypoints_with_visibility[np.where(keypoints_with_visibility[:, 0] > 0), 2] = 1

                
            self.keypoints.append(keypoints_with_visibility)

            if len(np.where(keypoints_with_visibility[:, 2] > 0)[0]) == 0:
                min_x = 1
                min_y = 1
                
                max_x = np.inf
                max_y = np.inf

            else:
                #print keypoints_with_visibility[np.where(keypoints_with_visibility[:, 2] > 0), 0], len(np.where(keypoints_with_visibility[:, 2] > 0)[0])
                min_x = np.min(keypoints_with_visibility[np.where(keypoints_with_visibility[:, 2] > 0), 0])
                max_x = np.max(keypoints_with_visibility[np.where(keypoints_with_visibility[:, 2] > 0), 0]) 

                min_y = np.min(keypoints_with_visibility[np.where(keypoints_with_visibility[:, 2] > 0), 1]) 
                max_y = np.max(keypoints_with_visibility[np.where(keypoints_with_visibility[:, 2] > 0), 1])

            box = np.zeros((4, 2))
            min_box_x = min(min_x-1, centers[i][0] - scales[i])
            max_box_x = max(max_x+1, centers[i][0] + scales[i])
            min_box_y = min(min_y-1, centers[i][1] - scales[i])
            max_box_y = max(max_y+1, centers[i][1] + scales[i])
 #           if i == 153:
 #               print(min_x, centers[i][0] - scales[i], min_box_x)
 #               print(max_x, centers[i][0] + scales[i], max_box_x)
 #               print(min_y, centers[i][1] - scales[i], min_box_y)
 #               print(max_y, centers[i][1] + scales[i], max_box_y)
            box[0, 0] = min_box_x
            box[0, 1] = min_box_y

            box[1, 0] = max_box_x
            box[1, 1] = min_box_y

            box[2, 0] = min_box_x
            box[2, 1] = max_box_y

            box[3, 0] = max_box_x
            box[3, 1] = max_box_y
            self.bounding_boxes.append(box)
            self.old_boxes.append(np.copy(box))


#        if len(keypoints) > 74 * 6:
#            print("Keypoints_old", keypoints[74*6])
#            print("Keypoints_new", self.keypoints[74*6])
        self.old_keypoints = keypoints


        self.filenames = open(join(root_dir, "annot", file_type+'_images.txt'), 'r').read().split('\n')[:-1]


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = join(self.root_dir, 'images', self.filenames[idx])
        if len(self.filenames[idx]) < 1:
            print(idx, img_path)
        image = Image.open(img_path).convert(mode='RGB')
        bounding_box = self.bounding_boxes[idx].copy()
        keypoints = self.keypoints[idx].copy()

        min_x, min_y = np.round(np.maximum(bounding_box.min(axis=0), np.array([0,0]))).astype(int)
        max_x, max_y = np.round(np.minimum(bounding_box.max(axis=0), np.array(image.size))).astype(int)

#        for i in range(keypoints.shape[0]):
#            if keypoints[i, 2] == 0:
#                continue
#            if keypoints[i,0] < min_x or keypoints[i,0] > max_x or keypoints[i,1] < min_y or keypoints[i,1] > max_y:
#                Pdb().set_trace()
        sample = {'image': image, 'bb': bounding_box, 'keypoints': keypoints}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
