import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFilter
from IPython.core.debugger import Pdb
from skimage.draw import circle
# # randomly flip and image horizontally with prob 0.5
# # assumes bounding box edges are parallel to the x and y axes

class RandomMorph:

    def __call__(self, sample):
        n = np.random.rand()
        new_mask = sample['mask'].copy()
        sample['in_mask'] = new_mask
        # was 2 and 3 initially
        k = 2*round(4*np.random.rand())+1
        if n < 0.33:
            return sample
        elif n < 0.67:
            sample['in_mask'] = new_mask.filter(ImageFilter.MinFilter(k))
        else:
            sample['in_mask'] = new_mask.filter(ImageFilter.MaxFilter(k))
        return sample

class RandomOcclusion:

    def __call__(self, sample):
        n = np.random.rand()
        if n < 0.7:
            return sample
        img_center = sample['image'].size[0]//2
        r, c = img_center + img_center/3*np.random.randn(2)
        if r < 0 or r > 255 or c < 0 or c > 255:
            return sample
        rad = img_center//8 + img_center//8*np.random.rand()
        [rr,cc] = circle(r, c, rad, shape=sample['image'].size[::-1])
        occlusion_mask = 255*np.ones(shape=sample['image'].size[::-1]).astype(np.uint8)
        occlusion_mask[rr,cc] = 0
        occluded_mask = Image.fromarray(np.zeros_like(occlusion_mask))
        occlusion_mask = Image.fromarray(occlusion_mask).convert(mode='L')
        sample['in_mask'] = Image.composite(sample['in_mask'], occluded_mask, occlusion_mask).convert(mode='L')
        return sample

class RandomAprilTag:
    
    def __init__(self, prob=1):
        self.prob = prob

    def __call__(self, sample):
        n = np.random.rand()
        if n > self.prob:
            return sample
        apriltag = 1.0*(np.random.rand(8,8) > 0.6)
        apriltag[[0, 7], :] = 0.0
        apriltag[:, [0, 7]] = 0.0
        scale = round(6+3*np.random.rand())
        apriltag = apriltag.repeat(scale, axis=0).repeat(scale, axis=1)
        new_mask = sample['in_mask']
        cols, rows = new_mask.size
        row, col = np.round([rows, cols] * np.random.rand(2)).astype(np.int)
        row_min = min(max(row, 0), rows-1)
        col_min = min(max(col, 0), cols-1)
        row_max = min(row_min+apriltag.shape[0],rows-1)
        col_max = min(col_min+apriltag.shape[1],cols-1)
        apriltag = apriltag[0:row_max-row_min, 0:col_max-col_min]
        apriltag = Image.fromarray(np.uint8(255*apriltag))
        # Pdb().set_trace()
        new_mask.paste(apriltag,box=(col_min, row_min, col_max, row_max))
        new_mask = new_mask.point(lambda p: p > 128 and 255) 
        # new_mask[row_min:row_max, col_min:col_max] = apriltag[0:row_max-row_min, 0:col_max-col_min]
        sample['in_mask'] = new_mask
        return sample

class Select:
    
    def __init__(self, select_keys):
        self.select_keys = select_keys

    def __call__(self, sample):
        new_sample = {}
        for key in self.select_keys:
            new_sample[key] = sample[key]
        return new_sample  

class RandomBlur:

    def __call__(self, sample):
        if np.random.rand() < 0.5:
            return sample
        sample['image'] = sample['image'].filter(ImageFilter.BLUR)
        return sample

class RandomBlend:

    def __init__(self, img_list):
        self.img_list = img_list

    def __call__(self, sample):
        if np.random.rand() < 0.5:
            return sample
        ind = round((len(self.img_list)-1)*np.random.rand())
        background_image = Image.open(self.img_list[ind])
        mask = sample['mask'].convert('RGB').filter(ImageFilter.BLUR).convert('L')
        rc = transforms.RandomCrop(size=mask.size[::-1], pad_if_needed=True)
        background = rc(background_image)
        sample['image'] = Image.composite(sample['image'], background, mask)
        return sample


class ColorJitter(transforms.ColorJitter):

    def __call__(self, sample):
        if (np.random.rand() > 0.5):
            sample['image'] = super(ColorJitter, self).__call__(sample['image'])
        return sample

class RandomGrayscale(transforms.RandomGrayscale):

    def __call__(self, sample):
        sample['image'] = super(RandomGrayscale, self).__call__(sample['image'])
        return sample

class RandomFlipLR:
    
    def __call__(self, sample):
        flip = np.random.random() > 0.5
        if not flip: 
            return sample
        cols, rows = sample['image'].size
        sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
        if 'mask' in sample:
            sample['mask'] = sample['mask'].transpose(Image.FLIP_LEFT_RIGHT)
        bb = sample['bb']
        bb[:,0] = cols-1-bb[:,0]
        sample['bb'] = bb[[1,0,3,2],:]
        if 'keypoints' in sample:
            keypoints = sample['keypoints']
            visible_keypoints = keypoints[:,-1] != 0
            keypoints[visible_keypoints,0] = cols-1-keypoints[visible_keypoints,0]
            for i in range(3):
                keypoints[6+i,:], keypoints[10+i,:] = keypoints[10+i,:], keypoints[6+i,:]
        return sample

# randomly rescale bounding box
# assumes bounding box edges are parallel to the x and y axes
class RandomRescaleBB:

    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        scale = (self.max_scale-self.min_scale)*np.random.rand() + self.min_scale
        bb = sample['bb']
        bb_width = bb[1,0] - bb[0,0]
        bb_height = bb[2,1] - bb[0,1]
        cols, rows = sample['image'].size
        bb[[0,2],0] = np.maximum(bb[[0,2],0] - (scale-1)/2 * bb_width, 0)
        bb[[1,3],0] = np.minimum(bb[[1,3],0] + (scale-1)/2 * bb_width, cols-1)
        bb[[0,1],1] = np.maximum(bb[[0,1],1] - (scale-1)/2 * bb_height, 0)
        bb[[2,3],1] = np.minimum(bb[[2,3],1] + (scale-1)/2 * bb_height, rows-1)
        return sample

# random rotation of the image
# the bounding box will be also rotated after this operation
class RandomRotation:

    def __init__(self, degrees=30):
        self.degrees = degrees

    def __call__(self, sample):
        rd = self.degrees*2*(np.random.rand()-0.5)
        R = np.array([[np.cos(np.pi/180*(-rd)), -np.sin(np.pi/180*(-rd))],
                           [np.sin(np.pi/180*(-rd)), np.cos(np.pi/180*(-rd))]]) 
        bb = sample['bb']
        img_center = np.array(sample['image'].size)/2
        sample['image'] = sample['image'].rotate(rd)
        if 'mask' in sample:
            sample['mask'] = sample['mask'].rotate(rd)
        for i in range(bb.shape[0]):
            bb[i,:] = np.matmul(R,(bb[i,:]-img_center)) + img_center

        if 'keypoints' in sample:
            keypoints = sample['keypoints']
            for i in range(keypoints.shape[0]):
                if keypoints[i,2] != 0:
                    keypoints[i,0:2] = np.matmul(R,(keypoints[i,0:2]-img_center)) + img_center
        return sample

# find the tightest bounding box, crop the image and rescale it to a
# fixed resolution
class CropAndResize:

    def __init__(self, out_size=(256,256)):
        self.out_size = out_size[::-1]

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        img_size = image.size
        center = sample['center']
        min_x, min_y = np.round(np.maximum(bb.min(axis=0), np.array([0,0]))).astype(int)
        max_x, max_y = np.round(np.minimum(bb.max(axis=0), np.array(img_size))).astype(int)
        sample['image'] = image.crop(box=(min_x,min_y,max_x,max_y))
        cropped_size = sample['image'].size

        if cropped_size[0] != cropped_size[1]:
            # new shape to preserve aspect ratio
            ratio = min(self.out_size[0]/cropped_size[0], self.out_size[1]/cropped_size[1])
            im = sample['image'].resize((round(ratio*cropped_size[0]), round(ratio*cropped_size[1])))
            # zero pad
            # want to preserve center loc, so ...
            offset_x = int(self.out_size[0]//2 - round((center[0] - min_x)*ratio))
            offset_y = int(self.out_size[1]//2 - round((center[1] - min_y)*ratio))
            # corner_offset = ((self.out_size[0]-im.size[0])//2, (self.out_size[1]-im.size[1])//2)
            new_im = Image.new('RGB', self.out_size, 0)
            new_im.paste(im, (offset_x, offset_y))
            sample['image'] = new_im
        else:
            ratio = self.out_size[0]/cropped_size[0]
            offset_x = 0
            offset_y = 0
            sample['image'] = sample['image'].resize(self.out_size)

        if 'mask' in sample:
            sample['mask'] = sample['mask'].crop(box=(min_x,min_y,max_x,max_y)).resize(self.out_size)
        if 'keypoints' in sample:
            keypoints = sample['keypoints']
            for i in range(keypoints.shape[0]):
                if keypoints[i,2] != 0:
                    if keypoints[i,0] < min_x or keypoints[i,0] > max_x or keypoints[i,1] < min_y or keypoints[i,1] > max_y:
#                        Pdb().set_trace()
                        keypoints[i,:] = [0,0,0]
                    else:
                        keypoints[i,0] = (keypoints[i,0] - min_x)*ratio + offset_x
                        keypoints[i,1] = (keypoints[i,1] - min_y)*ratio + offset_y
                        # keypoints[i,:2] = (keypoints[i,:2]-np.array([min_x, min_y]))*self.out_size/cropped_size
        # sample.pop('bb')
        return sample

# Convert keypoint locations to heatmaps
class LocsToHeatmaps:

    def __init__(self, img_size=(256,256), out_size=(64,64), sigma=1):
        self.img_size = img_size
        self.out_size = out_size
        self.x_scale = out_size[0]/img_size[0]
        self.y_scale = out_size[1]/img_size[1]
        self.sigma=sigma
        x = np.arange(0, out_size[1], dtype=np.float)
        y = np.arange(0, out_size[0], dtype=np.float)
        self.yg, self.xg = np.meshgrid(y,x, indexing='ij')
        return

    def __call__(self, sample):
        gaussian_hm = np.zeros((self.out_size[0], self.out_size[1], sample['keypoints'].shape[0]))
        for i,keypoint in enumerate(sample['keypoints']):
            if keypoint[2] != 0:
                kp_x = keypoint[0] * self.x_scale
                kp_y = keypoint[1] * self.y_scale
                gaussian_hm[:,:,i] = np.exp(-((self.xg-kp_x)**2+(self.yg-kp_y)**2)/(2*self.sigma**2))
        sample['keypoint_locs'] = sample['keypoints'][:,:2]
        sample['visible_keypoints'] = sample['keypoints'][:,2]
        sample['keypoint_heatmaps'] = gaussian_hm
        return sample

# Convert numpy arrays to Tensor objects
# Permute the image dimensions
class ToTensor:

    def __init__(self, downsample_mask=False):
        self.tt = transforms.ToTensor()
        self.downsample_mask=downsample_mask

    def __call__(self, sample):
        sample['image'] = self.tt(sample['image'])
        if 'mask' in sample:
            if self.downsample_mask:
                sample['mask'] = self.tt(sample['mask'].resize((64,64), Image.ANTIALIAS))
            else:
                sample['mask'] = self.tt(sample['mask'])
        if 'in_mask' in sample:
            sample['in_mask'] = self.tt(sample['in_mask'])
            # sample['in_mask'] = sample['in_mask'].unsqueeze(0)
        if 'keypoint_heatmaps' in sample:
            sample['keypoint_heatmaps'] =\
                torch.from_numpy(sample['keypoint_heatmaps'].astype(np.float32).transpose(2,0,1))
            sample['keypoint_locs'] =\
                torch.from_numpy(sample['keypoint_locs'].astype(np.float32))
            sample['visible_keypoints'] =\
                torch.from_numpy(sample['visible_keypoints'].astype(np.float32))
        return sample

class Normalize:

    def __call__(self, sample):
        sample['image'] = 2*(sample['image']-0.5)
        if 'in_mask' in sample:
            sample['in_mask'] = 2*(sample['in_mask']-0.5)
        return sample

class Denormalize:

    def __call__(self, sample):
        sample['image'] = 0.5*(sample['image']+1)
        if 'in_mask' in sample:
            sample['in_mask'] = 0.5*(sample['in_mask']+1)
        return sample
