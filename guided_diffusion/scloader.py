import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pprint
from .tools import *
# class SCDataset(Dataset):
#     def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


#         df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
#         self.name_list = df.iloc[:,0].tolist()
#         self.label_list = df.iloc[:,1].tolist()
#         self.data_path = data_path
#         self.mode = mode

#         self.transform = transform

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, index):
#         """Get the images"""
#         name = self.name_list[index]+'.jpg'
#         img_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',name)
        
#         mask_name = name.split('.')[0] + '_Segmentation.png'
#         msk_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',mask_name)

#         img = Image.open(img_path).convert('RGB')
#         mask = Image.open(msk_path).convert('L')

#         if self.mode == 'Training':
#             label = 0 if self.label_list[index] == 'benign' else 1
#         else:
#             label = int(self.label_list[index])

#         if self.transform:
#             state = torch.get_rng_state()
#             img = self.transform(img)
#             torch.set_rng_state(state)
#             mask = self.transform(mask)

#         if self.mode == 'Training':
#             return (img, mask)
#         else:
#             return (img, mask, name)

#             import sys
import torch.utils.data as data
from os import listdir
import sys
sys.path.append('..')
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import pprint
# import torchvision.transforms as transforms
from sc2image.dataset import StarCraftImage, StarCraftCIFAR10, StarCraftMNIST

class SCDataset(data.Dataset):
    def __init__(self, test_flag=False):
        super(SCDataset, self).__init__()
        # if with_subfolder:
        #     self.samples = self._find_samples_in_subfolders(data_path)
        # else:
        # #     self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        # self.data_path = data_path
        # self.image_shape = image_shape[:-1]
        # self.random_crop = random_crop
        is_train = True if not test_flag else False
        self.test_flag = test_flag
        self.data_dir = Path('..') / 'data'
        self.dataset = StarCraftCIFAR10(self.data_dir, train=is_train, download=True)
        self.path = './SC_2d_full_bg_Fold_0/visuals_step_181000_test_invert/'
        print('*****len(self.dataset)**********',len(self.dataset))
    def __getitem__(self, index):

        img,label = self.dataset[index]
        img =  np.moveaxis(np.array(img), -1, 0)
        bboxes = random_bbox()
        npz_file = os.path.join(self.path,f'{index}.npz')
        data = np.load(npz_file)
        stacked_mask = data['my_array'][-1]
        masked_img = img * stacked_mask
        masked_img[1,:,:] = img[1,:,:]
        mask = stacked_mask[0,:,:]

        # masked_img, mask, invert_result = mask_image(img, bboxes)


        return masked_img, img,mask

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples
    def __len__(self):
        return len(self.dataset)

        
def make_dataset_grid(dataset, n_samples, random_seed=0, label_idx_to_name=None):
    """Make a grid of n_samples from a dataset."""
    n_samples_sqrt = int(np.ceil(np.sqrt(n_samples)))
    n_samples = n_samples_sqrt ** 2  # alter n_samples to a number that is a perfect square

    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    fig, axes = plt.subplots(n_samples_sqrt, n_samples_sqrt, figsize=(n_samples_sqrt*3, n_samples_sqrt*3))
    axes = axes.flatten()
    for ax, sample_idx in zip(axes, indices):
        output = dataset[sample_idx]
        x, y = np.array(output[0]), output[1]
        if x.ndim == 2:
            ax.imshow(x, cmap='gray')
        else:
            if x.ndim == 3:
                if x.shape[0] == 3:
                    x = np.moveaxis(x, 0, -1)
            ax.imshow(x)
        ax.set_title(label_idx_to_name[y] if label_idx_to_name is not None else y)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./visuals.png')
    return fig, axes

def main():
    # from sc2image.dataset import StarCraftMNIST
    # scimage_mnist = StarCraftMNIST(root="'../SCdata/'", download=True)
    import sys
    sys.path.append('..')

    from sc2image.dataset import StarCraftImage, StarCraftCIFAR10, StarCraftMNIST

    data_dir = Path('..') / 'data'
    from sc2image.dataset import _DEFAULT_10_LABELS_DICT
    print('The 10 class labels are:')
    pprint.pprint(_DEFAULT_10_LABELS_DICT)
    cifar = StarCraftCIFAR10(data_dir, train=True, download=True)
    print('len(cifar)',len(cifar))
    print(np.array(cifar[0][0]).shape)
    make_dataset_grid(cifar, n_samples=9, label_idx_to_name=_DEFAULT_10_LABELS_DICT)
    # plt.imshow()
    # plt.imsave('./visuals.png')

if __name__ == "__main__":
    main()