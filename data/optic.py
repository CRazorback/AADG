import os
import random

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir,
                 phase='train',
                 splitid=[2, 3, 4],
                 transform=None,
                 state='train',
                 ):
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.image_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.label_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.img_name_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        SEED = 1023
        random.seed(SEED)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})

        self.transforms = transform
        self._read_img_into_memory()
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        # Display stats
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample=anco_sample
        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            if self.splitid[0] == 4:
                self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').crop((144, 144, 144+512, 144+512)).resize((256, 256), Image.LANCZOS))
                _target = np.asarray(Image.open(self.image_list[index]['label']).convert('L'))
                _target = _target[144:144+512, 144:144+512]
                _target = Image.fromarray(_target)
            else:
                self.image_pool[Flag].append(
                    Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                _target = Image.open(self.image_list[index]['label'])

            if _target.mode == 'RGB':
                _target = _target.convert('L')
            if self.state != 'prediction':
                _target = _target.resize((256, 256))
            self.label_pool[Flag].append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)
