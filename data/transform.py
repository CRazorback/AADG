import random
import numbers
import torch

import numpy as np

# from torchvision.transforms import transforms, InterpolationMode
from torchvision.transforms import transforms
from PIL import Image, ImageOps
from data.basic import CutoutDefault, augment_list


NUM_MAGS = 10
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_ORIGA_MEAN, _ORIGA_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, img, mask):
        # print(img.size)
        w, h = img.size
        if self.padding > 0 or w < self.size[0] or h < self.size[1]:
            padding = np.maximum(self.padding,np.maximum((self.size[0]-w)//2+5,(self.size[1]-h)//2+5))
            img = ImageOps.expand(img, border=padding, fill=0)
            mask = ImageOps.expand(mask, border=padding, fill=0)

        assert img.width == mask.width
        assert img.height == mask.height
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return img, mask

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        # print(img.size)
        return img, mask


class DGRandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding
        self.crop = RandomCrop(size, padding)

    def __call__(self, sample):
        sample['image'], sample['label'] = self.crop(sample['image'], sample['label'])
        return sample


class RandomScaleCrop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # print(img.size)
        assert img.width == mask.width
        assert img.height == mask.height

        seed = random.random()
        if seed > 0.5:
            w = int(random.uniform(1, 1.5) * img.size[0])
            h = int(random.uniform(1, 1.5) * img.size[1])

            img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        sample['image'], sample['label'] = self.crop(img, mask)
        
        return sample


class DGRandomScaleCrop(object):
    def __init__(self, size, scale_range=[1, 1.5]):
        self.size = size
        self.scale_range = scale_range
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def scale(self, img, mask):
        seed = random.random()
        if seed > 0.2:
            w = int(random.uniform(self.scale_range[0], self.scale_range[1]) * img.size[0])
            h = int(random.uniform(self.scale_range[0], self.scale_range[1]) * img.size[1])

            img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return img, mask

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # print(img.size)
        assert img.width == mask.width
        assert img.height == mask.height

        scaled_img, scaled_mask = self.scale(img, mask.copy())
        sample['image'], sample['label'] = self.crop(scaled_img, scaled_mask)

        if 'aug_images' in sample.keys():
            aug_images = []
            aug_labels = []
            for img in sample['aug_images']:
                scaled_img, scaled_mask = self.scale(img, mask.copy())
                cropped_img, cropped_mask = self.crop(scaled_img, scaled_mask)
                aug_images.append(cropped_img)
                aug_labels.append(cropped_mask)
            
            sample['aug_images'], sample['aug_labels'] = aug_images, aug_labels
        
        return sample


class Normalize_dg(object):
    """Normalize augmented tensor images with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, dataset_name, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name

    def normalize(self, img, mask):
        img = np.array(img, dtype=np.float32)
        img /= 127.5
        img -= 1.0
        __mask = np.array(mask).astype(np.uint8)
        _mask = np.zeros_like(__mask)
        if self.dataset_name == 'optic':
            _mask[__mask > 200] = 255
            # index = np.where(__mask > 50 and __mask < 201)
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
        elif self.dataset_name == 'brain':
            mask = np.expand_dims(_mask, axis=2)
        else:
            _mask[__mask != 0] = 1
            mask = np.expand_dims(_mask, axis=2)

        return img, mask

    def __call__(self, sample):
        if 'aug_images' in sample.keys():
            # NCHW
            for i, (image, label) in enumerate(zip(sample['aug_images'], sample['aug_labels'])):
                augmented_img, augmented_mask = self.normalize(image, label)
                sample['aug_images'][i] = augmented_img
                sample['aug_labels'][i] = augmented_mask
        
        img, mask = self.normalize(sample['image'], sample['label'])
        sample['image'] = img
        sample['label'] = mask

        return sample  


class GenerateMask(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def normalize(self, mask):
        __mask = np.array(mask).astype(np.uint8)
        _mask = np.zeros_like(__mask)
        _mask[__mask != 0] = 1
        mask = np.expand_dims(_mask, axis=2).astype(np.uint8).transpose((2, 0, 1))
        mask_t = torch.from_numpy(mask).bool()

        return mask_t

    def __call__(self, sample):
        sample['roi'] = self.normalize(sample['roi'])

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, dataset_name) -> None:
        super().__init__()
        if dataset_name in ['optic', 'vessel']:
            self.n = 3
        else:
            self.n = 2

    def __call__(self, sample):    
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        map = np.array(sample['label']).astype(np.uint8).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        map = torch.from_numpy(map).float()
        sample['image'] = img
        sample['label'] = map
        domain_code = torch.from_numpy(SoftLable(ToMultiLabel(sample['dc'], self.n))).float()
        sample['dc'] = domain_code

        if 'aug_images' in sample.keys():
            augmented_img = np.array(sample['aug_images']).astype(np.float32).transpose((0, 3, 1, 2))
            augmented_map = np.array(sample['aug_labels']).astype(np.uint8).transpose((0, 3, 1, 2))
            augmented_img = torch.from_numpy(augmented_img).float()
            augmented_map = torch.from_numpy(augmented_map).float()
            sample['aug_images'] = augmented_img.contiguous()
            sample['aug_labels'] = augmented_map.contiguous()
            sample['dc'] = torch.stack([domain_code]*augmented_img.size(0), dim=0).contiguous()

        return sample   


class Identity(object):
    def __call__(self, sample):
        return sample


def to_multilabel(pre_mask, classes=2):
    mask = np.zeros((*pre_mask.shape, classes))
    mask[pre_mask==1] = [0, 1]
    mask[pre_mask==2] = [1, 1]

    return mask


def ToMultiLabel(dc, c):
    new_dc = np.zeros([c])
    for i in range(new_dc.shape[0]):
        if i == dc:
            new_dc[i] = 1
            return new_dc 


def SoftLable(label):
    new_label = label.copy()
    label = list(label)
    index = label.index(1)
    new_label[index] = 0.8 + random.random()*0.2
    accelarate = new_label[index]
    for i in range(len(label)):
        if i != index:
            if i == len(label) - 1:
                new_label[i] = 1 - accelarate
            else:
                new_label[i] = random.random()*(1 - accelarate)
                accelarate += new_label[i]

    return new_label


def identity(x):
    return [x]


def get_dg_segtransform(dataset):
    if 'optic' in dataset:
        transform_train_imgs = transforms.Compose([       
            Identity(), ## Locate new policy
            DGRandomScaleCrop(256),
            Normalize_dg('optic'),
            ToTensor('optic')
        ])
        transform_test_imgs = transforms.Compose([
            DGRandomCrop(256),
            Normalize_dg('optic'),
            ToTensor('optic')
        ])
    elif 'rvs' in dataset:
        transform_train_imgs = transforms.Compose([       
            Identity(), ## Locate new policy
            DGRandomScaleCrop(256, scale_range=[0.5, 2]),
            Normalize_dg('vessel'),
            ToTensor('vessel')
        ])
        transform_test_imgs = transforms.Compose([
            # DGRandomCrop(512),
            # Identity(),
            Normalize_dg('vessel'),
            ToTensor('vessel'),
            GenerateMask('vessel')
        ])

    return transform_train_imgs, transform_test_imgs


def train_collate_fn(batch):
    """
    batch = [((M,3,H,W), label)]*batch_size
    """
    
    data = torch.cat([b[0] for b in batch], dim=0)
    label = torch.stack([b[1] for b in batch], dim=0)
    
    return data, label


def train_dg_collate_fn(batch):
    """
    batch = {augmented_images, image, label, dc, img_name}*batch_size
    """
    new_batch = {}
    batch = [item for sublist in batch for item in sublist]
    
    new_batch['image'] = torch.stack([b['image'] for b in batch], dim=0)
    new_batch['label'] = torch.stack([b['label'] for b in batch], dim=0)
    new_batch['dc'] = torch.stack([b['dc'] for b in batch], dim=0)
    new_batch['img_name'] = [b['img_name'] for b in batch] 
    
    if 'aug_images' in batch[0].keys():
        new_batch['aug_images'] = torch.cat([b['aug_images'] for b in batch], dim=0)
        new_batch['aug_labels'] = torch.cat([b['aug_labels'] for b in batch], dim=0)
        new_batch['dc'] = torch.cat([b['dc'] for b in batch], dim=0)
    
    return new_batch


def test_dg_collate_fn(batch):
    """
    batch = {augmented_images, image, label, dc, img_name}*batch_size
    """
    new_batch = {}
    
    new_batch['image'] = torch.stack([b['image'] for b in batch], dim=0)
    new_batch['label'] = torch.stack([b['label'] for b in batch], dim=0)
    new_batch['dc'] = torch.stack([b['dc'] for b in batch], dim=0)
    new_batch['img_name'] = [b['img_name'] for b in batch] 

    if 'aug_images' in batch[0].keys():
        new_batch['aug_images'] = torch.cat([b['aug_images'] for b in batch], dim=0)
        new_batch['aug_labels'] = torch.cat([b['aug_labels'] for b in batch], dim=0)
        new_batch['dc'] = torch.cat([b['dc'] for b in batch], dim=0)

    if 'roi' in batch[0].keys():
        new_batch['roi'] = torch.stack([b['roi'] for b in batch], dim=0)
    
    return new_batch


def test_collate_fn(batch):
    """
    batch = [((3,H,W), label)]*batch_size
    """
    
    data = torch.stack([b[0] for b in batch], dim=0)
    label = torch.stack([b[1] for b in batch], dim=0)
    
    return data, label
