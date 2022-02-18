import random

import cv2
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch


random_mirror = True


def ShearX(img, mask, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), \
           mask.transform(mask.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, mask, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), \
           mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), \
           mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), \
           mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))



def TranslateXAbs(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), \
           mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), \
           mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, mask, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v), mask.rotate(v)


def AutoContrast(img, mask, _):
    return PIL.ImageOps.autocontrast(img), mask


def Invert(img, mask, _):
    return PIL.ImageOps.invert(img), mask


def Equalize(img, mask, _):
    return PIL.ImageOps.equalize(img), mask


def Flip(img, mask, _):  # not from the paper
    return PIL.ImageOps.mirror(img), mask


def Solarize(img, mask, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v), mask


def Posterize(img, mask, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v), mask


def Posterize2(img, mask, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v), mask


def Contrast(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v), mask


def Color(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v), mask


def Brightness(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v), mask


def Sharpness(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v), mask


def GammaCorrection(img, mask, gamma=1.0):
    assert 0.5 <= gamma <= 4.5
    img_np = np.asarray(img)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    img_np = cv2.LUT(img_np, table)
    img = PIL.Image.fromarray(img_np)
    return img, mask


def Cutout(img, mask, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img, mask

    v = v * img.size[0]
    return CutoutAbs(img, mask, v)


def CutoutAbs(img, mask, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img, mask
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (127, 127, 127)
    label_color = 0
    # color = (0, 0, 0)
    img = img.copy()
    mask = mask.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    PIL.ImageDraw.Draw(mask).rectangle(xy, label_color)
    return img, mask


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def CutMix(img, mask, img2, mask2, v):
    lam = np.random.beta(1, 1)
    H, W = img.size 
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    img, img2 = np.asarray(img), np.asarray(img2)
    mask, mask2 = np.asarray(mask), np.asarray(mask2)

    img[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]
    mask[bby1:bby2, bbx1:bbx2] = mask2[bby1:bby2, bbx1:bbx2]
    img_mix = PIL.Image.fromarray(img)
    mask_mix = PIL.Image.fromarray(mask).convert('L')

    return img_mix, mask_mix


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list(for_autoaug=False):  # 16 operations and their ranges
    l = [
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l

augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def get_augment(name):
    return augment_dict[name]

def apply_augment(img, mask, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), mask.copy(), level * (high - low) + low)

def apply_cutmix(img, mask, img2, mask2, level):
    augment_fn, low, high = get_augment('CutMix')
    return augment_fn(img.copy(), mask.copy(), img2.copy(), mask2.copy(), level * (high - low) + low)