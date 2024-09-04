from __future__ import print_function
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import inspect, re
import os
import collections
import torch.nn.functional as F

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im_old(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[2] < 3:
        image_numpy = np.dstack([image_numpy]*3)
    return image_numpy.astype(imtype)

def tensor2im(image_tensor, imtype=np.uint8):
    img = image_tensor[0].cpu().float().numpy()
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # img = img.squeeze()
    img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(imtype)
    if img.shape[2] < 3:
        img = np.dstack([img]*3)
    return img

def gkern_2d(size=5, sigma=3):
    # Create 2D gaussian kernel
    dirac = np.zeros((size, size))
    dirac[size//2, size//2] = 1
    mask = gaussian_filter(dirac, sigma)
    # Adjust dimensions for torch conv2d
    return np.stack([np.expand_dims(mask, axis=0)] * 3)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    # print(type(image_numpy))
    if type(image_numpy) is np.ndarray:
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)
    else:
        image_numpy.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 0, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)

zero_pad = 256 * 3 - len(palette) - 3
for i in range(zero_pad):
    palette.append(0)
for j in range(3):
    palette.append(255)

def colorize_mask(pred_tensor):
    if len(pred_tensor.shape) == 4:
        pred_tensor_resize = F.interpolate(pred_tensor, size=[256, 256], mode='bilinear', align_corners=True)
        # pred_tensor_resize = pred_tensor
        # print(pred_tensor.shape)
        sm = torch.nn.Softmax(dim = 1)
        pred_sm = sm(pred_tensor_resize)
        pred_sm = pred_sm.cpu().data.numpy()
        pred_sm = pred_sm.transpose(0,2,3,1)
        pred_sm = np.asarray(np.argmax(pred_sm, axis=3), dtype=np.uint8)
        mask = pred_sm[0, :, :]
    else:
        mask_ori = pred_tensor[0].detach().cpu().numpy().astype(np.uint8)
        # print(mask_ori.shape)
        # print(Image.fromarray(mask_ori))
        mask = np.asarray(Image.fromarray(mask_ori).resize((256, 256), Image.NEAREST), dtype=np.uint8)
        # mask = np.asarray(mask_ori, dtype=np.uint8)
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
