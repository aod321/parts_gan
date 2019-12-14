import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from skimage.util import random_noise
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt


def resize_img_keep_ratio(img, target_size):
    old_size = img.shape[0:2]
    # ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])

    interpol = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR

    img = cv2.resize(img, dsize=(new_size[1], new_size[0]), interpolation=interpol)
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, parts_labels = sample['image'], sample['parts_labels']
        parts = sample['parts']
        parts_labels = [TF.resize(TF.to_pil_image(parts_labels[x]), self.size, Image.ANTIALIAS)
                        for x in ['eye1', 'eye2', 'nose', 'mouth']
                         ]
        parts = [TF.resize(TF.to_pil_image(parts[r]), self.size, Image.ANTIALIAS)
                 for r in range(len(parts))]

        resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        sample = {'image': resized_image,
                  'parts_labels': parts_labels,
                  'parts': parts
                  }

        return sample


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        parts_labels = sample['parts_labels']
        parts = sample['parts']
        parts_labels = [TF.to_tensor(parts_labels[r])
                        for r in range(len(parts_labels))
                        ]
        parts_labels = torch.cat(parts_labels, dim=0)
        parts = [TF.to_tensor(parts[r])
                 for r in range(len(parts))]
        parts = torch.stack(parts, dim=0)
        return {'image': TF.to_tensor(sample['image']),
                'parts_labels': parts_labels,
                'parts': parts
                }


class Normalize(object):
    """Normalize Tensors.
    """

    def __call__(self, sample):
        """
        Args:
            sample (dict of Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensors of sample: Normalized Tensor sample. Only the images need to be normalized.
        """

        image_tensor, labels_tensor = sample['image'], sample['labels']
        # mean = image_tensor.mean(dim=[1, 2]).tolist()
        # std = image_tensor.std(dim=[1, 2]).tolist()
        mean = [0.369, 0.314, 0.282]
        std = [0.282, 0.251, 0.238]
        inplace = True
        sample = {'image': TF.normalize(image_tensor, mean, std, inplace),
                  'labels': labels_tensor,
                  'orig': sample['orig'],
                  'parts': sample['parts']
                  }

        return sample
