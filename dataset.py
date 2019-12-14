from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
import jpeg4py as jpeg


class FaceImageDataset(Dataset):
    # HelenDataset
    def __len__(self):
        return len(self.name_list)

    def __init__(self, txt_file, img_root_dir, part_root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(img_root_dir, txt_file), dtype="str", delimiter=',')
        self.img_root_dir = img_root_dir
        self.part_root_dir = part_root_dir
        self.transform = transform
        self.names = ['eye1', 'eye2', 'nose', 'mouth']
        self.label_id = {'eye1': [2, 4],
                         'eye2': [3, 5],
                         'nose': [6],
                         'mouth': [7, 8, 9]
                         }

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.img_root_dir, 'images',
                                img_name + '.jpg')
        image = jpeg.JPEG(img_path).decode()  # [1, H, W]
        part_path = [os.path.join(self.part_root_dir, '%s' % x, 'images',
                                  img_name + '.jpg')
                     for x in self.names]
        plabels_path = {x: [os.path.join(self.part_root_dir, '%s' % x,
                                         'labels', img_name,
                                         img_name + "_lbl%.2d.png" % i)
                            for i in self.label_id[x]]
                        for x in self.names}

        parts_image = [io.imread(part_path[i])
                       for i in range(4)]

        plabels = {x: np.array([io.imread(plabels_path[x][i])
                                for i in range(len(self.label_id[x]))
                                ])
                   for x in self.names
                   }
        for x in self.names:
            bg = 255 - np.sum(plabels[x], axis=0, keepdims=True)  # [1, 64, 64]
            plabels[x] = np.uint8(np.concatenate([bg, plabels[x]], axis=0))  # [L + 1, 64, 64]

        sample = {'image': image, 'parts': parts_image, 'parts_labels': plabels}

        if self.transform:
            sample = self.transform(sample)
        return sample
