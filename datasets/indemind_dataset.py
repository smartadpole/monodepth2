from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
import torch

class IndemindDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(IndemindDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.23, 0, 0.86, 0],
        #                    [0, 0.23, 0.53, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        self.K = np.array([[0.44, 0, 0.53, 0],
                           [0, 0.71, 0.53, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.check_image()

    def check_image(self):
        from tqdm import tqdm
        print("check bad image")
        with open('{}_bad_images.txt'.format("train" if self.is_train else "val"), 'w') as file:
            for f in tqdm(self.filenames):
                image_group = {}
                folder, image_group[0], image_group[-1], image_group[1], side = f.split()
                for key, val in image_group.items():
                    try:
                        self.loader(self.get_image_path(folder, val, 'l'))
                        self.loader(self.get_image_path(folder.replace('cam0', 'cam1'), val, 'r'))
                    except:
                        file.write(f+'\n')
                        break

    def check_depth(self):
        return True

    def get_image_path(self, folder, file_name, side):
        if side == 'l':
            folder = folder.replace('cam1', 'cam0')
        else:
            folder = folder.replace('cam0', 'cam1')

        image_path = os.path.join(self.data_path, folder, file_name)
        # print(folder, " ", file_name, " ", side, " ", image_path)
        return image_path

    def get_color(self, folder, file_name, side, do_flip):
        color = self.loader(self.get_image_path(folder, file_name, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        folder = folder.replace('/cam0/', '/cam1/') if side == 'r' else folder
        file = self.get_image_path(folder, frame_index, side)
        depth = None
        if os.path.exists(file):
            depth = self.loader(file)
            depth = depth.resize((self.width, self.height))

        return depth

    def get_images(self, index, do_flip):
        do_flip = False
        inputs = {}
        image_group = {}

        folder, image_group[0], image_group[-1], image_group[1], side = self.filenames[index].split()

        # print('{} item--------------------'.format(index))
        # print(self.filenames[index])
        for i in self.frame_idxs:
            # print("id: ", i)
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder.replace('cam0', 'cam1'), image_group[0], other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, image_group[i], side, do_flip)

        if self.load_depth:
            depth_gt = self.get_depth(folder, image_group[0], side, do_flip)
            if depth_gt is not None:
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)[:, :, :, 0]
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs, side, do_flip
