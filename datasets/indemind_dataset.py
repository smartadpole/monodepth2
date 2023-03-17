from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from imageio.v2 import imread
import threading
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
import torch
CONFIG_FILE = ['config.yaml', 'MODULE.yaml', 'MoudleParam.yaml']

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

        # # data from jingliannwen
        # self.K = np.array([[0.44, 0, 0.5, 0],
        #                    [0, 0.7, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        # self.check_image()
        self.K_dict = {}

    def check_image(self):
        from tqdm import tqdm
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
            folder = folder.replace('/cam1', '/cam0')
            folder = folder.replace('/right', '/left')
        else:
            folder = folder.replace('/cam0', '/cam1')
            folder = folder.replace('/left', '/right')

        image_path = os.path.join(self.data_path, folder, file_name)
        # print(folder, " ", file_name, " ", side, " ", image_path)
        return image_path

    def get_color(self, folder, file_name, side, do_flip):
        color = self.loader(self.get_image_path(folder, file_name, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        folder = folder.replace('/cam0', '/cam1') if side == 'r' else folder
        folder = folder.replace('/left', '/right') if side == 'r' else folder
        file = self.get_image_path(folder, frame_index, side)
        depth = None
        if os.path.exists(file):
            depth = self.loader(file)
            depth = depth.resize((self.width, self.height))

        return depth

    def GetConfigFile(self, path):
        for file_name in CONFIG_FILE:
            file = os.path.join(path, file_name)
            if os.path.exists(file):
                break
        if not os.path.exists(file):
            assert 0, "camera config {} not exist! exit".format(file)
        return file

    def set_by_config_yaml(self, folder):
        config_file = "/" + os.path.join(self.data_path, *(folder.split('/')[:-2]))
        config_file = self.GetConfigFile(config_file)

        if config_file in self.K_dict:
            self.K = self.K_dict[config_file]
        else:
            with open(config_file, 'r') as f:
                lines = f.readlines()
                width = 640
                height = 400
                for i in range(len(lines)):
                    if 'image_dimension' in lines[i]:
                        image_dimension = lines[i] + lines[i + 1]
                        image_dimension = image_dimension.split(',')

                        for j in range(len(image_dimension)):

                            if 'image_dimension' in image_dimension[j]:
                                width = image_dimension[j].split('[')[1]
                                height = image_dimension[j + 1].split(']')[0]
                    elif "width_l" in lines[i]:
                        width = float(lines[i].split(' ')[1])
                    elif "height_l" in lines[i]:
                        height = float(lines[i].split(' ')[1])
                    elif "Pl" in lines[i]:
                        config_Pl_x = lines[i + 4]
                        Pl_00 = config_Pl_x.split(' ')[5]
                        Pl_02 = config_Pl_x.split(' ')[7]
                        config_Pl_y = lines[i + 5]

                        Pl_11 = config_Pl_y.split(' ')[7]
                        Pl_12 = config_Pl_y.split(' ')[8]
                        self.K[0][0] = float(Pl_00.split(',')[0]) / float(width)
                        self.K[0][2] = float(Pl_02.split(',')[0]) / float(width)
                        self.K[1][1] = float(Pl_11.split(',')[0]) / float(height)
                        self.K[1][2] = float(Pl_12.split(',')[0]) / float(height)
                        self.K_dict[config_file] = self.K
                        break
    def get_images(self, index, do_flip):
        do_flip = False
        inputs = {}
        image_group = {}
        if len(self.filenames[index].split()) == 5:
            folder, image_group[0], image_group[-1], image_group[1], side = self.filenames[index].split()
        elif len(self.filenames[index].split()) == 4:
            image_group[-1], image_group[0], image_group[1], side = self.filenames[index].split()
            folder = os.path.join(*(image_group[0].split('/')[:-1]))
            image_group_temp = image_group.copy()
            image_group[-1] = image_group[-1].split('/')[-1]
            image_group[0] = image_group[0].split('/')[-1]
            image_group[1] = image_group[1].split('/')[-1]

        else:
            assert 0, "file list error, every line's length is not 4 or 5"

        self.set_by_config_yaml(folder)
        # print('{} item--------------------'.format(index))
        # print(self.filenames[index])
        for i in self.frame_idxs:
            # print("id: ", i)
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder.replace('/cam0', '/cam1'), image_group[0], other_side, do_flip)
                inputs[("color", i, -1)] = self.get_color(folder.replace('/left', '/right'), image_group[0], other_side, do_flip)

            else:
                if len(self.filenames[index].split()) == 5:
                    inputs[("color", i, -1)] = self.get_color(folder, image_group[i], side, do_flip)
                elif len(self.filenames[index].split()) == 4:
                    inputs[("color", i, -1)] = self.get_color(os.path.join(*(image_group_temp[i].split('/')[:-1])),
                                                              image_group[i], side, do_flip)

        if self.depth_dir != "":
            if side == 'l':
                file = os.path.join(self.depth_dir, folder, image_group[0]).replace(".jpg", ".png")
                if os.path.exists(file):
                    tgt_pseudo_depth = imread(file).astype(np.float32)
                    tgt_pseudo_depth.resize(self.height, self.width)
                    if tgt_pseudo_depth.ndim < 3:  # depth
                        tgt_pseudo_depth = np.expand_dims(tgt_pseudo_depth, axis=0)
                    inputs["tgt_pseudo_depth"] = torch.from_numpy(tgt_pseudo_depth).float()
                else:
                    assert 0, "depth image: {} used for dc_depth_pl is not exist!".format(file)
            else:
                inputs["tgt_pseudo_depth"] = torch.from_numpy(np.zeros((1, self.height, self.width))).float()

        if self.load_depth:
            depth_gt = self.get_depth(folder, image_group[0], side, do_flip)
            if depth_gt is not None:
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)[:, :, :, 0]
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs, side, do_flip
