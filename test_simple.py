# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from file import MkdirSimple, Walk
from tqdm import tqdm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='path to output folder of depth images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320", "indemind_data"])

    parser.add_argument('--model_path', type=str,
                        help='name of pretrained model by custom dataset')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    if args.model_name in [
        "mono_640x192",
        "stereo_640x192",
        "mono+stereo_640x192",
        "mono_no_pt_640x192",
        "stereo_no_pt_640x192",
        "mono+stereo_no_pt_640x192",
        "mono_1024x320",
        "stereo_1024x320",
        "mono+stereo_1024x320"]:
        download_model_if_doesnt_exist(args.model_name)
        model_path = os.path.join("models", args.model_name)
    else:
        assert args.model_path != "","custom dataset trained model is not set!"
        model_path = args.model_path
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = Walk(args.image_path, ['jpg', 'png', 'jpeg'])
        root_len = len(args.image_path.rstrip('/'))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    output_directory = args.output_dir
    output_dir_depth =  os.path.join(output_directory, "depth")
    output_dir_disp = os.path.join(output_directory, "disp")
    output_dir_disp_color = os.path.join(output_directory, "disp_color")
    output_dir_concat_color = os.path.join(output_directory, "concat")
    output_dir_parula_color = os.path.join(output_directory, "colormap_parula")


    MkdirSimple(output_dir_depth)
    MkdirSimple(output_dir_disp)
    MkdirSimple(output_dir_disp_color)
    MkdirSimple(output_dir_concat_color)
    MkdirSimple(output_dir_parula_color)

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for image_path in tqdm(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            origin_image = pil.open(image_path).convert('RGB')
            original_width, original_height = origin_image.size
            input_image = origin_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
 
            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(image_path[root_len+1:])[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_dir_depth, "{}.npy".format(output_name))
                MkdirSimple(name_dest_npy)
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_dir_disp, "{}.npy".format(output_name))
                MkdirSimple(name_dest_npy)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving PARULA img
            dis_array = disp_resized.cpu().numpy()[0][0]
            dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
            dis_array = dis_array.astype("uint8")
            import cv2
            showImg = cv2.resize(dis_array,(dis_array.shape[-1],dis_array.shape[0]))
            showImg = cv2.applyColorMap(cv2.convertScaleAbs(showImg,1), cv2.COLORMAP_PARULA)
            origin_image_array = cv2.cvtColor(np.array(origin_image),cv2.COLOR_BGR2RGB)
            showImg = np.hstack([origin_image_array, showImg])
            PARULA_file = os.path.join(output_dir_parula_color, "{}.jpeg".format(output_name))
            cv2.imwrite(PARULA_file,showImg)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            concat_img = np.hstack([origin_image, im])

            # gray = (disp_resized_np * 255).astype(np.uint8)
            # im = pil.fromarray(gray)

            name_dest_im = os.path.join(output_dir_disp_color, "{}.jpeg".format(output_name))
            MkdirSimple(name_dest_im)
            im.save(name_dest_im)

            # concat_img =pil.Image(concat_img)
            # concat_img.save(os.path.join(output_dir_concat_color, "{}.jpeg".format(output_name)))
            import cv2
            concat_img = cv2.cvtColor(concat_img,cv2.COLOR_RGB2BGR)
            concat_file = os.path.join(output_dir_concat_color, "{}.jpeg".format(output_name))
            MkdirSimple(concat_file)
            cv2.imwrite(concat_file, concat_img)


    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
