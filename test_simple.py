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
from file import MkdirSimple, Walk
from tqdm import tqdm

import torch
from torchvision import transforms, datasets

import networks
from evaluate_depth import STEREO_SCALE_FACTOR
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--output', type=str,
                        help='path to output folder of depth images', required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument('--bf', type=float, default=14.2, help="baseline length multiply focal length")

    return parser.parse_args()

def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)

    return depth_img_rgb.astype(np.uint8)


def WriteDepth(predict_np, limg, path, name, bf):
    if isinstance(predict_np, torch.Tensor):
        predict_np = torch.squeeze(predict_np).cpu().detach().numpy()
    name = os.path.splitext(name)[0] + ".png"
    output_concat_color = os.path.join(path, "concat_color", name)
    output_concat_gray = os.path.join(path, "concat_gray", name)
    output_gray = os.path.join(path, "gray", name)
    output_depth = os.path.join(path, "depth", name)
    output_color = os.path.join(path, "color", name)
    output_concat_depth = os.path.join(path, "concat_depth", name)
    output_concat = os.path.join(path, "concat", name)
    output_gray_scale = os.path.join(path, "gray_scale", name)

    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_gray)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_gray)
    MkdirSimple(output_depth)
    MkdirSimple(output_color)
    MkdirSimple(output_concat)
    MkdirSimple(output_gray_scale)

    predict_np /= 0.005
    depth_img = bf / predict_np * 100  # to cm

    predict_np_int = predict_np.astype(np.uint8)
    color_img = cv2.applyColorMap(predict_np_int, cv2.COLORMAP_HOT)
    limg_cv = limg # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_rgb = GetDepthImg(depth_img)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)
    cv2.imwrite(output_gray, predict_np)
    cv2.imwrite(output_depth, depth_img_rgb)
    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)
    depth_img *= 300
    depth_img[depth_img > 65535] = 65535
    depth_img[depth_img < 1] = 1


    cv2.imwrite(output_gray_scale, depth_img.astype("uint16"))

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
        paths = [f for f in paths if 'cam0' in f]
        root_len = len(args.image_path.rstrip('/'))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for image_path in tqdm(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            origin_image = cv2.imread(image_path)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
            original_height, original_width, _ = origin_image.shape
            org_input_image = cv2.resize(origin_image, (feed_width, feed_height))
            input_image = transforms.ToTensor()(org_input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
 
            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(image_path[root_len+1:])[0]

            output_display = os.path.join(args.output, "display",  os.path.splitext(output_name)[0] + ".png")
            MkdirSimple(output_display)

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            im.save(output_display)

            WriteDepth(disp, org_input_image, args.output, output_name, args.bf)

    print('-> Done!')

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
