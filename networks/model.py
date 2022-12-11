import numpy as np
import torch
import torch.nn as nn

import networks


class Monodepth2(nn.Module):
    def __init__(self):
        super(Monodepth2, self).__init__()

        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)

        return outputs[("disp", 0)]