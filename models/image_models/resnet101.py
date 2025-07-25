import torch
import torch.nn as nn

from typing import Any

import torchxrayvision as xrv


class ImageEncoderResNet101(nn.Module):  # using 101-elastic ResNet 101
    def __init__(self, configs: dict[str, Any]):
        super(ImageEncoderResNet101, self).__init__()
        self.configs = configs
        # specify weights instead of pretrained=True
        model = xrv.autoencoders.ResNetAE(
            weights="101-elastic",
        )
        modules = list(nn.Sequential(*list(model.children())).children())[
            :7
        ]  # use the first 7 layers
        modules.append(nn.Conv2d(1024, 2048, kernel_size=1))
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        out = self.model(x)

        out = torch.flatten(out, start_dim=2).transpose(1, 2).contiguous()

        vis_pe = torch.arange(out.size(1), dtype=torch.long, device=out.device)
        vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), out.size(1))

        random_sampling = torch.randperm(out.size(1))[
            : self.configs["num_image_embeds"]
        ]

        random_sampling, _ = torch.sort(random_sampling)

        random_sample = out[:, random_sampling]
        random_position = vis_pe[:, random_sampling]

        return random_sample, random_position
