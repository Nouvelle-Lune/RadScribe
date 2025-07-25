import torch
import torch.nn as nn
from torchvision.models import densenet121


class CheXNet(nn.Module):
    def __init__(self, out_size=14):
        super().__init__()
        self.model = densenet121(pretrained=False)
        self.model.classifier = nn.Linear(
            in_features=self.model.classifier.in_features, out_features=out_size
        )

    def forward(self, x):
        return self.model(x)


class ImageEncoderCheXNet121(nn.Module):
    def __init__(self, configs):
        super(ImageEncoderCheXNet121, self).__init__()
        self.configs = configs
        # specify weights instead of pretrained=True

        model: CheXNet = torch.load(
            "models/pre_model_weights/CheXnet/" "CheXnet_rebased_full_model.pt",
            map_location="cuda",
            weights_only=False,
        )

        modules = list(list(model.children())[0].children())[0]
        modules.append(nn.Conv2d(1024, 2048, kernel_size=1))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
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
