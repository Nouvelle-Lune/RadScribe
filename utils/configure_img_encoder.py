import torch


# For unfreezing specific layers of image encoders
def configure_img_encoder(encoder: torch.nn.Module, configs: dict):
    # encoder：MedViLLEncoder

    name = configs.get("img_encoder", None)
    if name == "resnet101_101-elastic":
        # train layer2 and layer3 of the ResNet101
        seq = list(encoder.img_encoder.model.children())
        for idx in (6, 7, 8):
            for p in seq[idx].parameters():
                p.requires_grad = True

    elif name == "chexnet121":
        # train layer3 and layer4 of the CheXNet121
        for c in list(encoder.img_encoder.model.children())[-6:]:
            for p in c.parameters():
                p.requires_grad = True

    # freeze the first 8 layers of the ViT encoder
    elif name == "ViT":
        for param in encoder.img_encoder.parameters():
            param.requires_grad = True

        for idx in range(8):
            for param in encoder.img_encoder.model.encoder.layer[idx].parameters():
                param.requires_grad = False

        # freeze Patch Embedding
        for param in encoder.img_encoder.model.embeddings.patch_embeddings.parameters():
            param.requires_grad = False

        for param in encoder.class_transform_layer.parameters():
            param.requires_grad = True

        for param in encoder.img_embeddings.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown img_encoder type: {name}")

    if hasattr(encoder, "cross_attention_layers"):
        for p in encoder.cross_attention_layers.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    pass
