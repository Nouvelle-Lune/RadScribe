import torch
from torch import nn
from transformers import ViTModel


class ImageBertEmbeddingsViT(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.img_embeddings = nn.Identity()
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(0.1)
        self.position_embeddings = embeddings.position_embeddings

    def forward(self, input_imgs, img_pos, token_type_ids):

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(img_pos)

        embeddings = imgs_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ImageEncoderViT(nn.Module):
    def __init__(self, configs):
        super(ImageEncoderViT, self).__init__()
        self.configs = configs

        self.model_name = "google/vit-base-patch16-224-in21k"
        self.model = ViTModel.from_pretrained(
            self.model_name, ignore_mismatched_sizes=True
        )

        self.linear = nn.Linear(1024, 256)

    def forward(self, x):
        outputs = self.model(x, interpolate_pos_encoding=True)
        patch_img = outputs.last_hidden_state[:, 1:, :]
        patch_img = patch_img.transpose(1, 2)
        patch_img = self.linear(patch_img)
        patch_img = patch_img.transpose(1, 2)
        cls = outputs.last_hidden_state[:, 0].unsqueeze(1)

        return cls, patch_img


if __name__ == "__main__":
    pass
