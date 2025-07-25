# -*- coding: utf-8 -*-
from click import Option
import torch
from torch import nn

from typing import Optional

from einops import rearrange

from transformers import AutoTokenizer
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput

from models.image_models.resnet101 import ImageEncoderResNet101
from models.image_models.chexnet121 import ImageEncoderCheXNet121
from models.image_models.bertViT import Img_patch_embedding
from models.image_models.ViT import ImageEncoderViT, ImageBertEmbeddingsViT


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):

        attn_output, _ = self.attn(query, key, value)

        query = self.norm(query + self.dropout(attn_output))
        return query


class Img_patch_embedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=3):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "image dimensions must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

    def forward(self, img, mask=None):
        p = self.patch_size
        out = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        out = self.patch_to_embedding(out)
        return out


class ImageBertEmbeddings(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.img_embeddings = nn.Linear(2048, 768)
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


class MedViLLEncoder(BertPreTrainedModel):
    def __init__(self, model_config, configs):
        super().__init__(model_config)
        self.configs = configs

        bert = BertModel(model_config)
        bert.attn_implementation = "eager"

        self.txt_embeddings = bert.embeddings

        self.img_embeddings = ImageBertEmbeddings(self.txt_embeddings)

        if configs["img_encoder"] == "bertViT":
            raise NotImplementedError(
                "BertViT is not implemented in this version. Please use a different image encoder."
            )
            img_size = configs["img_size"]
            patch_sz = 16
            self.img_encoder = Img_patch_embedding(
                image_size=img_size, patch_size=patch_sz, dim=2048
            )
        elif configs["img_encoder"] == "resnet101_101-elastic":
            self.img_encoder = ImageEncoderResNet101(configs)

        elif configs["img_encoder"] == "chexnet121":
            self.img_encoder = ImageEncoderCheXNet121(configs)

        elif configs["img_encoder"] == "ViT":
            self.img_encoder = ImageEncoderViT(configs)
            self.img_embeddings = ImageBertEmbeddingsViT(self.txt_embeddings)
            self.class_transform_layer = nn.Linear(768, 768)

        self.encoder = bert.encoder

        self.train_with_label = configs.get("train_with_label", False)
        if self.train_with_label:
            num_cross_layers = 3

            hidden_size = model_config.hidden_size
            num_attention_heads = model_config.num_attention_heads
            dropout_prob = model_config.attention_probs_dropout_prob

            self.cross_attention_layers = nn.ModuleList(
                [
                    CrossAttentionLayer(hidden_size, num_attention_heads, dropout_prob)
                    for _ in range(num_cross_layers)
                ]
            )

    def get_extended_attn_mask(self, attn_mask):
        if attn_mask.dim() == 2:
            extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        elif attn_mask.dim() == 3:
            extended_attn_mask = attn_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)
        extended_attn_mask = (1.0 - extended_attn_mask) * -10000.0
        return extended_attn_mask

    def forward(self, cls_tok, attn_mask, input_img, input_text, segment, sep_tok):
        extended_attn_mask = self.get_extended_attn_mask(attn_mask)

        batch_size = input_img.shape[0]

        img_tok = torch.zeros(
            batch_size,
            self.configs["num_image_embeds"],
            dtype=torch.long,
            device=input_img.device,
        )
        cls_segment = torch.LongTensor(input_text.size(0), 1).fill_(0).cuda()

        if self.configs["img_encoder"] != "ViT":
            cls_out = self.txt_embeddings(cls_tok, cls_segment)
            sep_out = self.txt_embeddings(sep_tok, cls_segment)
            img, position = self.img_encoder(input_img)
            img_embed_out = self.img_embeddings(
                img, position, img_tok
            )  # bsz, num_img_embeds, hsz
            txt_embed_out = self.txt_embeddings(input_text, segment)

            if self.train_with_label:
                for layer in self.cross_attention_layers:
                    img_embed_out = layer(
                        query=img_embed_out, key=txt_embed_out, value=txt_embed_out
                    )

            encoder_input = torch.cat(
                [cls_out, img_embed_out, sep_out, txt_embed_out], 1
            )  # B x (TXT + IMG) x HID
        else:
            cls_out, patch_img = self.img_encoder(input_img)
            cls_transform = self.class_transform_layer(cls_out)
            sep_out = self.txt_embeddings(sep_tok, cls_segment)
            position_ids = torch.arange(256, dtype=torch.long, device=input_img.device)
            img_pos = position_ids.unsqueeze(0).expand(batch_size, 256)
            img_embed_out = self.img_embeddings(patch_img, img_pos, img_tok)
            txt_embed_out = self.txt_embeddings(input_text, segment)

            if self.train_with_label:
                for layer in self.cross_attention_layers:
                    img_embed_out = layer(
                        query=img_embed_out, key=txt_embed_out, value=txt_embed_out
                    )

            encoder_input = torch.cat(
                [cls_transform, img_embed_out, sep_out, txt_embed_out], 1
            )

        encoded_layers = self.encoder(
            encoder_input,
            extended_attn_mask,
            output_hidden_states=False,
            output_attentions=True,
        )

        return encoded_layers[0]


class ClinicalT5Decoder(nn.Module):
    """
    Decoder-only wrapper for a pre-trained ClinicalT5 (T5ForConditionalGeneration).
    """

    def __init__(self, pretrained_model_name_or_path: str = ""):
        super().__init__()
        self.config = T5Config.from_pretrained(pretrained_model_name_or_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=self.config,
        )

        # freeze the encoder parameters
        for name, param in self.t5.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False

    def forward(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        hidden_state: dict = {},
        labels: Optional[torch.LongTensor] = None,
    ):
        # ensure all inputs are on the same device
        if decoder_input_ids is not None:
            device = decoder_input_ids.device
        elif labels is not None:
            device = labels.device
        else:
            raise ValueError(
                "Either decoder_input_ids or labels must be provided to determine device."
            )

        # get encoder outputs
        encoder_outputs = hidden_state["encoder_outputs"]

        # build encoder attention mask
        encoder_attention_mask = torch.ones(
            encoder_outputs.last_hidden_state.shape[:-1],
            dtype=torch.long,
            device=device,
        )

        decoder_outputs = self.t5(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
            past_key_values=None,
        )

        return decoder_outputs

    def generate(self, *args, **kwargs):
        return self.t5.generate(*args, **kwargs)

    @staticmethod
    def init_state(encoder_hidden_states):
        return {
            "encoder_outputs": BaseModelOutput(last_hidden_state=encoder_hidden_states)
        }


class MedViLLT5ForReportGeneration(nn.Module):
    def __init__(self, encoder_model_path, decoder_model_path, configs):
        super().__init__()

        # initialize the MedViLLLEncoder and ClinicalT5Decoder
        encoder_config = BertConfig.from_pretrained(encoder_model_path)

        # load encoder from checkpoint
        self.encoder: MedViLLEncoder = MedViLLEncoder(encoder_config, configs)

        encoder_state_dict = torch.load(
            f"{encoder_model_path}/pytorch_model.bin", map_location="cuda"
        )
        encoder_new_state = {}
        for old_key, tensor in encoder_state_dict.items():
            new_key = old_key.replace("enc.", "").replace("mlm.", "cls.")
            encoder_new_state[new_key] = tensor

        filtered_state = {
            k: v
            for k, v in encoder_new_state.items()
            if not k.startswith("img_encoder")
        }

        self.encoder.load_state_dict(filtered_state, strict=False)

        self.decoder: ClinicalT5Decoder = ClinicalT5Decoder(decoder_model_path)
        # initialize the T5 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_path)

    def forward(
        self,
        # Encoder inputs
        cls_tok,
        attn_mask,
        input_img,
        input_text,
        segment,
        sep_tok,
        # Decoder inputs
        decoder_input_ids=None,
        labels=None,
        return_encoder_output=False,
    ):

        encoder_outputs = self.encoder(
            cls_tok, attn_mask, input_img, input_text, segment, sep_tok
        )

        decoder_state = self.decoder.init_state(encoder_outputs)

        # build mask for decoder input ids
        pad_id = self.tokenizer.pad_token_id
        decoder_attention_mask = (decoder_input_ids != pad_id).long()
        decoder_attention_mask[:, 0] = 1  # ensure the first token [PAD] is attended to

        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            hidden_state=decoder_state,
            labels=labels,
        )

        if return_encoder_output:
            return decoder_outputs, encoder_outputs
        else:
            return decoder_outputs


if __name__ == "__main__":
    pass
