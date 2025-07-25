import json
import math
import tarfile

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from random import randint
from random import random as rand

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.loader_utils import Pipeline


class Img2txtDataset(Dataset):
    def __init__(
        self,
        configs,
        bert_tokenizer,
        t5_tokenizer,
        data_set_path=None,
        img_archive=None,
    ):
        super().__init__()
        assert data_set_path is not None
        self.configs = configs
        self.max_seq_len = configs["max_seq_len"]
        self.bert_tokenizer = bert_tokenizer
        self.t5_tokenizer = t5_tokenizer

        # self.tar = tarfile.open(img_archive, mode='r:gz')
        self.tar = None

        img_dat = [json.loads(l) for l in open(data_set_path, "r", encoding="utf-8")]

        self.ex_list = [(d["img"], d["text"]) for d in img_dat]

        # preprocess pipeline
        self.proc = PreprocessSeq2seqGen(
            configs=configs,
            bert_tokenizer=bert_tokenizer,
            t5_tokenizer=t5_tokenizer,
            img_archive=img_archive,
            root_prefix="data/preprocessed/mimic/",
        )

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        return self.proc(self.ex_list[idx])


def truncate_tokens_pair(
    tokens_a,
    tokens_b,
    max_seq_len,
    max_len_a=0,
    max_len_b=0,
    trunc_seg=None,
    always_truncate_tail=False,
):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_seq_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == "a":
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class PreprocessSeq2seqGen(Pipeline):
    def __init__(
        self,
        configs,
        bert_tokenizer,
        t5_tokenizer,
        img_archive: tarfile.TarFile,
        root_prefix: str,
    ):
        super().__init__()
        self.configs = configs
        self.bert_tokenizer = bert_tokenizer
        self.t5_tokenizer = t5_tokenizer

        self.img_archive = img_archive
        self.root_prefix = root_prefix.rstrip("/") + "/"

        self.max_seq_len = configs["max_seq_len"]  # maximum length of tokens
        self.len_vis_input = configs["num_image_embeds"]  # number of visual tokens
        self.seq_len = configs["seq_len"]  # maximum length of text tokens

        self.img_size = configs["img_size"]  # image size for resizing

        # image preprocessing
        self.gray3 = transforms.Grayscale(num_output_channels=3)
        self.resize = transforms.Resize(self.img_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.492, 0.493, 0.494], std=[0.293, 0.293, 0.293]
        )

    def __call__(self, instance):
        # img_path, text_b = instance
        full_img_path, text_b = instance
        member_path = full_img_path.replace(self.root_prefix, "", 1)

        tokens_b_bert = self.bert_tokenizer.tokenize(text_b)

        if self.tar is None:
            self.tar = tarfile.open(self.img_archive, mode="r:gz")

        member = self.tar.getmember(member_path)
        f = self.tar.extractfile(member)
        img = Image.open(BytesIO(f.read())).convert("RGB")

        if len(tokens_b_bert) > self.max_seq_len:
            tokens_b_bert = tokens_b_bert[-self.max_seq_len :]

        # build text tokens input_ids
        input_ids = len(tokens_b_bert) * ["[PAD]"] + ["[SEP]"]  # len(seq_len) + [SEP]

        if len(input_ids) < self.seq_len + 1:  # +1 for [SEP]
            pad_len = self.seq_len - len(input_ids) + 1
            input_ids.extend(["[PAD]"] * pad_len)
        else:
            input_ids = input_ids[: self.seq_len + 1]

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # build segment_ids [ seq_len + [SEP] ]
        segment_ids = [1 for _ in range(len(input_ids))]

        if len(segment_ids) < self.seq_len + 1:  # +1 for [SEP]
            segment_pad = self.seq_len + 1 - len(segment_ids)
            segment_ids.extend([0] * segment_pad)

        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        total = 1 + self.len_vis_input + 1 + self.seq_len + 1
        # build encoder input mask
        attn_mask = torch.zeros(total, total, dtype=torch.long)
        attn_mask[: self.len_vis_input + 2, :] = 1

        # image preprocessing
        # img = Image.open(img_path)
        img = self.gray3(img)
        img = self.resize(img)
        img = self.to_tensor(img)
        img = self.normalize(img)

        if isinstance(text_b, str) and text_b:
            t_b = self.t5_tokenizer.encode(
                text_b,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
            )
        else:
            t_b = []
        if len(t_b) < 512:
            t_b.extend([self.t5_tokenizer.pad_token_id] * (512 - len(t_b)))

        t_b = torch.tensor(t_b, dtype=torch.long)

        cls_tok = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"]))
        sep_tok = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(["[SEP]"]))

        return cls_tok, sep_tok, input_ids, segment_ids, attn_mask, img, t_b


if __name__ == "__main__":
    pass
