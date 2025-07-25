import tarfile
from random import randint

import torch
from torch.utils.data import get_worker_info


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.proc.tar = tarfile.open(dataset.proc.img_archive, mode="r:gz")


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


class Pipeline:
    """Pre-process Pipeline Class : callable"""

    def __init__(self):
        super().__init__()
        self.mask_same_word = None
        self.skipgram_prb = None
        self.skipgram_size = None

    def __call__(self, instance):
        raise NotImplementedError


if __name__ == "__main__":
    pass
