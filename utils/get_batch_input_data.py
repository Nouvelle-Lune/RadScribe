import torch


def get_batch_input_data(batch, device, tokenizer):
    # move data to device
    moved = [x.to(device) if hasattr(x, "to") else x for x in batch]

    cls_tok, sep_tok, input_ids, segment_ids, attn_mask, img, target_text = moved

    # build decoder_input_ids
    decoder_input_ids = torch.cat(
        [
            torch.full(
                (target_text.shape[0], 1),
                tokenizer.pad_token_id,
                dtype=target_text.dtype,
                device=target_text.device,
            ),
            target_text[:, :-1],
        ],
        dim=1,
    )

    labels = target_text.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    batch_data = {
        "cls_tok": cls_tok,
        "sep_tok": sep_tok,
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "attn_mask": attn_mask,
        "img": img,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }

    return batch_data


if __name__ == "__main__":
    pass
