# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import wandb
import yaml
from colorama import init, Fore
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from models.model import MedViLLT5ForReportGeneration
from utils.dataloader import Img2txtDataset
from utils.dataloaderwithlabel import Img2txtDatasetWithLabel
from utils.configure_img_encoder import configure_img_encoder
from utils.early_stopping import EarlyStopping
from utils.evaluator import evaluator
from utils.generator_scheduler import scheduler
from utils.get_batch_input_data import get_batch_input_data
from utils.loader_utils import worker_init_fn

from models.image_models.chexnet121 import ImageEncoderCheXNet121, CheXNet

init(autoreset=True)


def get_frozen_layers(model):
    return {name for name, p in model.named_parameters() if not p.requires_grad}


def get_params_summary(model, wandb_run):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tqdm.write(Fore.CYAN + f">> 🔍 Trainable parameters: {trainable_params:,}")
    tqdm.write(
        Fore.CYAN + f">> 🔍 Percentage: {trainable_params / total_params * 100:.2f}%"
    )

    wandb_run.summary["Trainable parameters after freeze encoder"] = trainable_params
    wandb_run.summary["Percentage trainable after freeze encoder"] = (
        f"{trainable_params / total_params * 100: .2f}%"
    )


def get_dataloaders(
    configs, bert_tokenizer, t5_tokenizer, dataset_path, img_archive=None
):
    with_label = configs.get("train_with_label", False)
    if with_label:
        dataset = Img2txtDatasetWithLabel(
            configs=configs,
            bert_tokenizer=bert_tokenizer,
            t5_tokenizer=t5_tokenizer,
            data_set_path=dataset_path,
            img_archive=img_archive,
        )
    else:
        dataset = Img2txtDataset(
            configs=configs,
            bert_tokenizer=bert_tokenizer,
            t5_tokenizer=t5_tokenizer,
            data_set_path=dataset_path,
            img_archive=img_archive,
        )

    tqdm.write(Fore.BLUE + f">> ✅ From {dataset_path} load {len(dataset)} samples <<")

    dataloader = DataLoader(
        dataset,
        batch_size=configs["batch_size"],
        num_workers=configs["num_workers"],
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    return dataloader


def get_unique_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path
    idx = 1
    while True:
        candidate = f"{base_path}_{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


class MedViLLT5Trainer:
    def __init__(
        self,
        configs,
        encoder_model_path,
        decoder_model_path,
        output_dir,
        need_evaluate=True,
        task_name=None,
        train_dataset_name=None,
        valid_dataset_name=None,
        log_step=1000,
        resume_id="",
        enabel_early_stopping=False,
    ):

        self.configs = configs

        self.train_dataset_path = (
            self.configs["train_dataset"]
            if train_dataset_name is None
            else self.configs[train_dataset_name]
        )
        self.valid_dataset_path = (
            self.configs["valid_dataset"]
            if valid_dataset_name is None
            else self.configs[valid_dataset_name]
        )

        self.task_name = task_name
        self.log_step = log_step

        self.model = MedViLLT5ForReportGeneration(
            encoder_model_path=encoder_model_path,
            decoder_model_path=decoder_model_path,
            configs=self.configs,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        tqdm.write(Fore.CYAN + f">> 🔍 Total parameters: {total_params:,} <<")
        tqdm.write(Fore.CYAN + f">> 🔍 Trainable parameters: {trainable_params:,} <<")
        tqdm.write(
            Fore.CYAN
            + f">> 🔍 Percentage trainable: {trainable_params / total_params * 100:.2f}% <<"
        )

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")

        self.train_loader = get_dataloaders(
            self.configs,
            self.bert_tokenizer,
            self.t5_tokenizer,
            self.train_dataset_path,
            self.configs.get("train_img_archive", None),
        )

        self.need_evaluate = need_evaluate
        self.valid_loader = None
        if self.need_evaluate:
            self.valid_loader = get_dataloaders(
                self.configs,
                self.bert_tokenizer,
                self.t5_tokenizer,
                self.valid_dataset_path,
                self.configs.get("valid_img_archive", None),
            )

            self.evaluator = evaluator(
                self.model, self.valid_loader, self.t5_tokenizer, self.device
            )

        # gradient accumulation steps
        self.grad_accum_steps = self.configs["gradient_accumulation_steps"]

        # initialize learning rates
        self.optimizer_config = self.configs["optimizer_config"]
        self.encoder_init_lr = self.optimizer_config["encoder"]["lr"]
        self.decoder_init_lr = self.optimizer_config["decoder"]["lr"]

        self.encoder_min_lr = self.optimizer_config["encoder"]["min_lr"]
        self.decoder_min_lr = self.optimizer_config["decoder"]["min_lr"]

        # Note: With gradient accumulation, we don't need to scale learning rates
        # The effective batch size is already handled by accumulation
        # Remove the problematic sqrt scaling

        assert (
            self.encoder_min_lr <= self.encoder_init_lr
            and self.decoder_min_lr <= self.decoder_init_lr
        )

        # warmup steps for encoder and decoder
        self.warmup_rate_enc = self.optimizer_config["encoder"]["warmup_rate_enc"]
        self.warmup_rate_dec = self.optimizer_config["decoder"]["warmup_rate_dec"]

        # weight decay for encoder and decoder
        self.encoder_weight_decay = self.optimizer_config["encoder"]["weight_decay"]
        self.decoder_weight_decay = self.optimizer_config["decoder"]["weight_decay"]

        # initialize epoch
        self.num_epochs = self.configs["epochs"]

        # only allocate cuda memory for trainable parameters
        encoder_trainable = [
            p for p in self.model.encoder.parameters() if p.requires_grad
        ]
        decoder_trainable = [
            p for p in self.model.decoder.parameters() if p.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_trainable,
                    "lr": self.encoder_init_lr,
                    "weight_decay": self.encoder_weight_decay,
                },
                {
                    "params": decoder_trainable,
                    "lr": self.decoder_init_lr,
                    "weight_decay": self.decoder_weight_decay,
                },
            ],
            betas=(0.9, 0.999),
        )

        # learning rate scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=[
                self.make_lambda(
                    self.warmup_rate_enc,
                    len(self.train_loader) * self.num_epochs,
                    self.grad_accum_steps,
                    self.encoder_init_lr,
                    self.encoder_min_lr,
                ),  # param_groups[0]
                self.make_lambda(
                    self.warmup_rate_dec,
                    len(self.train_loader) * self.num_epochs,
                    self.grad_accum_steps,
                    self.decoder_init_lr,
                    self.decoder_min_lr,
                ),  # param_groups[1] - Fixed: use decoder_min_lr
            ],
        )

        # frozen layers
        self.frozen_layers = get_frozen_layers(self.model)

        # frozen encoder learning rate
        self.frozen_encoder_lr = self.optimizer.param_groups[0]["lr"]

        # Mixed precision training setup
        self.use_amp = self.configs.get("use_amp", True)
        if self.use_amp:
            self.scaler = GradScaler()
            tqdm.write(Fore.YELLOW + ">> 🚀 Using mixed precision training (FP16)! <<")
        else:
            self.scaler = None
            tqdm.write(Fore.YELLOW + ">> 🚀 Full precision training (FP32) mode <<")

        self.epoch = 0
        self.global_step = 0

        self.es = None
        if enabel_early_stopping:
            self.es = EarlyStopping(patience=3, min_delta=0.001, mode="min")
            tqdm.write(Fore.RED + ">> ⚠️ Early stopping enabled! <<")

        # W&B init
        self.resume_id = resume_id
        self._init_wandb()

        model_log_frq = (
            len(self.train_loader) * self.num_epochs * 0.1
        )  # log every 10% of training
        # -- wandb graph -- #
        self.wandb_run.watch(
            self.model, log="all", log_freq=model_log_frq, log_graph=True
        )

        self.wandb_run.summary["Total params"] = total_params
        self.wandb_run.summary["Trainable params"] = trainable_params
        self.wandb_run.summary["Percentage trainable params"] = (
            f"{trainable_params / total_params * 100: .2f}%"
        )

    def _init_wandb(self):
        base_run_name = f"{self.task_name or 'run'}"
        unique_run_dir = get_unique_path(
            os.path.join(self.output_dir, "wandb", base_run_name)
        )
        os.makedirs(unique_run_dir, exist_ok=True)
        unique_run_name = os.path.basename(unique_run_dir)

        init_kwargs = {
            "project": self.configs.get("project_name"),
            "entity": self.configs.get("entity"),
            "config": self.configs,
            "name": unique_run_name,
            "dir": os.path.dirname(unique_run_dir),
        }
        if self.resume_id:
            init_kwargs["id"] = self.resume_id
            init_kwargs["resume"] = "must"
        self.wandb_run = wandb.init(**init_kwargs)

    def train(self, fine_tune=False):
        torch.cuda.empty_cache()
        if fine_tune:
            tqdm.write(
                Fore.YELLOW
                + ">> ⚡️ Fine-tuning mode enabled, freezing encoder parameters <<"
            )
            self.freeze_encoder()
            # get_params_summary(self.model, self.wandb_run)

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(self.epoch, self.num_epochs):

            epoch_loss = 0
            epoch_steps = 0
            total_nll = 0.0
            total_tokens = 0

            epoch_perplexity = []

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                position=0,
            )

            if epoch >= self.configs["encoder_freeze_epoch"] and fine_tune:
                tqdm.write(Fore.YELLOW + ">> ⚡️ Unfreezing encoder parameters")
                self.unfreeze_encoder()
                get_params_summary(self.model, self.wandb_run)

            current_frozen = get_frozen_layers(self.model)
            unfrozen = self.frozen_layers - current_frozen
            new_frozen = current_frozen - self.frozen_layers

            if unfrozen:
                tqdm.write(
                    Fore.CYAN
                    + f"\n>> ℹ️ The following layer(s) are unfrozen in epoch {epoch}: <<"
                )
                for name in sorted(unfrozen):
                    print("  •", name)
                get_params_summary(self.model, self.wandb_run)

            if new_frozen:
                print(
                    Fore.YELLOW
                    + f"\n>> ⚠️ The following layer(s) are frozen in epoch {epoch}: <<"
                )
                for name in sorted(new_frozen):
                    print("  •", name)
                get_params_summary(self.model, self.wandb_run)

            self.frozen_layers = current_frozen

            for batch in progress_bar:

                batch_data = get_batch_input_data(batch, self.device, self.t5_tokenizer)

                # Mixed precision forward pass
                if self.use_amp:
                    with autocast():
                        outputs, encoder_out = self.model(
                            cls_tok=batch_data["cls_tok"],
                            sep_tok=batch_data["sep_tok"],
                            input_text=batch_data["input_ids"],
                            segment=batch_data["segment_ids"],
                            attn_mask=batch_data["attn_mask"],
                            input_img=batch_data["img"],
                            decoder_input_ids=batch_data["decoder_input_ids"],
                            labels=batch_data["labels"],
                            return_encoder_output=True,
                        )
                        loss = outputs.loss / self.grad_accum_steps
                else:
                    # Standard precision forward pass
                    outputs, encoder_out = self.model(
                        cls_tok=batch_data["cls_tok"],
                        sep_tok=batch_data["sep_tok"],
                        input_text=batch_data["input_ids"],
                        segment=batch_data["segment_ids"],
                        attn_mask=batch_data["attn_mask"],
                        input_img=batch_data["img"],
                        decoder_input_ids=batch_data["decoder_input_ids"],
                        labels=batch_data["labels"],
                        return_encoder_output=True,
                    )
                    loss = outputs.loss / self.grad_accum_steps

                # Mixed precision backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    if self.use_amp:
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        # Step optimizer and scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # self.optimizer.step()
                # self.scheduler.step()

                # epoch_loss += loss.item()
                epoch_loss += loss.item() * self.grad_accum_steps
                avg_loss = epoch_loss / (epoch_steps + 1)  # avoid division by zero
                self.global_step += 1
                epoch_steps += 1

                mask = batch_data["labels"] != -100
                n_tokens = mask.sum().item()
                total_nll += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

                avg_nll = total_nll / total_tokens
                perplexity = np.exp(avg_nll)
                epoch_perplexity.append(perplexity)

                # —— wandb log —— #
                self.wandb_run.log(
                    data={
                        "train/loss_step": loss.item() * self.grad_accum_steps,
                        "train/avg_loss": avg_loss,
                        "train/perplexity": perplexity,
                        "lr/encoder": self.scheduler.get_last_lr()[0],
                        "lr/decoder": self.scheduler.get_last_lr()[1],
                        "step": self.global_step,
                        "epoch": epoch + 1,
                    },
                    step=self.global_step,
                )
                # ———————————————— #

                # update progress bar
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item() * self.grad_accum_steps:.4f}",
                        "Avg Loss": f"{avg_loss:.4f}",
                        "Perplexity": f"{perplexity:.4f}",
                        "lr_encoder": f"{self.scheduler.get_last_lr()[0]:.6f}",
                        "lr_decoder": f"{self.scheduler.get_last_lr()[1]:.6f}",
                    }
                )

                if self.global_step % self.log_step == 0:
                    encoder_hidden = encoder_out[0].unsqueeze(0)
                    encoder_hidden = self.model.decoder.init_state(encoder_hidden)
                    decode_params = scheduler.get_params(self.epoch)
                    decode_params.update(
                        {
                            "pad_token_id": self.t5_tokenizer.pad_token_id,
                            "eos_token_id": self.t5_tokenizer.eos_token_id,
                        }
                    )
                    generated_ids = self.model.decoder.generate(
                        encoder_outputs=encoder_hidden["encoder_outputs"],
                        **decode_params,
                    )
                    tqdm.write(Fore.MAGENTA + "\n" + "=" * 50)
                    cleaned = [
                        l if l >= 0 else self.t5_tokenizer.pad_token_id
                        for l in batch_data["labels"][0].tolist()
                    ]
                    tqdm.write(
                        Fore.MAGENTA
                        + f"Target text: \n{self.t5_tokenizer.decode(cleaned)}\n"
                        f"Generated text: \n{self.t5_tokenizer.decode(generated_ids[0])}"
                    )
                    tqdm.write(Fore.MAGENTA + "=" * 50)

                    if self.need_evaluate:
                        eval_metrics = self.evaluator.evaluate(self.epoch)
                        self.wandb_run.log(
                            data={
                                "eval/CLINICAL_ACCURACY": eval_metrics[
                                    "CLINICAL_ACCURACY"
                                ],
                                "eval/CLINICAL_PRECISION": eval_metrics[
                                    "CLINICAL_PRECISION"
                                ],
                                "eval/CLINICAL_RECALL": eval_metrics["CLINICAL_RECALL"],
                                "eval/CLINICAL_F1": eval_metrics["CLINICAL_F1"],
                                "eval/BLEU-4": eval_metrics["BLEU-4"],
                                "eval/METEOR": eval_metrics["METEOR"],
                                "eval/CIDEr": eval_metrics["CIDEr"],
                                "eval/ROUGE-1": eval_metrics["ROUGE-1"],
                                "eval/ROUGE-2": eval_metrics["ROUGE-2"],
                                "eval/ROUGE-L": eval_metrics["ROUGE-L"],
                                "eval/TOKEN_ACCURACY:": eval_metrics["TOKEN_ACCURACY"],
                                "eval/TOKEN_PRECISION": eval_metrics["TOKEN_PRECISION"],
                                "eval/TOKEN_RECALL": eval_metrics["TOKEN_RECALL"],
                                "eval/TOKEN_F1": eval_metrics["TOKEN_F1"],
                                "step": self.global_step,
                                "epoch": epoch + 1,
                            },
                            step=self.global_step,
                        )

            if (self.global_step + 1) % self.grad_accum_steps != 0:
                if self.use_amp:
                    # For mixed precision, we need to check if there are scaled gradients
                    # Only step if we have accumulated gradients
                    if any(param.grad is not None for param in self.model.parameters()):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    # For full precision, check if there are gradients to step
                    if any(param.grad is not None for param in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # Always step the scheduler if we stepped the optimizer
                if any(param.grad is not None for param in self.model.parameters()):
                    self.scheduler.step()

            self.epoch += 1

            self.wandb_run.log(
                data={
                    "train/loss_epoch": epoch_loss / len(self.train_loader),
                    "epoch": epoch + 1,
                },
                step=self.global_step,
            )

            # save checkpoint at the end of each epoch
            self.save_checkpoint(self.global_step, self.epoch, self.task_name)
            tqdm.write(
                Fore.GREEN
                + f">> 🎯 Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(self.train_loader):.4f} <<"
            )

            try:
                if self.es is not None:
                    self.es(np.mean(epoch_perplexity))
                    if self.es.early_stop:
                        tqdm.write(
                            Fore.RED
                            + f">> ⚠️ Early stopping triggered on epoch {self.epoch}<<"
                        )
                        break
            except Exception as e:
                tqdm.write(Fore.RED + f">> ⚠️ Early stopping error: {e} <<")

    def freeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        configure_img_encoder(self.model.encoder, self.configs)

        self.optimizer.param_groups[0]["lr"] = 0.0
        tqdm.write(
            Fore.YELLOW
            + f">> ⚠️ Encoder frozen, Saved lr = {self.frozen_encoder_lr:.6f}, Current lr = 0 <<"
        )

    def unfreeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        for name, child in self.model.encoder.named_children():
            if name == "img_encoder":
                continue
            for p in child.parameters():
                p.requires_grad = True

        current_lr = self.scheduler.get_last_lr()[0]
        self.optimizer.param_groups[0]["lr"] = current_lr
        tqdm.write(
            Fore.YELLOW + f">> ⚠️ Encoder unfrozen, Current lr = {current_lr:.6f} <<"
        )

    # learning rate scheduler lambda function
    def make_lambda(
        self,
        warmup_rate_rate: float,
        total_s: int,
        grad_accum_steps: int,
        base_lr: float = 0.0001,
        min_lr: float = 0.0,
    ):
        min_rate = min_lr / base_lr
        total_updates = max(1, total_s // grad_accum_steps)
        warmup_updates = max(1, int(warmup_rate_rate * total_updates))
        decay_updates = max(1, total_updates - warmup_updates)

        def lr_lambda(update_idx: int):
            if update_idx < warmup_updates:
                factor = update_idx / warmup_updates
            else:
                factor = (total_updates - update_idx) / decay_updates

            factor = max(min_rate, min(1.0, factor))
            return float(factor)

        return lr_lambda

    def get_current_learning_rates(self):
        encoder_lr = self.optimizer.param_groups[0]["lr"]
        decoder_lr = self.optimizer.param_groups[1]["lr"]
        scheduler_lrs = self.scheduler.get_last_lr()

        return {
            "encoder_actual": encoder_lr,
            "decoder_actual": decoder_lr,
            "encoder_scheduled": scheduler_lrs[0],
            "decoder_scheduled": scheduler_lrs[1],
        }

    def save_checkpoint(self, step, epoch, task_name=None):
        if task_name:
            base_path = os.path.join(
                self.output_dir, f"{task_name}_checkpoint_epoch_{epoch}_step_{step}"
            )
        else:
            base_path = os.path.join(
                self.output_dir, f"checkpoint_epoch_{epoch}_step_{step}"
            )
        filename = base_path + ".pt"
        if os.path.exists(filename):
            idx = 1
            while True:
                new_name = f"{base_path}_{idx}.pt"
                if not os.path.exists(new_name):
                    filename = new_name
                    break
                idx += 1

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": step,
                "epoch": epoch,
                "configs": self.configs,
            },
            filename,
        )
        print(f"Saved checkpoint to {filename} (epoch={epoch}, step={step})")

    def resume_checkpoint(self, checkpoint_path="", initialize=True):
        if initialize:
            self._load_checkpoint(checkpoint_path)
        else:
            epoch, step = self._load_checkpoint(checkpoint_path, initialize=False)  # type: ignore
            # ensure we do not exceed configured epochs
            assert epoch + self.configs["encoder_freeze_epoch"] <= self.num_epochs

            if epoch > self.num_epochs:
                raise ValueError(
                    f"Checkpoint epoch {epoch} exceeds configured epochs {self.num_epochs}."
                )
            self.epoch = epoch
            self.global_step = step
        configure_img_encoder(self.model.encoder, self.configs)

    def _load_checkpoint(self, checkpoint_path, initialize=True):
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])

        if initialize:
            tqdm.write(Fore.BLUE + f" >> ✅ Loaded checkpoint '{checkpoint_path}' << ")
            return None
        else:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            step = ckpt.get("step", None)
            epoch = ckpt.get("epoch", None)

            tqdm.write(
                Fore.BLUE
                + f" >> ✅ Loaded checkpoint '{checkpoint_path}' (epoch={epoch}, step={step}) << "
            )
            return epoch, step


def parse_args():
    parser = argparse.ArgumentParser(description="Run pretraining or fine‑tuning")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to the YAML config file (default: configs/pretrain.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    # Load YAML configuration
    with open(config_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    encoder_model_path = configs["encoder_model_path"]
    decoder_model_path = configs["decoder_model_path"]

    train_dataset_name = configs["train_dataset_name"]
    valid_dataset_name = configs["valid_dataset_name"]

    output_dir = configs["output_dir"]
    checkpoint_path = configs["checkpoint_path"]

    task_name = configs["task_name"]

    need_evaluate = configs["need_evaluate"]
    log_step = configs["log_step"]

    is_early_stopping = configs["is_early_stopping"]

    trainer = MedViLLT5Trainer(
        configs=configs,
        encoder_model_path=encoder_model_path,
        decoder_model_path=decoder_model_path,
        output_dir=output_dir,
        need_evaluate=need_evaluate,
        task_name=task_name,
        train_dataset_name=train_dataset_name,
        valid_dataset_name=valid_dataset_name,
        log_step=log_step,
        enabel_early_stopping=is_early_stopping,
    )

    # if resume checkpoint is needed, use the following line before trainer.train()
    is_resume = configs["is_resume"]
    is_initialize = configs["is_initialize"]
    if is_resume:
        tqdm.write(Fore.CYAN + f">> Resuming from checkpoint: {checkpoint_path} <<")
        # resume training from the checkpoint
        trainer.resume_checkpoint(
            checkpoint_path=checkpoint_path, initialize=is_initialize
        )

    # if you want to fine-tune the decoder, set fine_tune=True, the hyperparameters setting is in the config f
    is_finetune = configs["is_finetune"]
    if is_finetune:
        tqdm.write(Fore.MAGENTA + ">> ⚡️ Fine-tuning the decoder... ⚡️ <<")
    else:
        tqdm.write(Fore.MAGENTA + ">> ⚡️ Training the model from scratch... ⚡️ <<")

    trainer.train(fine_tune=is_finetune)

    tqdm.write(Fore.GREEN + ">> 🎉 Training completed successfully! 🚀🎉 <<")

    trainer.wandb_run.finish()


if __name__ == "__main__":
    main()
