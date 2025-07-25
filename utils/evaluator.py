import re
import sys
import json
import numpy as np
from typing import List, Dict
from typing import Set, Tuple
from collections import Counter

import evaluate
import spacy
import torch
from colorama import init, Fore
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer, scoring
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from utils.get_batch_input_data import get_batch_input_data
from utils.generator_scheduler import scheduler

from scispacy.linking import EntityLinker

init(autoreset=True)


class evaluator:
    def __init__(self, model, valid_data_loader, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.valid_loader = valid_data_loader

        # self.clinical_terms = set()

        self.linker = SciSpacyLinker()

    def compute_rouge(self, cands: List[str], refs: List[List[str]]):
        """
        Compute ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
        """
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        aggregator = scoring.BootstrapAggregator()

        valid_pairs = 0

        for cand, ref_list in zip(cands, refs):
            # filter out empty strings
            valid_refs = [ref.strip() for ref in ref_list if ref.strip()]
            cand = cand.strip()

            if not cand or not valid_refs:
                continue

            try:
                # Get scores from all valid references
                scores = []
                for ref in valid_refs:
                    score = scorer.score(ref, cand)

                    # validate the structure of the returned score object
                    if hasattr(score["rouge1"], "fmeasure"):
                        scores.append(score)
                    else:
                        print(
                            f"Warning: Unexpected score object structure: {type(score['rouge1'])}"
                        )

                if not scores:
                    continue

                # Select the highest score for each metric
                best = {}
                for metric in ["rouge1", "rouge2", "rougeL"]:
                    try:
                        best_score_obj = max(scores, key=lambda s: s[metric].fmeasure)[
                            metric
                        ]
                        best[metric] = best_score_obj
                    except (AttributeError, KeyError, TypeError) as e:
                        print(f"Warning: Error processing {metric}: {e}")
                        best[metric] = scoring.Score(
                            precision=0.0, recall=0.0, fmeasure=0.0
                        )

                aggregator.add_scores(best)
                valid_pairs += 1

            except Exception as e:
                print(f"Warning: Error processing candidate text: {e}")
                continue

        if valid_pairs == 0:
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

        try:
            result = aggregator.aggregate()

            return {
                "rouge1_f": result["rouge1"].mid.fmeasure,
                "rouge2_f": result["rouge2"].mid.fmeasure,
                "rougeL_f": result["rougeL"].mid.fmeasure,
            }
        except Exception as e:
            print(f"Warning: Error aggregating results: {e}")
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

    def compute_cider(self, cands: List[str], refs: List[List[str]]) -> float:
        """
        Compute CIDEr metric

        Args:
            cands: List of candidate texts
            refs: List of reference text lists

        Returns:
            CIDEr score (average value)
        """
        # pycocoevalcap interface requires format: {idx: [cand]}, {idx: [ref1, ref2, ...]}
        hypo = {i: [c] for i, c in enumerate(cands)}
        ref_dict = {i: rs for i, rs in enumerate(refs)}
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(ref_dict, hypo)
        return score  # CIDEr score (average value)

    def compute_clinical_f1(
        self, cands: List[str], refs: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute Clinical F1 metric based on clinical terms
        For each candidate and its corresponding references (take union of multiple refs):
        - Extract clinical terms from predicted and reference texts
        - Compute overall Precision, Recall, F1, and Accuracy across all samples

        Args:
            cands: List of candidate texts
            refs: List of reference text lists

        Returns:
            Dictionary containing clinical precision, recall, F1, and accuracy scores
        """

        def get_unique_word(text_set: set):
            return {
                token
                for term in text_set
                for token in re.findall(r"[A-Za-z0-9]+", term)
            }

        def length_penalty(pred_len, ref_len, sigma=5.0, alpha=2.0, rate=1):
            assert 0 < rate <= 1
            d = abs(pred_len - ref_len)
            return np.exp(alpha * np.exp(-(d**2) / (2 * sigma**2))) ** rate

        y_true = []
        y_pred = []

        penalties = []

        for cand, ref_list in zip(cands, refs):

            pred_terms = set(self.linker.extract_umls_cuis(cand))

            ref_terms = set()
            for ref in ref_list:
                ref_terms.update(self.linker.extract_umls_cuis(ref))

            pred_terms = get_unique_word(pred_terms)
            ref_terms = get_unique_word(ref_terms)

            pen = length_penalty(len(pred_terms), len(ref_terms))
            penalties.append(pen)

            for cui in ref_terms.union(pred_terms):
                y_true.append(1 if cui in ref_terms else 0)
                y_pred.append(1 if cui in pred_terms else 0)

        if len(y_true) == 0 or sum(y_true) == 0:
            return {
                "clinical_precision": 0.0,
                "clinical_recall": 0.0,
                "clinical_f1": 0.0,
                "clinical_accuracy": 0.0,
            }

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true)

        avg_penalty = float(np.mean(penalties))

        p, r, f1, accuracy = map(
            lambda x: np.tanh(avg_penalty * x), [p, r, f1, accuracy]
        )

        return {
            "clinical_precision": p,
            "clinical_recall": r,
            "clinical_f1": f1,
            "clinical_accuracy": accuracy,
        }

    def compute_token_metrics(
        self, pred_tokens: List[str], ref_tokens: List[str]
    ) -> Dict[str, float]:
        """
        Computes token-based metrics based on presence in the reference list, not on exact position.
        - Accuracy: What proportion of predicted tokens are valid (i.e., exist in ref_tokens)?
        - Precision/Recall/F1: Based on the overlap of unique tokens between predicted and reference sets.
        """
        if not pred_tokens and not ref_tokens:
            return {
                "token_accuracy": 1.0,
                "token_precision": 1.0,
                "token_recall": 1.0,
                "token_f1": 1.0,
            }
        if not pred_tokens or not ref_tokens:
            return {
                "token_accuracy": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "token_f1": 0.0,
            }

        # --- "Token Accuracy" ---
        correctly_predicted_count = sum(
            1 for p_token in pred_tokens if p_token in ref_tokens
        )
        accuracy = correctly_predicted_count / len(pred_tokens)

        # --- Precision, Recall, F1 ---
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)

        intersection = pred_set.intersection(ref_set)

        precision = len(intersection) / len(pred_set)

        recall = len(intersection) / len(ref_set)

        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "token_accuracy": accuracy,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
        }

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> Dict[str, float]:
        assert (
            self.valid_loader is not None
        ), "Validation loader is not initialized. Please set evaluate=True when creating the trainer."

        bleu = evaluate.load("bleu")
        meteor = evaluate.load("meteor")

        total_nll = 0.0
        total_tokens = 0

        all_preds = []
        all_refs = []

        flat_pred_tokens = []
        flat_ref_tokens = []

        self.model.eval()
        tqdm.write(Fore.MAGENTA + " >> ⚡️ Evaluating model... ⚡️ << ")
        evaluate_bar = tqdm(
            self.valid_loader, desc="Evaluating", file=sys.stdout, position=1
        )

        for batch in evaluate_bar:

            batch_data = get_batch_input_data(batch, self.device, self.tokenizer)

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

            mask = batch_data["labels"] != -100
            n_tokens = mask.sum().item()
            total_nll += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

            encoder_hidden = self.model.decoder.init_state(encoder_out)

            decode_params = scheduler.get_params(epoch)

            decode_params.update(
                {
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
            )

            generated_ids = self.model.decoder.generate(
                encoder_outputs=encoder_hidden["encoder_outputs"], **decode_params
            )
            preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            batch_size = batch_data["labels"].size(0)
            refs = []
            for i in range(batch_size):
                label_seq = batch_data["labels"][i].tolist()

                try:
                    idx = label_seq.index(-100)
                    label_seq = label_seq[:idx]
                except ValueError:
                    label_seq = [tok for tok in label_seq if tok >= 0]
                if not label_seq:
                    continue
                refs.append(self.tokenizer.decode(label_seq, skip_special_tokens=True))

            n = min(len(preds), len(refs))
            all_preds.extend(preds[:n])
            all_refs.extend([[r] for r in refs[:n]])

            for p, r in zip(preds[:n], refs[:n]):
                ptoks = self.tokenizer.tokenize(p)
                rtoks = self.tokenizer.tokenize(r)
                flat_pred_tokens.extend(ptoks)
                flat_ref_tokens.extend(rtoks)

        # BLEU & METEOR
        bleu4 = bleu.compute(predictions=all_preds, references=all_refs, max_order=4)[
            "bleu"
        ]
        meteor_score = meteor.compute(
            predictions=all_preds, references=[r[0] for r in all_refs]
        )["meteor"]

        # Compute new evaluation metrics with error handling
        try:
            rouge_scores = self.compute_rouge(all_preds, all_refs)
        except Exception as e:
            tqdm.write(f"Warning: ROUGE computation failed: {e}")
            rouge_scores = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

        try:
            cider_score = self.compute_cider(all_preds, all_refs)
        except Exception as e:
            tqdm.write(f"Warning: CIDEr computation failed: {e}")
            cider_score = 0.0

        try:
            clinical_f1_scores = self.compute_clinical_f1(all_preds, all_refs)
        except Exception as e:
            tqdm.write(f"Warning: Clinical F1 computation failed: {e}")
            clinical_f1_scores = {
                "clinical_precision": 0.0,
                "clinical_recall": 0.0,
                "clinical_f1": 0.0,
                "clinical_accuracy": 0.0,
            }

        try:
            token_metrics = self.compute_token_metrics(
                flat_pred_tokens, flat_ref_tokens
            )
        except Exception as e:
            tqdm.write(f"Warning: Token metrics computation failed: {e}")
            token_metrics = {
                "token_accuracy": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "token_f1": 0.0,
            }

        self.model.train()

        # Combine all metrics
        metrics = {
            "BLEU-4": bleu4,
            "METEOR": meteor_score,
            # New ROUGE metrics
            "ROUGE-1": rouge_scores["rouge1_f"],
            "ROUGE-2": rouge_scores["rouge2_f"],
            "ROUGE-L": rouge_scores["rougeL_f"],
            # CIDEr metric
            "CIDEr": cider_score,
            # Clinical F1 metrics
            "CLINICAL_PRECISION": clinical_f1_scores["clinical_precision"],
            "CLINICAL_RECALL": clinical_f1_scores["clinical_recall"],
            "CLINICAL_F1": clinical_f1_scores["clinical_f1"],
            "CLINICAL_ACCURACY": clinical_f1_scores["clinical_accuracy"],
            "TOKEN_ACCURACY": token_metrics["token_accuracy"],
            "TOKEN_PRECISION": token_metrics["token_precision"],
            "TOKEN_RECALL": token_metrics["token_recall"],
            "TOKEN_F1": token_metrics["token_f1"],
        }

        return metrics


class SciSpacyLinker:

    def __init__(
        self,
        model: str = "en_core_sci_scibert",
        resolve_abbreviations: bool = True,
        linker_name: str = "umls",
    ):

        spacy.require_gpu()
        self.nlp = spacy.load(model)
        self.linker = self.nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": resolve_abbreviations,
                "linker_name": linker_name,
            },
        )
        self.kb = self.linker.kb

    def extract_umls_cuis(
        self, text: str, score_threshold: float = None, top_k: int = None
    ) -> List[str]:
        doc = self.nlp(text)
        all_cuis: List[str] = []

        for ent in doc.ents:
            # ent._.kb_ents is List[Tuple[cui, score]]
            candidates: List[Tuple[str, float]] = ent._.kb_ents

            if score_threshold is not None:
                candidates = [
                    (cui, score)
                    for cui, score in candidates
                    if score >= score_threshold
                ]

            if top_k is not None:
                candidates = candidates[:top_k]

            for cui, _ in candidates:
                all_cuis.append(cui)

        return all_cuis

    def count_medical_terms_in_text(self, text: str) -> Counter:
        """
        Count UMLS CUIs in a given text.
        :param text: Input text to process.
        :return: Counter with CUIs and their counts.
        """
        counter = Counter()
        counter.update(self.extract_umls_cuis(text))
        return counter

    def count_medical_terms(self, file_paths: List[str]) -> Counter:
        counter = Counter()

        for fp in file_paths:
            with open(fp, "r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
            with open(fp, "r", encoding="utf-8") as f:
                for line in tqdm(
                    f, desc=f"Processing {fp}", total=total, unit="line", leave=False
                ):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text", "")
                    if text:
                        counter.update(self.extract_umls_cuis(text))
        return counter

    def filter_and_export(
        self, source: List[str] | str, output_file: str = ""
    ) -> Set[str]:
        source_type = type(source)

        if source_type == list:
            cui_counts = self.count_medical_terms(source)
        else:
            assert source_type == str and output_file == ""
            cui_counts = self.count_medical_terms_in_text(source)

        terms = set()

        for cui in cui_counts:
            ent = self.kb.cui_to_entity.get(cui)
            # if ent and set(ent.types) & self.allowed_types:
            if ent and set(ent.types):
                terms.add(ent.canonical_name.lower())

        if source_type == list:
            with open(output_file, "w", encoding="utf-8") as fout:
                json.dump({"terms": list(terms)}, fout, ensure_ascii=False, indent=2)

            tqdm.write(f"Wrote {len(terms)} terms to {output_file}")
            return terms

        return terms


if __name__ == "__main__":
    pass
