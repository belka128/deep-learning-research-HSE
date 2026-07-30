import csv

import numpy as np
import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import compute_eer
from src.trainer.base_trainer import BaseTrainer


def _cm_scores(logits: torch.Tensor) -> np.ndarray:
    """CM score = logit_bonafide - logit_spoof (higher = more bonafide-like)."""
    return (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()


class Inferencer(BaseTrainer):
    """
    Inferencer for the LCNN antispoofing countermeasure.

    For every partition it runs on, writes a "{part}_predictions.csv" file
    with one "key,score" row per utterance (no header) — exactly the
    format expected by the course's grading.py / compute_eer. If the
    partition's protocol has ground-truth labels (as the provided
    ASVspoof2019 LA eval protocol does), it also reports EER using the
    official compute_eer routine, so the same number can be sanity-checked
    against grading.py before submission.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (Path): directory to save "{part}_predictions.csv" files.
            metrics: unused, kept for interface compatibility with train.py/inference.py.
            batch_transforms (dict[nn.Module] | None): transforms applied
                to the whole batch (on device).
            skip_model_load (bool): if False, require a pretrained checkpoint.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                (EER, if labels are available) for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch, part_scores, part_labels, part_keys):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        part_scores.append(_cm_scores(batch["logits"]))
        part_labels.append(batch["labels"].detach().cpu().numpy())
        part_keys.extend(batch["key"])

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        part_scores, part_labels, part_keys = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
                self.process_batch(batch, part_scores, part_labels, part_keys)

        scores = np.concatenate(part_scores)
        labels = np.concatenate(part_labels)

        if self.save_path is not None:
            csv_path = self.save_path / f"{part}_predictions.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                for key, score in zip(part_keys, scores):
                    writer.writerow([key, float(score)])
            print(f"Saved {len(part_keys)} predictions to {csv_path}")

        logs = {}
        bona = scores[labels == 1]
        spoof = scores[labels == 0]
        if bona.size > 0 and spoof.size > 0:
            eer, _ = compute_eer(bona, spoof)
            logs["EER"] = eer * 100

        return logs
