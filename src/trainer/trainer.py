import numpy as np
import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import compute_eer
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


def _cm_scores(logits: torch.Tensor) -> np.ndarray:
    """
    Countermeasure (CM) score from model logits: logit_bonafide -
    logit_spoof. Higher score means more likely bonafide, matching the
    convention expected by compute_eer (and by the official grading.py).
    """
    return (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()


class Trainer(BaseTrainer):
    """
    Trainer for the LCNN antispoofing countermeasure.

    Equal Error Rate (EER) is the required performance metric, but it
    cannot be computed as a simple per-batch average like accuracy: a
    64-sample mini-batch (with a ~1:9 bonafide:spoof ratio) does not
    contain enough bonafide trials for a meaningful EER estimate, and the
    official definition requires scores pooled over a whole evaluation
    set. We therefore accumulate (score, label) pairs and compute EER once
    per logging window (train) / once per full epoch (dev), instead of
    using the generic per-batch MetricTracker averaging used for the
    loss and other metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_scores, self._train_labels = [], []
        self._eval_scores, self._eval_labels = [], []

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss, do a
        training step, and accumulate CM scores for the EER computation.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        scores = _cm_scores(batch["logits"])
        labels = batch["labels"].detach().cpu().numpy()
        if self.is_train:
            self._train_scores.append(scores)
            self._train_labels.append(labels)
        else:
            self._eval_scores.append(scores)
            self._eval_labels.append(labels)

        return batch

    @staticmethod
    def _eer_from_buffers(scores_list, labels_list):
        """Compute EER (in %) from accumulated score/label buffers, or None if degenerate."""
        if len(scores_list) == 0:
            return None
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)
        bona = scores[labels == 1]
        spoof = scores[labels == 0]
        if bona.size == 0 or spoof.size == 0:
            return None
        eer, _ = compute_eer(bona, spoof)
        return eer * 100

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch. Identical to BaseTrainer, except the
        rolling EER over the current logging window is computed and logged
        alongside the loss every 'log_step' batches (see class docstring).
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self._train_scores, self._train_labels = [], []
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        last_train_metrics = {}
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(batch, metrics=self.train_metrics)
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                train_eer = self._eer_from_buffers(self._train_scores, self._train_labels)
                if train_eer is not None:
                    self.train_metrics.update("EER", train_eer, n=1)
                self._train_scores, self._train_labels = [], []

                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.
        Identical to BaseTrainer, except EER is computed once over the
        whole partition (see class docstring) instead of per-batch.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        self._eval_scores, self._eval_labels = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                try:
                    batch = self.process_batch(batch, metrics=self.evaluation_metrics)
                except torch.cuda.OutOfMemoryError as e:
                    if self.skip_oom:
                        self.logger.warning("OOM on eval batch. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            eer = self._eer_from_buffers(self._eval_scores, self._eval_labels)
            if eer is not None:
                self.evaluation_metrics.update("EER", eer, n=1)

            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(batch_idx, batch, part)

        return self.evaluation_metrics.result()

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log a histogram of bonafide/spoof CM scores for the current batch,
        similar to the score-distribution plots (Fig. 1) in the STC paper.
        """
        scores = _cm_scores(batch["logits"])
        labels = batch["labels"].detach().cpu().numpy()
        bona = scores[labels == 1]
        spoof = scores[labels == 0]
        # WandBWriter.add_histogram's default bins=None is not accepted by
        # np.histogram, so an explicit bin count must always be passed.
        if bona.size > 0:
            self.writer.add_histogram("bonafide_scores", torch.from_numpy(bona), bins=20)
        if spoof.size > 0:
            self.writer.add_histogram("spoof_scores", torch.from_numpy(spoof), bins=20)
