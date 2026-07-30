import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors. Also contains "key" (list[str]) with the
            ASVspoof utterance ids, used to write predictions in the
            official grading.py format during inference.
    """

    result_batch = {}

    # data_object is a fixed-size (n_freq, n_frames) spectrogram per item,
    # stack adds the batch dimension -> (batch, n_freq, n_frames)
    result_batch["data_object"] = torch.stack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor(
        [elem["labels"] for elem in dataset_items], dtype=torch.long
    )

    if "key" in dataset_items[0]:
        result_batch["key"] = [elem["key"] for elem in dataset_items]

    return result_batch
