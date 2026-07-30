import logging
from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json

logger = logging.getLogger(__name__)

# Official ASVspoof2019 LA protocol file name fragments and audio folder
# name fragments, per the dataset release used for this homework
# (https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset,
# see also the provided ASVspoof2019.LA.cm.eval.trl.txt file, which follows
# the same naming/format as below).
_PROTOCOL_HINTS = {
    "train": "cm.train.trn",
    "dev": "cm.dev.trl",
    "eval": "cm.eval.trl",
}
_AUDIO_DIR_HINTS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}


def _find_protocol_path(data_dir: Path, part_name: str) -> Path:
    hint = _PROTOCOL_HINTS[part_name]
    candidates = sorted(data_dir.rglob(f"*{hint}*.txt"))
    assert len(candidates) > 0, (
        f"Could not find a protocol file matching '*{hint}*.txt' under {data_dir}. "
        f"Pass 'protocol_path' explicitly in the dataset config."
    )
    return candidates[0]


def _find_audio_dir(data_dir: Path, part_name: str) -> Path:
    hint = _AUDIO_DIR_HINTS[part_name]
    candidates = [p for p in data_dir.rglob(f"*{hint}*") if p.is_dir()]
    # prefer a candidate that itself contains flac files (e.g. .../flac/)
    for cand in candidates:
        if any(cand.rglob("*.flac")):
            return cand
    assert len(candidates) > 0, (
        f"Could not find an audio directory matching '*{hint}*' under {data_dir}. "
        f"Pass 'audio_dir' explicitly in the dataset config."
    )
    return candidates[0]


def _build_flac_index(audio_dir: Path) -> dict:
    """Map utterance id (file stem) -> absolute flac path."""
    return {p.stem: str(p) for p in audio_dir.rglob("*.flac")}


class ASVspoofDataset(BaseDataset):
    """
    Dataset for the Logical Access (LA) partition of the ASVspoof2019
    dataset, used for the voice anti-spoofing Countermeasure (CM) homework.

    Builds an index from the official CM protocol file (5 whitespace
    separated columns: SPEAKER_ID, AUDIO_FILE_NAME, "-", SYSTEM_ID, KEY,
    where KEY is "bonafide" or "spoof"), matching each entry to the
    corresponding .flac file discovered under 'data_dir'.
    """

    def __init__(
        self,
        data_dir,
        part_name,
        protocol_path=None,
        audio_dir=None,
        sample_rate=16000,
        rebuild_index=False,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir (str): root directory of the (possibly nested)
                ASVspoof2019 LA dataset, e.g. a Kaggle-attached input dir.
            part_name (str): one of "train", "dev", "eval".
            protocol_path (str | None): explicit path to the CM protocol
                file. If None, auto-discovered under 'data_dir'.
            audio_dir (str | None): explicit path to the directory
                containing the .flac files for this partition. If None,
                auto-discovered under 'data_dir'.
            sample_rate (int): target sample rate; audio is resampled to
                this rate if needed.
            rebuild_index (bool): if True, ignore any cached index.json and
                rebuild the index from scratch.
        """
        assert part_name in _PROTOCOL_HINTS, f"Unknown part_name '{part_name}'"

        self.data_dir = Path(data_dir)
        self.part_name = part_name
        self.sample_rate = sample_rate

        index_cache_path = self.data_dir / f".cm_index_{part_name}.json"

        if index_cache_path.exists() and not rebuild_index:
            index = read_json(str(index_cache_path))
        else:
            protocol_path = (
                Path(protocol_path)
                if protocol_path is not None
                else _find_protocol_path(self.data_dir, part_name)
            )
            audio_dir = (
                Path(audio_dir)
                if audio_dir is not None
                else _find_audio_dir(self.data_dir, part_name)
            )
            index = self._create_index(protocol_path, audio_dir)
            write_json(index, str(index_cache_path))

        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path: Path, audio_dir: Path):
        logger.info(f"Building index for '{self.part_name}' from {protocol_path}")
        flac_index = _build_flac_index(audio_dir)

        index = []
        with protocol_path.open("r") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) != 5:
                    continue
                _, utt_id, _, _, key = fields
                assert utt_id in flac_index, (
                    f"Utterance '{utt_id}' listed in {protocol_path} was not found "
                    f"as a .flac file under {audio_dir}"
                )
                index.append(
                    {
                        "path": flac_index[utt_id],
                        "label": 1 if key == "bonafide" else 0,
                        "key": utt_id,
                    }
                )
        return index

    def load_object(self, path):
        """
        Load a mono waveform resampled to self.sample_rate.

        Args:
            path (str): path to the .flac file.
        Returns:
            waveform (Tensor): 1D waveform tensor.
        """
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.squeeze(0)

    def __getitem__(self, ind):
        """
        Same as BaseDataset.__getitem__, but also propagates the
        utterance 'key' (needed to write scores in the official
        grading.py format during inference).
        """
        data_dict = self._index[ind]
        data_object = self.load_object(data_dict["path"])
        instance_data = {
            "data_object": data_object,
            "labels": data_dict["label"],
            "key": data_dict["key"],
        }
        instance_data = self.preprocess_data(instance_data)
        return instance_data
