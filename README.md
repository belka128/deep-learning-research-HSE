# Anti-spoofing (LightCNN) — ASVspoof2019 LA

Реализация и обучение модели **LightCNN (LCNN)** для детекции подделок голоса
(bonafide/spoof) на партиции Logical Access датасета ASVspoof2019. Код
построен на основе [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

Архитектура — по Table 1 из статьи STC (Lavrentyeva et al., [arXiv:1904.05576](https://arxiv.org/abs/1904.05576)),
front-end — STFT-спектрограмма оттуда же. Схема подготовки данных
(fixed-length trim-pad) и training recipe — по [arXiv:2103.11326](https://arxiv.org/abs/2103.11326).
Loss — Cross-Entropy (обоснование выбора вместо A-Softmax — в отчёте).

## Установка

```bash
pip install -r requirements.txt
```

## Датасет

Датасет — [ASVspoof2019 LA на Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset).
В Kaggle-ноутбуке подключается как Input (обычно монтируется в
`/kaggle/input/asvpoof-2019-dataset`). Точную структуру папок внутри знать не
нужно — код сам находит protocol-файлы и flac-аудио рекурсивно по пути,
переданному в `data_dir`.

## WandB

```bash
wandb login
```

После обучения нужно собрать **WandB Report** (не просто ссылку на run) с
графиками train/eval loss и EER, и сделать его публичным.

## Запуск

```bash
# 1. Sanity-check пайплайна (обязательно перед полным обучением)
python3 train.py -cn=lcnn_onebatchtest data_dir=/kaggle/input/asvpoof-2019-dataset

# 2. Полное обучение
python3 train.py -cn=lcnn data_dir=/kaggle/input/asvpoof-2019-dataset

# 3. Скоринг eval-сета
python3 inference.py -cn=lcnn_inference data_dir=/kaggle/input/asvpoof-2019-dataset \
    inferencer.from_pretrained=saved/lcnn_fft_ce/model_best.pth
```

Гиперпараметры — в `src/configs/lcnn.yaml`, можно переопределять через Hydra
CLI (`trainer.n_epochs=20` и т.п.).

Шаг 3 пишет `data/saved/eval/test_predictions.csv` (`key,score`, без
заголовка). Для сдачи переименовать в `<hse_username>.csv` — часть email до
`@edu.hse.ru` (например `iiivanov.csv`), затем проверить официальным
`grading.py`.

## Kaggle / Colab

```bash
!git clone https://YOUR_TOKEN@github.com/USERNAME/REPO_ID
%cd REPO_ID
!pip install -q -r requirements.txt
!wandb login
!python3 train.py -cn=lcnn data_dir=/kaggle/input/asvpoof-2019-dataset
```

Токен не коммитить и не оставлять в файлах перед сдачей.

## Credits

[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) (Blinorot et al.)
