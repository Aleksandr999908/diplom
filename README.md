# Контроль техники упражнений (RTMPose + классификатор скелета)

Десктопное приложение (Tkinter): поза → класс упражнения → фаза → ошибки техники и подсказки.

## Требования

- **Python** 3.9+ (рекомендуется 3.10–3.11; 3.12+ и 3.13 обычно работают — при ошибке колёс см. сообщение `pip`).
- **Windows** (для `start_gui.bat`; Linux/macOS — запуск через `python run_gui.py`).

## Установка

Из корня проекта:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

На Windows после установки зависимостей можно один раз запустить **`prepare_project.bat`**: установка пакетов, скачивание RTMPose, `verify_project.py --strict`, смоук-тест `train.py --one-batch`.

Полный конвейер по шагам (долго на CPU): **`run_all_ordered.bat`** — `extract_skeletons` → `extract_errors` → `train.py` → `train_fault.py` → проверки.

Скачать модель позы **RTMPose** (ONNX):

```bash
python scripts/download_model.py
```

Появится файл `models/rtmpose-m.onnx`.

Проверка окружения и данных:

```bash
python scripts/verify_project.py
```

Строгая проверка (в том числе наличие обученного `.pt`):

```bash
python scripts/verify_project.py --strict
```

## Быстрый запуск GUI (Windows)

Двойной щелчок по **`start_gui.bat`** или из консоли:

```bash
start_gui.bat
```

Вручную:

```bash
python run_gui.py
```

## Подготовка данных и обучение

1. Положите видео в папки **`data/raw/<название_класса>/`** (имя папки = класс упражнения).

2. Извлечь скелеты в `.npy` и `manifest.json`:

```bash
python scripts/extract_skeletons.py
```

3. Обучить **Shift-GCN** (16 каналов признаков, по умолчанию GPU при наличии):

```bash
python scripts/train.py --device auto
```

На CPU обучение медленное; можно ускорить (меньше итераций на эпоху, без temporal-attention):

```bash
python scripts/train.py --device cpu --sampler-multiplier 1 --no-temporal-attn --ema-decay 0
```

Проверка пайплайна без полного обучения (один батч):

```bash
python scripts/train.py --one-batch --device cpu
```

4. После обучения метаданные пишутся в **`models/training_meta.json`**, лучшие веса — в **`models/shift_gcn_best.pt`**.

### Разделение «чистых» видео и ошибок техники

- **`data/raw/<класс>/`** → `python scripts/extract_skeletons.py` → **`manifest.json`** → **`scripts/train.py`** (класс упражнения).
- **`data/Ошибки/<класс>/<тип_ошибки>/`** → `python scripts/extract_errors.py` → **`manifest_errors.json`** → **`scripts/train_fault.py`** (тип ошибки, метки вида `упражнение__fault_id`). Эти данные **не смешиваются** с основным manifest (защита в `SkeletonSequenceDataset`).

### Метрики и логи экспериментов

- Анализ путаниц после обучения: **`python scripts/analyze_confusion.py`** (читает `models/val_confusion.json`).
- Совместный backend **Shift-GCN + LightGBM + TCN**: обучите `scripts/train.py`,
  `scripts/train_manual_lgb.py`, `scripts/train_manual_tcn.py`, затем выполните
  `python scripts/tune_manual_ensemble.py` или
  `python scripts/set_exercise_backend.py hybrid_ensemble`.
- **TensorBoard**: `python scripts/train.py --tensorboard runs/exp1` → `tensorboard --logdir runs`.
- **Weights & Biases**: `pip install wandb` и `python scripts/train.py --wandb-project exercise-gcn`.
- Печать худших классов по recall на каждой эпохе: **`--val-class-metrics`**.

5. Экспорт классификатора в **ONNX** (опционально):

```bash
python scripts/export_classifier_onnx.py --checkpoint models/shift_gcn_best.pt --verify
```

## Опционально: ИИ-тренер (LLM)

Задайте переменные окружения перед запуском GUI (OpenAI-совместимый API):

- `OPENAI_API_KEY`
- `OPENAI_API_BASE` (по умолчанию `https://api.openai.com/v1`)
- `LLM_MODEL` (по умолчанию `gpt-4o-mini`)

Для Ollama: `OPENAI_API_BASE=http://127.0.0.1:11434/v1` и ключ-заглушка при необходимости.

**Chad (chadgpt.ru):** в корне проекта создайте файл `.env` (не коммитьте) или задайте переменные:

- `LLM_PROVIDER=chad`
- `CHAD_API_KEY` — ключ из кабинета «Для разработчиков»
- опционально `CHAD_ENDPOINT` — URL модели, по умолчанию `https://ask.chadgpt.ru/api/public/gpt-4o-mini`

## Тесты

```bash
python -m pytest tests/ -v
```

## Структура (кратко)

| Путь | Назначение |
|------|------------|
| `run_gui.py` | Точка входа GUI |
| `start_gui.bat` | Запуск GUI на Windows |
| `requirements.txt` | Зависимости Python |
| `scripts/download_model.py` | Скачивание RTMPose ONNX |
| `scripts/extract_skeletons.py` | Скелеты из видео → `data/processed/` |
| `scripts/train.py` | Обучение Shift-GCN |
| `scripts/verify_project.py` | Проверка окружения |
| `scripts/export_classifier_onnx.py` | Экспорт классификатора в ONNX |
| `scripts/train_fault.py` | Обучение классификатора типа ошибки (`manifest_errors.json`) |
| `scripts/analyze_confusion.py` | Разбор матрицы ошибок на валидации |
| `models/` | Веса и `training_meta.json` |
| `src/exercise_recognition/` | Код приложения и пайплайна |
