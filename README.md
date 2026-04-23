# Лабораторная работа 1: Семантическая сегментация
  
**Датасет:** [Football Semantic Segmentation](https://www.kaggle.com/datasets/sadhliroomyprime/football-semantic-segmentation)  

---

## Описание

Данная лабораторная работа исследует модели семантической сегментации на датасете кадров футбольных трансляций. Датасет содержит **100 изображений** 1920×1080 с полигональными аннотациями **11 классов** (газон, игроки, мяч, зрители и др.) в формате COCO JSON.

### Что реализовано

| Раздел | Содержание |
|--------|-----------|
| **1. Обоснование** | Выбор датасета и метрик (mIoU, Pixel Accuracy, Dice) |
| **2. Бейзлайн (SMP)** | U-Net/ResNet-18 (CNN) и U-Net/MiT-B0 (Transformer) |
| **3. Улучшение** | FPN/ResNet-34 и DeepLabV3+/MiT-B0 с аугментацией |
| **4. Custom U-Net** | U-Net, реализованный вручную без smp |

---

## Структура репозитория

```
.
├── football_segmentation.ipynb   # Основной notebook с результатами
├── training_results.json         # Сохранённые метрики обучения
├── README.md                     # Данный файл
├── dataset/
│   └── football/
│       ├── images/               # 100 .jpg + маски .png
│       └── COCO_Football Pixel.json
├── train_common.py               # Общие утилиты (Dataset, метрики, train_model)
├── train_part1.py                # Обучение: CNN baseline + Transformer baseline
├── train_part2.py                # Обучение: CNN improved + Transformer improved
└── train_part3.py                # Обучение: Custom U-Net
```

---

## Требования

- Python 3.10+
- pip (рекомендуется виртуальное окружение)

### Зависимости

```
torch>=2.0
torchvision>=0.15
segmentation-models-pytorch>=0.3
albumentations>=1.3
pycocotools
scikit-learn
matplotlib
Pillow
numpy
jupyter
```

---

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone <url-репозитория>
cd <папка-репозитория>
```

### 2. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows
```

### 3. Установка зависимостей

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# или для CPU:
pip install torch torchvision

pip install segmentation-models-pytorch albumentations pycocotools \
            scikit-learn matplotlib Pillow numpy jupyter
```

### 4. Скачивание датасета

```bash
mkdir -p dataset
curl -L "https://www.kaggle.com/api/v1/datasets/download/sadhliroomyprime/football-semantic-segmentation" \
     -o dataset/football_seg.zip
unzip dataset/football_seg.zip -d dataset/football
```

> **Альтернатива:** через Kaggle CLI  
> ```bash
> pip install kaggle
> kaggle datasets download sadhliroomyprime/football-semantic-segmentation -p dataset/ --unzip
> ```

### 5. Запуск notebook

```bash
jupyter notebook football_segmentation.ipynb
```

Notebook содержит все результаты обучения (встроены как JSON-словари).  
Для воспроизведения обучения с нуля раскомментируйте вызовы `train_model(...)` в соответствующих ячейках.

### 6. (Опционально) Переобучение моделей

Если хотите запустить обучение заново (занимает ~15 минут на GPU / ~30 мин на CPU):

```bash
python3 train_part1.py   # CNN baseline + Transformer baseline
python3 train_part2.py   # CNN improved + Transformer improved
python3 train_part3.py   # Custom U-Net (base + improved)
```

Результаты сохраняются в `training_results.json`.

---

## Результаты обучения

Все модели обучались на `128×128`, `batch_size=8`, `lr=1e-3`, CPU.

| Модель | mIoU | Pixel Acc | Dice | Эпох |
|--------|------|-----------|------|------|
| U-Net / ResNet-18 (SMP, base) | 0.3385 | 0.8723 | 0.3908 | 5 |
| U-Net / MiT-B0 (SMP, base) | 0.2758 | 0.8186 | 0.3204 | 5 |
| FPN / ResNet-34 (SMP, improved) | 0.2582 | 0.7961 | 0.3141 | 8 |
| **DeepLabV3+ / MiT-B0 (SMP, improved)** | **0.3653** | **0.8537** | **0.4411** | 8 |
| Custom U-Net (base) | 0.3688 | 0.8690 | 0.4382 | 5 |
| Custom U-Net (improved) | 0.2962 | 0.8424 | 0.3522 | 8 |

> Лучший mIoU достигнут у **DeepLabV3+ / MiT-B0** (trансформерный энкодер + ASPP + аугментация).

---

## Метрики качества

| Метрика | Описание | Почему выбрана |
|---------|----------|----------------|
| **mIoU** | Среднее IoU по всем классам | Стандарт сегментации, нечувствительна к дисбалансу |
| **Pixel Accuracy** | Доля правильных пикселей | Интуитивная интерпретация |
| **Dice / F1** | 2·TP / (2·TP + FP + FN) | Устойчива при дисбалансе (мяч << газон) |

---

## Классы датасета

| ID | Класс |
|----|-------|
| 0 | Background |
| 1 | Goal Bar (штанга/ворота) |
| 2 | Referee (арбитр) |
| 3 | Advertisements |
| 4 | Ground (газон) |
| 5 | Ball (мяч) |
| 6 | Coaches & Officials |
| 7 | Audience (зрители) |
| 8 | Goalkeeper B |
| 9 | Goalkeeper A |
| 10 | Team B |
| 11 | Team A |

---

## Стек технологий

- **PyTorch** — фреймворк глубокого обучения
- **segmentation_models.pytorch** — готовые архитектуры сегментации
- **albumentations** — аугментация изображений
- **pycocotools** — парсинг COCO-аннотаций
- **scikit-learn** — разбивка train/val
- **matplotlib** — визуализация

---

## Воспроизводимость

Зафиксированы все seed'ы:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Разбивка train/val выполняется через `train_test_split(..., random_state=42)`.
