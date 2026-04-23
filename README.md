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

```bash
git clone https://github.com/kumaroid/CVLabs.git
cd CVLabs
```
### 2. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate
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

