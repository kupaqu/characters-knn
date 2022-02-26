import cv2
from cv2 import IMREAD_GRAYSCALE
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops
import numpy as np

def image_to_text(image, knn):
    text = []
    # бинаризация
    binary = image.copy()
    binary[binary > 0] = 1
    # формируем данные
    labeled = label(binary)
    regions = sorted(regionprops(labeled), key=lambda region: region.bbox[-1]) # сортировка по горизонтали

    mean = mean_dist(regions)
    # предсказание
    i = 0
    while i < len(regions):
    # for i in range(len(regions)):
        features = extract_features(regions[i].image).reshape(1,5)
        ret, _, _, _ = knn.findNearest(features, k=3)
        if i != len(regions)-1:
            left = regions[i].bbox[-1]
            right = regions[i+1].bbox[1]
            if abs(left-right) > mean: # проверка на пробел
                text.append(chr(int(ret)))
                text.append(' ')
                i += 1
                continue
            elif (regions[i+1].bbox[1] > regions[i].bbox[1] and regions[i+1].bbox[1] < regions[i].bbox[-1]) or (regions[i].bbox[-1] > regions[i+1].bbox[1] and regions[i].bbox[-1] < regions[i].bbox[-1]):
                text.append('i')
                i += 2
                continue
        text.append(chr(int(ret)))
        i += 1

    return ''.join(text)

def mean_dist(regions):
    # сначала рассчитаем среднее расстояние между буквами по горизонтали
    dists = np.zeros(len(regions) - 1)
    for i in range(0, len(regions) - 1):
        dists[i] = abs(regions[i].bbox[-1] - regions[i + 1].bbox[1])
    dist = dists.mean() + dists.std()*0.2

    return dist

# извлечение свойств одной буквы
def extract_features(image):
    features = []
    labeled = label(image)
    region = regionprops(labeled)[0]
    # нормализованный центроид 
    features.extend(
        np.array(region.local_centroid)/np.array(region.image.shape)
        )
    # эксцентриситет
    features.append(region.eccentricity)
    # area / (rows * cols)
    features.append(region.extent)
    # момент изображения инвариантный к перемещению, масштабу и повороту
    features.append(region.moments_hu[0])

    return np.array(features, dtype=np.float32)

def load_data(path):
    data = defaultdict(list)
    for p in sorted(path.glob("*")):
        if p.is_dir():
            data.update(load_data(p))
        else:
            gray = cv2.imread(str(p), 0)
            binary = gray.copy()
            binary[binary > 0] = 1
            data[path.name[-1]].append(binary)
    return data

def build_trained_knn():
    knn = cv2.ml.KNearest_create()
    dir = Path.cwd() / "out"
    train_data = load_data(dir / "train")
    if len(train_data) == 0:
        print("Train data is empty!")
    else:
        features = []
        responses = []
        for i, symbol in enumerate(train_data):
            for img in train_data[symbol]:
                features.append(extract_features(img))
                responses.append(ord(symbol))
        features = np.array(features, dtype=np.float32)
        responses = np.array(responses, dtype=np.float32)
        knn.train(features, cv2.ml.ROW_SAMPLE, responses)
    return knn

knn = build_trained_knn()

for i in range(6):
    print(image_to_text(cv2.imread(f'out/{i}.png', IMREAD_GRAYSCALE), knn))