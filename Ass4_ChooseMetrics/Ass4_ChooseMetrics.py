#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

# Мы будем использовать в данном задании набор данных Boston, где нужно предсказать стоимость жилья на основе различных характеристик расположения (загрязненность воздуха, близость к дорогам и т.д.). Подробнее о признаках можно почитать по адресу https://archive.ics.uci.edu/ml/datasets/Housing
# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston(). Результатом вызова данной функции является объект, у которого признаки записаны в поле data, а целевой вектор — в поле target.
boston_dataset = load_boston()
observations = boston_dataset.data
targets = boston_dataset.target

print(observations)
print(targets)

# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
observations_scaled = scale(observations)

# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace). Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей. В качестве метрики качества используйте среднеквадратичную ошибку (параметр scoring='mean_squared_error' у cross_val_score). Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42, не забудьте включить перемешивание выборки (shuffle=True).
p_values = np.linspace(1.0, 10.0, num=200)
kf = KFold(len(observations_scaled), n_folds=5, shuffle=True, random_state=42)

cv_accuracy = [cross_val_score(estimator=KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_i, metric='minkowski'), X=observations_scaled, y=targets, cv=kf).mean() for p_i in p_values]
print(cv_accuracy)

# Определите, при каком p качество на кросс-валидации оказалось оптимальным (обратите внимание, что cross_val_score возвращает массив показателей качества по блокам; необходимо максимизировать среднее этих показателей). Это значение параметра и будет ответом на задачу.
best_p = p_values[max(cv_accuracy)]

file_answer = open("answer.txt", "w")
file_answer.write(repr(round(best_p, 1)))
file_answer.close()
