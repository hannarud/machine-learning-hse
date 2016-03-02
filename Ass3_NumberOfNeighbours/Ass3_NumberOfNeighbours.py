#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Загрузите выборку Wine по адресу https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data или из рабочей директории.
data = pandas.read_csv('wine.data', index_col=None, header=None)

# Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний. Более подробно о сути признаков можно прочитать по адресу https://archive.ics.uci.edu/ml/datasets/Wine (или в файле wine.names в рабочей директории)
classes = data[0]
observations = data.ix[:,1:].copy()

# Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True). Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42. В качестве меры качества используйте долю верных ответов (accuracy).
kf = KFold(len(observations), n_folds=5, shuffle=True, random_state=42)

# Найдите точность классификации на кросс-валидации для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)? Данные результаты и будут ответами на вопросы 1 и 2.
cv_accuracy = [cross_val_score(estimator=KNeighborsClassifier(n_neighbors=k), X=observations, y=classes, cv=kf).mean() for k in range(1,51)]

print(cv_accuracy)

answer2 = max(cv_accuracy)
answer1 = cv_accuracy.index(answer2)+1

file1 = open("answer1.txt", "w")
file1.write(repr(answer1))
file1.close()

file2 = open("answer2.txt", "w")
file2.write(repr(round(answer2, 2)))
file2.close()

# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale. Снова найдите оптимальное k на кросс-валидации.
observations_scaled = scale(observations)

scaled_cv_accuracy = [cross_val_score(estimator=KNeighborsClassifier(n_neighbors=k), X=observations_scaled, y=classes, cv=kf).mean() for k in range(1,51)]

print(scaled_cv_accuracy)

answer4 = max(scaled_cv_accuracy)
answer3 = scaled_cv_accuracy.index(answer4)+1

# Какое значение k получилось оптимальным после приведения признаков к одному масштабу? Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?

file3 = open("answer3.txt", "w")
file3.write(repr(answer3))
file3.close()

file4 = open("answer4.txt", "w")
file4.write(repr(round(answer4, 2)))
file4.close()
