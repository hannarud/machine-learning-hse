#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Данное задание основано на материалах лекций по логистической регрессии.

import pandas
from sklearn.metrics import roc_auc_score

# Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой принимает значения -1 или 1.
logistic_data = pandas.read_csv('data-logistic.csv', index_col=None, header=None)

logistic_classes = logistic_data[0]
logistic_observations = logistic_data.ix[:,1:].copy()

print(logistic_observations[0:6])
print(logistic_classes[0:6])

# Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10) логистической регрессии. Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
# Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов на соседних итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число итераций десятью тысячами.





# В качестве метрики качества будем использовать AUC-ROC (Area Under ROC-Curve). Она предназначена для алгоритмов бинарной классификации, выдающих оценку принадлежности объекта к одному из классов. По сути, значение этой метрики является агрегацией показателей качества всех алгоритмов, которые можно получить, выбирая какой-либо порог для оценки принадлежности.
# В Scikit-Learn метрика AUC реализована функцией sklearn.metrics.roc_auc_score. В качестве первого аргумента ей передается вектор истинных ответов, в качестве второго — вектор с оценками принадлежности объектов к первому классу.

# Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании? Эти величины будут ответом на задание.
# Обратите внимание, что на вход функции roc_auc_score нужно подавать оценки вероятностей, подсчитанные обученным алгоритмом. Для этого воспользуйтесь сигмоидной функцией: a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).
auc_roc_no_regularization = 0.0
auc_roc_regularization = 0.01

# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.421. При необходимости округляйте дробную часть до трех знаков.
resulting_string = " ".join([repr(round(auc_roc_no_regularization, 3)), repr(round(auc_roc_regularization, 3))])

print(resulting_string)

file_answer = open("answer.txt", "w")
file_answer.write(resulting_string)
file_answer.close()

# Попробуйте поменять длину шага. Будет ли сходиться алгоритм, если делать более длинные шаги? Как меняется число итераций при уменьшении длины шага?

# Попробуйте менять начальное приближение. Влияет ли оно на что-нибудь?
