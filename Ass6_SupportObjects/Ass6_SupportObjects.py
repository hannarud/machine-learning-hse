#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.svm import SVC

# Загрузите выборку из файла svm-data.csv. В нем записана двумерная выборка (целевая переменная указана в первом столбце, признаки — во втором и третьем). Файлы находятся в рабочей директории.
svm_data = pandas.read_csv('svm-data.csv', index_col=None, header=None)

svm_classes = svm_data[0]
svm_observations = svm_data.ix[:,1:].copy()

print(svm_classes)
print(svm_observations)

# Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241. Такое значение параметра нужно использовать, чтобы убедиться, что SVM работает с выборкой как с линейно разделимой. При более низких значениях параметра алгоритм будет настраиваться с учетом слагаемого в функционале, штрафующего за маленькие отступы, из-за чего результат может не совпасть с решением классической задачи SVM для линейно разделимой выборки.
clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(svm_observations, svm_classes)

# Найдите номера объектов, которые являются опорными (нумерация с единицы). Они будут являться ответом на задание. Обратите внимание, что в качестве ответа нужно привести номера объектов в возрастающем порядке через запятую или пробел. Нумерация начинается с 1.
# Индексы опорных объектов обученного классификатора хранятся в поле support_

support_indexes = clf.support_
# Обратите внимание, что индексация объектов начинается с единицы
str_support_indexes = [repr(i+1) for i in support_indexes]
resulting_string = " ".join(str_support_indexes)

print(resulting_string)

file_answer = open("answer.txt", "w")
file_answer.write(resulting_string)
file_answer.close()
