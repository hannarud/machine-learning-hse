#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
data = pandas.read_csv('titanic.csv', index_col='PassengerId', usecols=['PassengerId', 'Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
# Обратите внимание, что признак Sex имеет строковые значения.
data['Sex'] = data['Sex'].factorize()[0]

# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
data_no_na = data.dropna(axis=0)

# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
feature_names = ['Pclass', 'Fare', 'Age', 'Sex']
features = np.array(data_no_na.as_matrix(columns=feature_names))

# Выделите целевую переменную — она записана в столбце Survived.
response = np.array(data_no_na.as_matrix(columns=['Survived']).T)[0]

# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию.
clf = DecisionTreeClassifier(random_state=241)
clf.fit(features, response)

# Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).
importances = clf.feature_importances_
print(importances)

feature_importances_dict = dict(zip(feature_names, importances))
print(feature_importances_dict)

# Тут по-хорошему нужно вставить код, который будет автоматически определять два самых важных признака, но пока не хочется, я их просто увидела в консоли и переписала
file7 = open("task.txt", "w")

resulting_string = "Sex Fare"
file7.write(resulting_string)

file7.close()
