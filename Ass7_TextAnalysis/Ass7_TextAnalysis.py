#!/usr/bin/env python
# -*- coding: utf-8 -*-

# В этом задании мы применим метод опорных векторов для определения того, к какой из тематик относится новость: атеизм или космос.

import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# Для начала вам потребуется загрузить данные. В этом задании мы воспользуемся одним из датасетов, доступных в scikit-learn'е — 20 newsgroups. Для этого нужно воспользоваться модулем datasets.
# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм". Обратите внимание, что загрузка данных может занять несколько минут
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
# После выполнения этого кода массив с текстами будет находиться в поле newsgroups.data, номер класса — в поле newsgroups.target.
print(newsgroups.data[0:6])
print(newsgroups.target[0:6])

# Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным. При таком подходе получается, что признаки на обучающем множестве используют информацию из тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения целевой переменной из теста. На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения, и поэтому можно ими пользоваться при обучении алгоритма.
# Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.
vectorizer = TfidfVectorizer()
newsgroups_train = vectorizer.fit_transform(newsgroups.data)

# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM, и для KFold. В качестве меры качества используйте долю верных ответов (accuracy).
# Подбор параметров удобно делать с помощью класса sklearn.grid_search.GridSearchCV.
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(len(newsgroups.target), n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(newsgroups_train, newsgroups.target)

# После того, как перебор окончен, можно проанализировать значения качества для всех значений параметров и выбрать наилучший вариант (оптимальное значение C).
best_C = gs.get_params()["estimator__C"]

# Обучите SVM по всей выборке с оптимальным параметром best_C, найденным на предыдущем шаге и random_state=241.
clf = SVC(C=best_C, kernel='linear', random_state=241)
clf.fit(newsgroups_train, newsgroups.target)

# Найдите 10 слов с наибольшим абсолютным значением веса (веса каждого признака у обученного классификатора svm.SVC хранятся в поле coef_). Они являются ответом на это задание. Укажите эти слова через запятую или пробел, в нижнем регистре, в лексикографическом порядке.
# Чтобы понять, какому слову соответствует i-й признак, можно воспользоваться методом get_feature_names() у TfidfVectorizer:
most_important_words_indexes = np.argsort(abs(clf.coef_.toarray()[0]))[-10:]
print(most_important_words_indexes)

most_important_words = np.array(vectorizer.get_feature_names())[most_important_words_indexes]

most_important_words_sorted = sorted(most_important_words)
resulting_string = " ".join(most_important_words_sorted)
print(resulting_string)

file_answer = open("answer.txt", "w")
file_answer.write(resulting_string)
file_answer.close()
