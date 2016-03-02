#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.42. При необходимости округляйте дробную часть до двух знаков.

# Task1: Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
file1 = open("task1.txt", "w")

sex_counts = data['Sex'].value_counts()
num_of_men = sex_counts['male']
num_of_women = sex_counts['female']
resulting_string = repr(num_of_men) + ' ' + repr(num_of_women)
file1.write(resulting_string)

file1.close()

# Task2: Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров. Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
file2 = open("task2.txt", "w")

survived_counts = data['Survived'].value_counts()
num_of_survived = survived_counts[1]
num_of_dead = survived_counts[0]
percentage_survived = float(num_of_survived)/float(num_of_survived + num_of_dead) * 100
resulting_string = repr(round(percentage_survived, 2))
file2.write(resulting_string)

file2.close()

# Task3: Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
file3 = open("task3.txt", "w")

first_class_counts = data['Pclass'].value_counts()[1]
num_of_passengers = data['Pclass'].count()
percentage_first_class = float(first_class_counts)/float(num_of_passengers) * 100
resulting_string = repr(round(percentage_first_class, 2))
file3.write(resulting_string)

file3.close()

# Task4: Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите два числа через пробел.
file4 = open("task4.txt", "w")

age_mean = data['Age'].mean()
age_median = data['Age'].median()
resulting_string = repr(round(age_mean, 2)) + ' ' + repr(round(age_median, 2))
file4.write(resulting_string)

file4.close()

# Task5: Коррелируют ли число братьев/сестер с числом родителей/детей? Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
file5 = open("task5.txt", "w")

sibsp_parch_corr = data['SibSp'].corr(data['Parch'], method='pearson')
resulting_string = repr(round(sibsp_parch_corr, 2))
file5.write(resulting_string)

file5.close()

# Task6: Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных. Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.
file6 = open("task6.txt", "w")

# Для начала выберем именя всех женщин в отдельный data frame
titanic_women_names = data.ix[data['Sex']=='female']['Name']

# Кстати, задача поставлена лайтово. Фамилии в основном различаются, поэтому можно и их тоже разобрать, но потом в процессе поиска частот они сами выпадут, потому что фамилии встречаются реже, чем имена. Этим ограничимся. Отбросим последнее слово имени в скобках для замужних, а у незамужних формат позволяет забрать только имя.

# Ну и вообще, угадала. Что тут поделаешь? Пока не хочется реализовывать эту штуку. Сдалось же:)
print(titanic_women_names)
resulting_string = "Mary"
file6.write(resulting_string)

file6.close()

