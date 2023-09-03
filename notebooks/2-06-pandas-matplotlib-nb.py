# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     cell_metadata_json: true
#     notebook_metadata_filter: 'all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version,
#
#       -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,
#
#       -language_info.file_extension, -language_info.mimetype, -toc'
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#   nbhosting:
#     title: '`matplotlib` et `pandas`'
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %% {"scrolled": true}
from IPython.display import HTML
HTML(url="https://raw.githubusercontent.com/ue12-p23/numerique/main/notebooks/_static/style.html")

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import IPython

# %% [markdown]
# # `matplotlib` et `pandas`

# %% [markdown] {"tags": ["framed_cell"]}
# ## introduction
#
# ````{admonition} →
# Les fonctionnalités de `matplotlib` ont été intégrées avec la librairie `pandas`  
# afin de faciliter leur utilisation à partir de dataframe et de séries
#
# nous allons illustrer les quelques fonctions sur un petit exemple  
# *(référez-vous à la documentation pour aller plus loin dans les réglages -nous resterons ici très simples)*
#
# une `pandas.DataFrame` est une table de données en dimension 2  
# `matplotlib` lui apporte des facilités de visualisations
# 1. de données des `pandas.Series`  
# e.g plot, boxplots (boîtes à moustaches), histogrammes, barcharts...
# 1. de nuages de points 2D ou 3D impliquant plusieurs colonnes
#
# nous allons voir quelques plots intéressants sur l'exemple des iris
# ````

# %% [markdown]
# ***

# %% [markdown] {"tags": ["framed_cell"]}
# ## la dataframe des `iris`
#
# ````{admonition} →
# lisons le `csv` des `iris`  avec `pandas`  
# affichons les 2 premières lignes
#
# ```python
# df = pd.read_csv('iris.csv')
# df.head(2)
# ->
#      SepalLength    SepalWidth    PetalLength    PetalWidth    Name
# 0    5.1            3.5           1.4            0.2           Iris-setosa
# 1    4.9            3.0           1.4            0.2           Iris-setosa
# ```
#
# affichons les petites statistiques  
# elles donnent une bonne première idée des données, de leur répartition...
#
# ```python
# df.head(2)
# df.describe()
# ->       SepalLength SepalWidth PetalLength PetalWidth
# count    150.000000  150.000000 150.000000  150.000000
# mean     5.843333    3.054000   3.758667    1.198667
# std      0.828066    0.433594   1.764420    0.763161
# min      4.300000    2.000000   1.000000    0.100000
# 25%      5.100000    2.800000   1.600000    0.300000
# 50%      5.800000    3.000000   4.350000    1.300000
# 75%      6.400000    3.300000   5.100000    1.800000
# max      7.900000    4.400000   6.900000    2.500000
# ```
#
# remarquez que `describe` par défaut  
# n'affiche que les 4 colonnes numériques
#
# (*dans le code ci-dessous, pour plus de lisibilité  
# nous utilisons l'affichage `html` avec `IPython.display.display`*)
# ````

# %%
# le code
df = pd.read_csv('iris.csv')
IPython.display.display(   df.head(2)      )

IPython.display.display(   df.describe()   )

# %% [markdown] {"tags": ["framed_cell"]}
# ## visualisation de la dataframe - `df.plot()`
#
# ````{admonition} →
# la méthode `plot`  des objets de type `pandas.DataFrame` i.e. `pandas.DataFrame.plot`  
# permet une première visualisation simple, rapide et informative **des colonnes numériques**  
# qui apporte beaucoup d'informations sur ces données
#
# la fonction `pandas.DataFrame.plot` possède les mêmes paramètres que la fonction `matplotlib.pyplot.plot`  
# elle permet les mêmes réglages  
# (en fait elles utilisent toutes les deux la même fonction)
#
# ```python
# df.plot()
# ```
#
# <img src='media/iris-plot.png'>
# ````

# %%
# le code
df.plot();

# %% [markdown] {"tags": ["framed_cell"]}
# ## boxplots des colonnes `df.boxplot`
#
# ````{admonition} →
# un `boxplot` montre:  
# le minimun, le maximum, la médiane, le premier et le troisième quartile  
# les (éventuels) outliers
#
# les **outliers**  
# sont les points en dehors de *bornes* décidées par `boxplot`  
# ces points sont potentiellement aberrants ou simplement des extrêmes  
# (lire la doc pour connaître les bornes considérées)  
#
# nous pouvons dessiner les boxplots ensemble  
# ils sont alors mis à la même échelle  
#
# ```python
# df.boxplot()
# ```
#
# <img src='media/iris-boxplot.png'>
#
# on remarque des outliers dans la colonne des `SepalWidth`  
#
# nous pouvons dessiner les boxplots des colonnes indiquées par une liste
#
# ```python
# df.boxplot(['SepalWidth', 'PetalWidth'])
# ```
#
# nous pouvons regrouper les boxplots suivant les valeurs d'une colonne  
# (cela nous rappelle `groupby`, c'est très utile)
#
# ```python
# df.boxplot(['PetalLength'], by='Name')
# ```
#
# <img src='media/iris-boxplot-by.png'>
#
# nous remarquons  
# que les iris *Setosa* ont des `PetalLength` bien plus petits que ceux des autres types d'iris  
# ce qui permet de les discriminer des deux autres types d'iris
# ````

# %%
# le code

df.boxplot()

plt.show() # afin de ne pas superposer les plots

df.boxplot(['SepalWidth', 'PetalWidth']);

plt.show()

df.boxplot(['PetalLength'], by='Name')

plt.tight_layout() # le padding

# %% [markdown] {"tags": ["framed_cell"]}
# ## histogrammes `df.hist`
#
# ````{admonition} →
# un histogramme donne la distribution des valeurs d'une colonne
#
# les valeurs de la colonne sont rangées dans des intervalles - ou *bins*  
# les nombres de valeurs par intervalle sont affichés
#
# ```python
# df.hist()
# ```
# <img src='media/iris-hist.png'>
#
# on remarque 3 pics dans `SepalLength`  
# correspondent-ils aux 3 types d'iris ?
#
# on peut dessiner l'histogramme d'une seule colonne  
# on peut changer des paramètres comme le nombre d'intervalles `bins=`, la couleur `color=`...
#
# ```python
# df.hist('SepalLength', bins=10, color='lightblue')
# ```
# ````

# %% {"cell_style": "center", "scrolled": true}
# le code
df.hist()
df.hist('SepalLength', bins = 10, color='lightblue')
plt.title('histogramme de la colonne SepalLength');

# %% [markdown] {"tags": ["framed_cell"]}
# ## barchart `df.plot.bar()`
#
# ````{admonition} →
# prenons un exemple pour illustrer le dessin des barres  
# la dataframe `df_animals` des animaux, leur vitesse et leur durée de vie
#
# barres verticales
#
# ```python
# df_animals.plot.bar()
# ```
#
# barres horizontales
#
# ```python
# df_animals.plot.hbar()
# ```
#
# une seule colonne
#
# ```python
# df_animals.plot.barh(x='lifespan')
# ```
#
# une colonne
#
# ```python
# df_animals.plot.barh(x='lifespan')
# ```
#
# une colonne en fonction d'une autre
#
# ```python
# df_animals.plot.barh(x='lifespan')
# ```
#
# utilisez le `help`
# ````

# %%
# le code
df_animals = pd.DataFrame({'speed' : [0.1, 17.5, 40, 48, 52, 69, 88],
                   'lifespan' : [2, 8, 70, 1.5, 25, 12, 28]},
                  index = ['snail', 'pig', 'elephant',
                           'rabbit', 'giraffe', 'coyote', 'horse'])

df_animals.plot.bar()

df_animals.plot.barh()

df_animals.plot.bar(x='lifespan', y='speed');

# %% [markdown] {"tags": ["framed_cell"]}
# ## la colonne des `'Name'`
#
# ````{admonition} →
# revenons à nos `iris`
#
# affichons la description de la colonne des types de fleurs `'Name'`
#
# ```python
# df[['Name']].describe()
# ->
# Name
# count	150
# unique	3
# top	Iris-versicolor
# freq	50
# ```
#
# nous avons 3 noms uniques donc 3 types différents d'iris
#
# comptons le nombre d'observations  
# par valeurs dans cette colonne
#
# ```python
# df['Name'].value_counts()
# ->
# Iris-versicolor    50
# Iris-virginica     50
# Iris-setosa        50
# Name: Name, dtype: int64
# ```
#
# on remarque que les 3 types sont bien répartis dans les données (1/3)
#
# affichons le type des éléments de la colonne `Name`
#
# ```python
# df['Name'].dtype
# -> 'O'
# ```
#
# `O` signifie `object`
#
# ce type est `object` ici ce sont des objets de type chaînes de caractères
#
# pourtant ...  
# la colonne des noms des `iris` est plutôt une colonne de type catégorie  
# avec ses 3 valeurs `Iris-versicolor`, `Iris-virginica` et `Iris-setosa`
#
# nous allons changer le type des éléments de la série `df['Name']`  
# ````

# %%
# le code
IPython.display.display(   df[['Name']].describe()   )
df['Name'].value_counts()

# %%
#le code
df['Name'].dtype

# %% [markdown]
# ## encodage des `'Names'` en codes de catégorie

# %% [markdown]
# ````{admonition} →
#
# la colonne `df['Name']` est de type `pandas.Series`  
#
# avec la méthode `astype` des `pandas.Series`  
# on crée une nouvelle colonne    
# avec ici le type `'category'`
#
# ```python
# col = df['Name'].astype('category')
# col.head(2)
# ->
# 0    Iris-setosa
# 1    Iris-setosa
# Name: Name, dtype: category
# Categories (3, object): ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# ```
#
# **remarquez** l'ordre dans la liste des catégories  
# (`'Iris-setosa'` est à l'indice 0...)
#
#
#
# Nous allons maintenant extraire de cette nouvelle colonne  
# les **codes** générés par `pandas` pour les trois catégories d'`iris`
#
# **à savoir:** sur une colonne de type `category`  
# - `cat` permet d'accéder aux méthodes et attributs des objets de type category  
# (comme `str` le permet sur les colonnes d'éléments de type `str`)
# - l'attribut `codes` des colonnes category permet d'accéder aux codes numériques  
# donné par `pandas` aux 3 catégories  
# (dans l'ordre de la liste des catégories)
#
# - on crée une nouvelle colonne `'Name-code'` avec ces codes  
# on regarde ce qu'elle contient
#
# ```python
# df['Name-code'] = col.cat.codes
# df['Name-code'].value_counts()
# ->
# Name-code
# 0    50
# 1    50
# 2    50
# Name: count, dtype: int64
# ```
#
# -------------------------
#
# À quoi cela va-t-il nous servir ?  
# par exemple à améliorer nos visualisations  
# où ces codes peuvent servir de code-couleur lors d'affichage des Iris  
# (nous y reviendrons lors de `scatter`)
# ````

# %%
# le code
col = df['Name'].astype('category')
col.head(2)

# %%
# le code
df['Name-code'] = col.cat.codes
df['Name-code'].value_counts()

# %%
# et en une seul ligne
df['Name-code'] = df['Name'].astype('category').cat.codes

# %% [markdown]
# ## nuages de points `df.plot.scatter`

# %% [markdown]
#
# pour mettre en valeur des informations sur nos données  
# on peut dessiner en 2D les colonnes les unes par rapport aux autres  
# avec `pandas.DataFrame.plot.scatter`
#
# dessinons les `'SepalLength'` en fonction des `'SepalWidth'`
#
# ```python
# df.plot.scatter(x='SepalLength', y='SepalWidth')
# ```
#
# on peut le faire directement en `matplotlib.pyplot.plot`  
# mais il faut alors préciser tous les paramètres (noms des axes...)
#
# ```python
# plt.scatter(df['SepalLength'], df['SepalWidth'])
# ```
#
# avec le paramètre `c=`  
#
# * on peut changer la couleur  
# * mais on peut aussi, indiquer une **couleur par point**  
# * une idée du code couleur intéressant à utiliser ?
#
# oui, on peut représenter ainsi la catégorie des points  
#
# * chacun des 3 types d'iris, est une valeur entière différente  
# * on va considérer cette valeur comme un code dans une table de couleurs
# * attention au code `0` (il peut être très peu coloré dans certaines tables)
#
# ```python
# df.plot.scatter(x='SepalLength', y='SepalWidth', c='Name-code', cmap='viridis');
# ```
#
# avec `matplotlib.pyplot.plot`  
# mais vous n'avez alors que les paramètres par défaut
#
# ```python
# plt.scatter(df['SepalLength'], df['SepalWidth'], c=df['Name-code'], cmap='viridis')
# plt.colorbar() # sinon pas de jolie barre de couleur
# ```
#
# avec le paramètre `s=` on peut changer la taille des points  
# ou la taille de chaque point  
# par exemple, donnons leur une taille proportionnelle à la largeur des pétales  
#
# ```python
# plt.scatter(df['SepalLength'], df['SepalWidth'], c=df['Name-code'], s=df['PetalWidth']);
# ```
#
# ainsi sur un même dessin on peut voir 4 informations  
# le nuage, la couleur et la taille des points
#
# il faut travailler un peu les paramètres pour que ce soit visible  
# (là la taille est trop peu différenciée, multipliez la)

# %%
# le code
df.plot.scatter(x='SepalLength', y='SepalWidth');

# %%
# le code
plt.scatter(df['SepalLength'], df['SepalWidth'])
# plt.xlabel('SepalLength')
# plt.ylabel('SepalWidth')

# %%
# le code
df.plot.scatter(x='SepalLength', y='SepalWidth', c='Name-code', cmap='viridis');

# %%
# le code
plt.scatter(df['SepalLength'], df['SepalWidth'], c=df['Name-code'], cmap='viridis')
plt.colorbar();

# %%
# le code
plt.scatter(df['SepalLength'], df['SepalWidth'], c=df['Name-code'], s=df['PetalWidth']*50);

# %% [markdown] {"tags": ["level_intermediate"]}
# ## fabriquer son propre type `category`

# %% [markdown] {"tags": ["level_intermediate"]}
# *pour les avancés*
#
# avec la technique précédente on n'a pas de **contrôle sur l'ordre** parmi les différentes catégories
#
# imaginez que nous avons maintenant une colonne dont les valeurs uniques sont  
# `bad`, `average`, `good`, `excellent`  
# cette colonne est clairement une colonne de type catégorie ordonnée
#
# on peut définir *son propre type catégoriel* avec la fonction  
# `pd.CategoricalDtype()`  
# dont le paramètre `ordered` permet de dire si la catégorie est ordonnée ou non
#
# en l'appliquand à la colonne des `'Names'` je peux ensuite trier la dataframe  
# sur cette colonne
#
# ```python
# iris_ord_cat = pd.CategoricalDtype(
#                     categories=['Iris-versicolor', 'Iris-virginica', 'Iris-setosa'],
#                     ordered=True)
# df.Name = df.Name.astype(iris_ord_cat)
# df.sort_values(by='Name')
# ```

# %%
iris_ord_cat = pd.CategoricalDtype(
                    categories=['Iris-versicolor', 'Iris-virginica', 'Iris-setosa'],
                    ordered=True)
iris_ord_cat

# %% {"tags": ["level_intermediate"]}
df.Name = df.Name.astype(iris_ord_cat)

# %% {"tags": ["level_intermediate"]}
df.sort_values(by='Name').head(4)

# %% [markdown]
# ***
