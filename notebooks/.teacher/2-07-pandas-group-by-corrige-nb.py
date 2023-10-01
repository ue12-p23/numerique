# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     cell_metadata_json: true
#     notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version,-language_info.version, -language_info.codemirror_mode.version,
#       -language_info.codemirror_mode,-language_info.file_extension, -language_info.mimetype,
#       -toc, -rise, -version
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
#     title: "regrouper par crit\xE8res"
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(filename="_static/style.html")

# %% [markdown]
# # regrouper par critères

# %% [markdown]
# ## les données et les librairies

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('titanic.csv', index_col=0)
df.head(3)

# %% [markdown] {"tags": ["framed_cell"]}
# ## introduction
#
# ````{admonition} →
# en `pandas`, une table de données (encore appelée *dataframe*) a uniquement 2 dimensions
#
# mais elle peut indiquer, avec ces deux seules dimensions, des sous-divisions dans les données
#
# les passagers du Titanic sont ainsi divisés
#
# * en homme/femme par la colonne `Sex`
# * en passagers de première, seconde ou troisième classe par la colonne `Pclass`
# * en survivants ou décédés par la colonne `Survived`
# * on pourrait même les diviser en classe d'âge par la colonne `Age`  
#    *enfants* (avant 12 ans), *jeunes* (entre 12 et 20), *adultes* (entre 20 et 60), *personne agées* (+ de 60 ans)
#
# des analyses mettant en exergue ces groupes de personnes peuvent être intéressantes
#
# lors du naufrage du Titanic, valait-il mieux être une femme en première classe ou un enfant en troisième ?
#
# on va calculer des regroupements de lignes (des partitions de la dataframe)  
# en utilisant la méthode `pandas.DataFrame.groupby`  
# à laquelle on indique un ou plusieurs critères.
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ## groupement par critère unique
#
# ````{admonition} →
# le groupement (la partition) se fait par la méthode `pandas.DataFrame.groupby`
#
# prenons le seul critère de genre des passagers  
# de la colonne `Sex`
#
# la colonne a deux valeurs: `female` et `male`
#
# ```python
# df['Sex'].unique()
# -> array(['male', 'female'], dtype=object)
# ```
#
# avec `groupby` `pandas` permet de partitionner la dataframe  
# en autant de sous-dataframes que de valeurs uniques dans la colonne
#
# faisons la partition de notre dataframe en
#
# * la sous-dataframe des hommes i.e. `male`
# * la sous-dataframe des femmes i.e. `female`
# * nous pourrons alors procéder à des analyses différenciées par genre
#
# partition par (`by`) l'unique colonne `Sex`  
# ```python
# by_sex = df.groupby(by='Sex')
# ```
#
# l'objet rendu par la méthode est de type `pandas.DataFrameGroupBy`
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ### accès aux sous-dataframes
#
# ````{admonition} →
# la méthode `pandas.DataFrameGroupBy.size`  
# donne la taille des deux partitions  
# (dans un objet de type `pandas.Series`)
#
# ```python
# by_sex.size()
# -> Sex
# female    314
# male      577
# dtype: int64
# ```
#
# l'objet `pandas.DataFrameGroupBy` est un objet **itérable**  
# qui vous donne les couples `key, dataframe`
#
# ```python
# for group, subdf in by_sex:
#     print(group, subdf.shape) # subdf est de type pandas.DataFrame
#
# -> female (314, 11)
#    male (577, 11)
# ```
#
# vous pouvez donc facilement parcourir toutes les sous-dataframes
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ### proxying : propagation de fonctions sur les sous-dataframes
#
# ````{admonition} →
# itérer est intéressant d'un point de vue pédagogique  
# pour bien comprendre la nature d'un objet `DataFrameGroupBy`  
# et éventuellement inspecter son contenu de visu  
#
# mais en pratique, on peut souvent utiliser une méthode des dataframes  
# **directement** sur l'objet `DataFrameGroupBy` et il est rarement  
# nécessaire d'itérer explicitement dessus  
# (on n'aime pas avoir à écrire un for-Python)
#
# dans ce cas l'objet `DataFrameGroupBy` se comporte comme un *proxy*:
# - il propage le traitement à ses différents morceaux  
# - et s'arrange pour combiner les résultats
#
#
#
# par exemple on peut extraire une colonne sur toutes les sous-dataframe  
# en utilisant la syntaxe `group[colonne]`, et faire des traitements sur le résultat
#
# ```python
# # quel age ont le plus vieil homme et la plus vieille femme
# by_sex['Age'].max()
# # ou encore
# by_sex.Age.max()
#
# # on remarque qu'on peut traiter un groupby comme une dataframe
# # ce qui a l'effet d'appliquer l'opération (ici ['Age'])
# # à toutes les sous-dataframegroupby comme une dataframe
# ```
#
# ou encore on peut fabriquer une dataframe qui contient les sommes
# de certaines colonnes de départ, mais par sexe
#
# ```python
# # les sommes des colonnes 'Survived' et 'Fare', mais par sexe
# by_sex[['Survived', 'Fare']].sum()
# ```
#
# ```{note}
# on peut spécifier plusieurs fonctions d'agrégation
# ```python
# # on regroupe les passagers par classe et genre (5 dataframes)
# # et on calcule plusieurs fonctions d'agrégation
# df.groupby('Sex')['Age'].agg(['max', 'min', 'count', 'median', 'mean'])
# ->
#         max    min   count   median    mean
# Sex                   
# female  63.0   0.75  261     27.0      27.915709
# male    80.0   0.42  453     s9.0      30.726645
# ```
#
#
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ```{exercise} imputation de valeurs manquantes - l'exercice peut se faire plus tard dans le cours ou à-la-maison
# Nous allons remplacer dans la dataframe du Titanic, les ages manquants par la moyenne des ages des classes.
#
# 1. utiliser la méthode `groupby` pour calculer les moyennes des ages par classe
# 1. utiliser la méthode `to_dict` pour en déduire le dictionnaire  
# *qui fait correspondre à chaque classe l'age moyen des passagers de cette classe*
# 1. construisez le masque des passagers dont l'age manque *indice `isna`*  
# vérifiez qu'il y en a bien 117
# 1. appliquez ce masque sur la dataframe et sélectionnez la colonne 'Pclass'  
# vous obtenez la sous-série réduite aux classes des passagers sans age
# 1. utiliser la méthode `replace` pour remplacer, dans cette sous-série des classes des passagers sans age  
# la class par l'age moyen (`replace` peut prendre en argument un dictionnaire)
# 1. localisez dans la dataframe, la colonne 'Age' pour les passagers dont l'age manque  
# *i.e. appliquez le masque en sélectionnant la colonne 'Age'*  
# et modifier cette sous-série avec la série des ages moyens pour ces passagers
# 1. vérifiez que la colonne 'Age' ne contient plus de valeurs manquantes
# ```

# %%
# prune-cell

# on relit la dataframe
df = pd.read_csv('titanic.csv', index_col='PassengerId')

# 1. utiliser la méthode groupby pour calculer les moyennes des ages par classe
mean_age_per_class = df.groupby(by='Pclass')['Age'].mean()

# 2. utiliser la méthode to_dict pour en déduire le dictionnaire
# qui fait correspondre à chaque classe l'age moyen des passagers de cette classe
d = mean_age_per_class.to_dict()
print(d)

# 3. construisez le masque des passagers dont l'age manque *indice `isna`*  
# vérifiez qu'il y en a bien 117
mask = df['Age'].isna()
print(mask.sum() == 177)

# 4. appliquez ce masque sur la colonne 'Pclass'  
# pour obtenir la sous-série réduite aux classes des passagers sans age  
s = df.loc[mask, 'Pclass']

# 5. utiliser replace pour remplacer, dans cette séries des cllases des passagers sans age
# la class par l'age moyen (replace peut prendre en argument un dictionnaire)
s.replace(d, inplace=True)

# 6. localisez dans la dataframe, la colonne 'Age' pour les passagers dont l'age manque
# i.e. appliquez le masque en sélectionnant la colonne 'Age'
# et modifier cette sous-série avec la série des ages moyens pour ces passagers

df.loc[mask, 'Age'] = s

# 7.vérifiez que la colonne 'Age' ne contient plus de valeurs manquantes
df['Age'].notna().all()

# %% [markdown] {"tags": ["framed_cell"]}
# ### accéder à un groupe
#
# ````{admonition} →
# on a parfois besoin d'accéder à un groupe précis dans la partition  
# c'est possible avec la méthode `get_group()`  
# qui retourne une dataframe
#
# ```python
# by_sex.get_group('female')
# ```
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ## groupement multi-critères
#
# ````{admonition} →
# pour des partitions multi-critères  
# passez à `pandas.DataFrame.groupby` une **liste des colonnes**
#
# la méthode `pandas.DataFrame.groupby`
#
# * calcule les valeurs distinctes de chaque colonne (comme dans le cas du critère unique)
# * mais ensuite il en fait le **produit cartésien**
# * on obtient ainsi les clés des groupes sous la forme de tuples
#
# prenons les critères `Pclass` et`Sex`
#
# * le premier critère a trois valeurs `1`, `2` et `3` (pour les trois classes de cabines)
# * le second a 2 valeurs `female` et `male`
#
# on s'attend donc aux 6 clés  
# `(1, 'female')`, `(1, 'male')`  
# `(2, 'female')` `(2, 'male')`  
# `(3, 'female')` `(3, 'male')`  
# (ou du moins à un sous-ensemble de ces 6 clés)
#
# on regroupe
#
# ```python
# by_class_sex = df.groupby(['Pclass', 'Sex'])
# ```
#
# utilisons `size()` pour voir les clés du groupement  
# ici tous les cas du produit cartésien sont représentés
#
# ```python
# by_class_sex.size()
# ->
# Pclass  Sex
# 1       female     94
#         male      122
# 2       female     76
#         male      108
# 3       female    144
#         male      347
# dtype: int64
# ```
#
# nous découvrons là une `pandas.Series` avec un **`index` composé**  
# qu'en pandas on appelle **un *MultiIndex***
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ### multi-index pour les multi-critères
#
# ````{admonition} →
# inspectons de plus près l'index qui est en jeu ici  
# partons du résultat de `by_class_sex.size()` qui est une `pandas.Series`
#
# ```python
# type(by_class_sex.size())
# -> pandas.core.series.Series
# ```
#
# son `index` est un `MultiIndex`
#
# ```python
# df_by_class_sex.size().index
# ->
# MultiIndex([(1, 'female'),
#             (1,   'male'),
#             (2, 'female'),
#             (2,   'male'),
#             (3, 'female'),
#             (3,   'male')],
#            names=['Pclass', 'Sex'])
#
# ```
#
# les index sont les tuples du produit cartésien  
# on aurait pu aussi les calculer par une compréhension Python comme ceci
# ```python
# {(i, j) for i in df['Pclass'].unique() for j in df['Sex'].unique()}
# ->
# {(3, 'male'),
#  (3, 'female'),
#  (1, 'male'),
#  (1, 'female'),
#  (2, 'male'),
#  (2, 'female')}
# ```
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ### les éléments de l'index sont des tuples
#
# ````{admonition} →
# les éléments dans le `MultiIndex` sont des tuples Python
#
# par exemple, nous pouvons toujours itérer sur les sous-dataframes  
# de la partition, sauf qu'ici ce qui décrit le groupe, c'est un 2-tuple  
# donc on adapterait l'itération sur ce groupby multi-critère  
# comme ceci
#
# ```python
# for (class_, sex), subdf in by_class_sex:
#     print(f"there were {len(subdf)} {sex} in class {class_} ")
#
# there were 94 female in class 1
# there were 122 male in class 1
# there were 76 female in class 2
# there were 108 male in class 2
# there were 144 female in class 3
# there were 347 male in class 3
# ```
#
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ### display de `head()` avec IPython
#
# ````{admonition} →
# on veut afficher les 2 premières lignes de chaque dataframe de la partition
#
# utiliser la méthode `head()` avec `print` n'est pas aussi joli  
# que l'affichage de la dernière expression de la cellule
#
# ```python
# for group, subdf in by_class_sex:
#     print(group, subdf.head(1))
# ```
#
# pour retrouver la même qualité d'affichage (en html)  
# il faut utiliser la méthode `IPython.display.display()`  
# en important la librairie `IPython`
#
# ```python
# import IPython
# for group, subdf in by_class_sex:
#     print(group)
#     IPython.display.display(subdf.head(1))
# ```
#
# ```{note}
# les lignes apparaissent dans l'ordre de l'index
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ## accès au dictionnaire des groupes
#
# ````{admonition} →
# l'attribut `pandas.DataFrameGroupBy.groups`  
# est un dictionnaire qui décrit les partitions:  
# - la clé correspondent à un groupe  
# - et la valeur est une **liste des index** des lignes du groupe
#
# ```python
# by_sex = df.groupby(by='Sex')
# by_sex.groups
#     ->
# {'female': [499, 395, 703, 859, ...], 'male': [552, 638, 261, 811, ...]}
# ```
#
# on peut utiliser cette information pour inspecter plus finement  
# le contenu du groupby  
#
# par exemple pour afficher les noms des 3 premiers membres de chaque groupe
#
# ```python
# for group, indexes in by_sex.groups.items():
#     print(group, df.loc[indexes[:3], 'Name'])
# ```
# ````

# %%
# on se remet dans le contexte
df = pd.read_csv('titanic.csv', index_col=0)
by_sex = df.groupby(by='Sex')

# %% [markdown]
# ## groupby avec apply et transform

# %% [markdown] {"tags": ["framed_cell"]}
# ````{admonition}  →
#
# on peut appliquer une fonction aux sous-dataframes du groupby
#
# par exemple, nous voulons standardiser les colonnes des ages (enlever la moyenne et diviser par l'écart type)
#
# ```python
# standardization = lambda x: (x - x.mean()) / x.std()
# ```
#
# et nous voulons le faire en différenciant les femmes et les hommes  
#
#
# pour cela nous allons faire un `groupby` sur le genre des passagers  
# et appliquer la standardisation aux sous-dataframes
#
# ensuite, on peut
# 1. avec `apply` obtenir les résultats dans une série avec son multi-index
#    ```python
#    df.groupby(by='Sex')['Age'].apply(standardization)
#    ->
#         Sex     PassengerId
#         female  499           -0.206639
#                 395           -0.277510
#                                  ...   
#         male    396           -0.594531
#                 832           -2.036806
#         Name: Age, Length: 891, dtype: float64
#    ```
# 1. avec `transform` recombiner les résultats  en une seule colonne contenant tous les passagers  
# (par exemple pour ajouter cette colonne à la dataframe)  
#    ```python
#    df.groupby(by='Sex')['Age'].transform(standardization)
#    ->
#           PassengerId
#         552   -0.253890
#         638    0.018623
#         499   -0.206639
#         Name: Age, Length: 891, dtype: float64
#    ```
#
# ```{note}
# 1. `apply` sur une dataframe, applique la fonction le long d'un axis
#    ```python
#    df.loc[df['Sex'] == 'female', ['Age']].apply(lambda x: (x - x.mean()) / x.std())
#    # calcule mean et std sur la colonne des ages des femmes
#    ->
#          Age
#      PassengerId	
#      499   -0.206639
#      395   -0.277510
#      703   -0.702736
#      ...
#      314 rows × 1 columns
#    
# 1. `apply` sur une série, applique la fonction élément par élément
#    ```python
#    df.loc[df['Sex'] == 'female', 'Age'].apply(lambda x: x*365.25)
#    # calcule l'age en jours élément par élément
#    ->
#    PassengerId
#    499     9131.25
#    395     8766.00
#    703     6574.50
#    ...
#    Name: Age, Length: 314, dtype: float64
#    ```
#
# ````

# %% [markdown]
# ## intervalles de valeurs d'une colonne

# %% [markdown] {"tags": ["framed_cell"]}
# ###  introduction
#
# ````{admonition} →
# parfois il y a trop de valeurs différentes dans une colonne  
# du coup on veut faire un découpage de ces valeurs en intervalles
#
# par exemple dans la colonne des `Age`  
#
# * si nous faisons un groupement brutal sur cette colonne  
# comme nous avons 88 âges différents  
# cela ne donne pas d'information intéressante
#
# * mais ce serait intéressant de raisonner par **classes** d'âges par exemple
#    - *'enfant'* jusqu'à 12 ans
#    - *'jeune'* entre 12 ans (exclus) et 19 ans (inclus)
#    - *'adulte'* entre 19 (exclus) et 55 ans (inclus)
#    - *'+55'*  les personnes de strictement plus de 55 ans  
#
# afin de compléter la colonne des ages  
# `pandas` propose la fonction `pandas.cut`
#
# nous allons voir un exemple
#
# ```python
# pd.cut?
# ```
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ###  découpage en intervalles d'une colonne
#
# ````{admonition} →
# avec `pandas.cut` nous allons créer dans notre dataframe  
# une nouvelle colonne qui contient les intervalles d'ages  
# `(0, 12]`, `(12, 19]`, `(19, 55]` et  `(55, 100]`
#
# `pandas.cut`
#
# * s'applique à une colonne de votre dataframe
# * vous devez précisez les bornes de vos intervalles avec le paramètre `bins`  
# * les bornes min des intervalles seront exclues  
# * la fonction retourne une nouvelle colonne
#
# ```python
# pd.cut(df['Age'], bins=[0, 12, 19, 55, 100])
# ->
# PassengerId
# 552    (19.0, 55.0]
# 638    (19.0, 55.0]
# 499    (19.0, 55.0]
# 261             NaN   <- age inconnu au départ
# 395    (19.0, 55.0]
#            ...
# 326    (19.0, 55.0]
# 396    (19.0, 55.0]
# 832     (0.0, 12.0]
# Name: Age, Length: 891, dtype: category
# Categories (4, interval[int64, right]): [(0, 12] < (12, 19] < (19, 55] < (55, 100]]
# ```
#
# remarquez  
#
# * on doit donner toutes les bornes des intervalles  
#   (les bornes se comportent comme des poteaux  
#   ici 5 bornes produisent 4 intervalles)  
#
# * les bornes min des intervalles sont bien exclues
# * la colonne est de type `category` (cette catégorie est ordonnée)
# * des labels sont générés par défaut
# * les items en dehors des bornes sont transformés en `nan`
#
# vous pouvez donner des labels aux intervalles avec le paramètre `labels`
#
# ```python
# pd.cut(df['Age'],
#        bins=[0, 12, 19, 55, 100],
#        labels=['children', ' young', 'adult', '55+'])
# ```
#
# souvent on va ranger cette information dans une nouvelle colonne  
# et ça on sait déjà comment le faire
# ```python
# df['Age-class'] = pd.cut(
#     df['Age'],
#     bins=[0, 12, 19, 55, 100],
#     labels=['children', ' young', 'adult', '55+'])
# ```
#
# comment feriez-vous pour inspecter le type (des valeurs) de cette colonne ?  
# est-ce un type ordonné ?
#
# **révision**  
# comment feriez-vous pour vous débarrasser maintenant de la colonne `Age` dans la dataframe
#
# ````

# %%
# prune-cell

df['Age-class'] = pd.cut(
    df['Age'],
    bins=[0, 12, 19, 55, 100],
    labels=['children', ' young', 'adult', '55+'])
df['Age-class'].unique()

# %%
# prune-cell

# pour effacer la colonne 'Age'
print("avant", df.columns)
del df['Age']
print("après", df.columns)
# on peut utiliser aussi df.drop
# df.drop('Age', axis=1, inplace=True)

# %% [markdown] {"tags": ["framed_cell"]}
# ###  groupement avec ces intervalles
#
# ````{admonition} →
# nous avons la colonne `Age-classes`
#
# comme c'est un type catégorie, vous pouvez utiliser cette colonne dans un `groupby`
#
# ```python
# df.groupby(['Age-class', 'Survived', ])
# ```
#
# vous avez désormais  
# une idée de l'utilisation de `groupby`  
# pour des recherches multi-critères sur une table de données
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ```{exercise} calculez taux de survie par classe d'age par classes de cabines
# 1. pour les avancés faire l'exercice sans lire la cellule suivante
# 1. pour les moins avancés, allez à la cellule suivante où l'exercice est posé pas-à-pas
# ```

# %% {"scrolled": true}
# prune-cell

# calculez taux de survie par classe d'age par classes de cabines)
df.groupby(['Age-class', 'Pclass'])['Survived'].mean()

# %% [markdown] {"tags": ["framed_cell"]}
# ```{exercise} calcul pas-à-pas du taux de survie par classe d'age par classes de cabines
# 1. utiliser un `groupby` pour regrouper les passagers par classe d'age et par classe de cabines  
# *vous obtenez les 12 sous-dataframes `[('children', 1), [('children', 2)...['55+', 3)]`*   
#  utilisez `size` pour voir comment sont répartis les passagers dans les sous-dataframes  
#  vous voyez 4 enfants en première classe
#
# 1. utilisez `get_group` pour accéder à la dataframe des enfants de première classe  
# vous voyez les lignes des 4 enfants de première classe
# 1. sélectionnez, dans les 12 sous-dataframes, leur colonne `'Survived'`
# 1. en faisant la moyenne de cette colonne dans chaque sous-dataframe  
# vous avez bien le taux de survie des passagers pour les 12 sous-dataframes
# 1. à la place de `mean` utilisez `value_counts` à laquelle vous passez le paramètre `normalize=True`  
# les taux sont donnés pour la survie et 1-survie
# 1. faites le en une seule ligne par classe-d'age, classe de cabine et sexe
# ````

# %%
# prune-cell

# 1. utiliser un groupby pour regrouper les passagers par classe d'age et par classe de cabines
df.groupby(['Age-class', 'Pclass' ]).size()

# 2. utilisez get_group pour accéder à la dataframe des enfants de première classe
df.groupby(['Age-class', 'Pclass' ]).get_group(('children', 2))

# 3. et 4. sélectionnez, dans les 12 sous-dataframes, leur colonne 'Survived'
# en faisant la moyenne de cette colonne dans chaque sous-dataframe
# vous avez bien le taux de survie des passagers pour les 12 sous-dataframes
df.groupby(['Age-class', 'Pclass' ])['Survived'].mean()

# 5. à la place de mean utilisez value_counts à laquelle vous passer le paramètre normalize=True
df.groupby(['Age-class', 'Pclass' ])['Survived'].value_counts(normalize=True)

# 6. avec le genre
df.groupby(['Age-class', 'Pclass', 'Sex' ])['Survived'].mean()


# %% [markdown] {"jp-MarkdownHeadingCollapsed": true, "tags": ["framed_cell"]}
# ````{exercise} les partitions avec groupby
#
# On veut calculer la partition de la dataframe du Titanic avec, dans cet ordre, la classe `Pclass`, le genre `Sex`, et l'état de survie `Survived`
#
# 1. sans calculer la partition, proposez une manière de calculez le nombre probable de sous parties dans la partition  
# 1. calculez la partition avec `pandas.DataFrame.groupby` et affichez les nombres d'items par groupe
# 1. affichez la dataframe des femmes de première classe qui n'ont pas survécu
# 1. faites la même extraction sans utiliser un `groupby()` mais les conditions
# 1. créez un `dict` avec les taux de survie par genre dans chaque classe
# 1. à partir de ce `dict`, créez une `pandas.Series`  
#    avec comme nom `'taux de survie par genre dans chaque classe'`  
# ````
#
#

# %%
# prune-cell

df = pd.read_csv('titanic.csv').set_index('PassengerId')

# 1. sans calculer la partition proposez une manière de calculez le nombre probable de sous parties dans la partition  
len(df['Sex'].unique()) * len(df['Pclass'].unique()) * len(df['Survived'].unique())

# 2. calculez la partition avec `pandas.DataFrame.groupby`
# et affichez les nombres d'items par groupe

groups = df.groupby(['Pclass', 'Sex', 'Survived'])
groups.size()

# 3. affichez la dataframe des entrées pour les femmes qui ont péri et qui voyagaient en 1ère classe
groups.get_group((1, 'female', 0))

# 4.refaites la même extraction sans utiliser un groupby en utilisant les conditions
mask = (df.Pclass == 1) & (df.Sex == 'female') & ~df.Survived
df[mask]

# 5. créez un dict avec les taux de survie par genre dans chaque classe
D = df.groupby(['Sex', 'Pclass'])['Survived'].mean().to_dict()

# 6. à partir de ce dict, créez une pandas.Series
# de nom 'taux de survie par genre dans chaque classe'
pd.Series(D, name="taux de survie par genre dans chaque classe")

# %% [markdown] {"tags": ["framed_cell"]}
# ## pour en savoir plus
#
# pour les avancés
#
# on recommande la lecture de cet article dans la documentation `pandas`, qui approfondit le sujet et notamment la notion de `split-apply-combine`
#
# (qui rappelle, de loin, la notion de *map-reduce*)
#
# <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html>

# %% [markdown]
# ## `pivot_table`

# %% [markdown] {"tags": ["framed_cell"]}
# ````{admonition} → introduction
# le type d'opérations que l'on a fait dans ce notebook est fréquent  
# spécifiquement, on veut souvent afficher:
#
# * une valeur (précisément, une aggrégation des valeurs) d'une colonne  
# * en fonction de deux autres colonnes (catégorielles)    
# * qui sont utilisées dans les directions horizontale et verticale  
#   (une colonne sera en index et l'autre en columns)
#
# cette dataframe du titanic:  
# <img src="media/titanic-sex-class-survived.png" width=50px>
#
# on voudrait la visualiser comme ceci
#
# * le taux de survie (la valeur à agréger)  
# * par classe de cabine (l'index des lignes)
# * et par genre (les colonnes)
# <img src="media/pivot-titanic.png" width=200px>
#
# il existe une méthode `pivot_table()` qui s'avère très pratique  
# pour faire ce genre de traitement **en un seul appel**  
# comme toujours, pensez à lire la doc avec `df.pivot_table?`
#
# les paramètres les plus importants sont
#
# * `values` : la (ou les) colonne(s) qu'on veut regarder  
#   ce seront les valeurs **dans le tableau**
#
# * `index` : la (ou les) colonne(s) utilisée(s) pour **les lignes** du résultat
# * `columns` : idem pour **les colonnes**
# * `aggfunc` : la fonction d'aggrégation à utiliser sur les `values`  
#   il y a toujours plusieurs valeurs qui tombent dans une case du résultat  
#   il faut les agréger; par défaut on fait **la moyenne**  
#   (ce qui convient bien avec 'Survived')
#
# ainsi la table ci-dessus s'obtient **tout simplement** comme ceci
#
# ```python
# df.pivot_table(
#     values='Survived',
#     index='Pclass',
#     columns='Sex',
# )
# ```
# ````

# %% [markdown] {"tags": ["framed_cell"]}
# ### `pivot_table()` et agrégation
#
# ````{admonition} →
# dans le cas présent on n'a **pas précisé** la fonction d'**aggrégation**  
# du coup c'est la moyenne qui est utilisée, sur la valeur de `Survived`  
# qui vaut 0 ou 1 selon les cas, et donc on obtient le taux de survie  
#
# ```{exercise}
# obtenez la même table que ci-dessus avec cette fois le nombre de survivants
# ```
#
# ````

# %%
# prune-cell

df.pivot_table(
    values='Survived',
    index='Pclass',
    columns='Sex',
    aggfunc='sum',
)

# %% [markdown] {"tags": ["framed_cell"]}
# ### `pivot_table()` et multi-index
#
# ````{admonition} →
# comme on l'a vu, il est possible de passer aux 3 paramètres  
# `values`, `index` et `columns` des **listes** de colonnes
#
# le résultat dans ce cas utilise un `MultiIndex`  
# pour en quelque sorte "ajouter une dimension"  
# dans l'axe des x ou des y, selon les cas
#
# ```{exercise} ajout de dimensions  
# dans la pivot_table avec le taux de survie pour par `'Pclass'` et `'Sex'`  
# observez les résultats obtenus en ajoutant dans chacune des dimensions:
# 1. comme valeur supplémentaire `Age`
# 1. comme critère supplémentaire `Embarked`
#    1. en index
#    1. en colonne
# 1. que pouvez-vous dire des index (en lignes et en colonnes)  
# du résultat produit par `pivot_table()`
#
# ```
#
# ````

# %% {"cell_style": "center"}
# prune-cell

# relisons depuis le fichier pour être sûr d'avoir la colonne 'Age'
df = pd.read_csv('titanic.csv')

# 1. comme valeur supplémentaire Age
df1 = df.pivot_table(
    values=['Survived', 'Age'],
    index='Pclass',
    columns='Sex',
)

# 2. comme critère supplémentaire Embarked
# 2.1 en index
df2 = df.pivot_table(
    values='Age',
    index=['Pclass', 'Embarked'],
    columns='Sex', 
)

# 2. comme critère supplémentaire Embarked
# 2.1 en colonne
df3 = df.pivot_table(
    values='Age',
    index='Pclass',
    columns=['Sex', 'Embarked'],
)

# %% [markdown] {"jp-MarkdownHeadingCollapsed": true, "tags": ["framed_cell"]}
# ````{exercise} exercice avec des pivot_table
#
# 1. Lisez le fichier `wine.csv`
# 1. Affichez les valeurs min, max, et moyenne, de la colonne 'magnesium'
# 1. définissez deux catégories selon que le magnesium est en dessous ou au-dessus de la moyenne  
# (qu'on appelle 'mag-low' et 'mag-high'); rangez le résultat dans une colonne 'mag-cat'
# 1. calculez cette table
#
# <img src='media/pivot-table-expected.png' width="50">
#
#
# ````

# %%
# prune-cell

# 1. affichez les valeurs min, max, et moyenne, de la colonne 'magnesium'
df = pd.read_csv('wine.csv')
# df.head(2)

# 2. Affichez les valeurs min, max, et moyenne, de la colonne 'magnesium'
summary = df['magnesium'].describe()[['min', 'max', 'mean']]

# 3. définissez deux catégories selon que le magnesium est en dessous ou au-dessus de la moyenne
# 'mag-low' et 'mag-high'
# rangez le résultat dans une colonne 'mag-cat'
df['mag-cat'] = pd.cut(
    df.magnesium,
    bins=[summary['min'], summary['mean'], summary['max']],
    labels=('mag-low', 'mag-high'))

# 4 calculez cette table
df.pivot_table(values=('color-intensity', 'flavanoids', 'magnesium'),
               index='cultivator',
               columns='mag-cat',
               # aggfunc='mean', # by default
              )

# %% [markdown]
# ## `stack` et  `unstack`

# %% [markdown] {"tags": ["framed_cell"]}
# ````{admonition}  introduction
#
# il est parfois utile de savoir changer la structure de votre dataframe  
#
# 1. `stack` permet de faire passer un niveau de colonne en ligne
# 1. `unstack` (opération inverse) permet de faire passer un niveau de lignes dans les colonnes
#
# ```python
# # lisons quelques lignes et colonnes du Titanic sans valeurs manquantes et sauvons le fichier sur l'ordinateur
# df = pd.read_csv('titanic.csv',
#                  usecols=['PassengerId', 'Pclass', 'Sex', 'Age', 'Pclass'],
#                  index_col='PassengerId', 
#                  nrows=6).dropna()
# df.sort_index().to_csv('simple-titanic.csv')
# ```
#
# ```python
# # lisons le fichier
# df = pd.read_csv('simple-titanic.csv', index_col='PassengerId')
# df
# ->
#              Pclass  Sex     Age
# PassengerId                        
#         395  3       female  24.0
#         499  1       female  25.0
#         552  2       male    27.0
#         638  2       male    31.0
#         811  3       male    26.0
# ```
#
# ```python
# # on peut ranger toutes les colonnes en index de ligne 
# df.stack()
# ->
# PassengerId        
# 395          Pclass         3
#              Sex       female
#              Age         24.0
# 499          Pclass         1
#              Sex       female
#              Age         25.0
# 552          Pclass         2
#              Sex         male
#              Age         27.0
# 638          Pclass         2
#              Sex         male
#              Age         31.0
# 811          Pclass         3
#              Sex         male
#              Age         26.0
# dtype: object
# ```
#
# ```python
# # on remarque: i) le multi-index des lignes
# #              ii) les deux niveaux du multi-index (0 et 1)
# df.stack().index
# ->
# MultiIndex([(395, 'Pclass'),
#             (395,    'Sex'),
#             (395,    'Age'),
#             (499, 'Pclass'),
#             (499,    'Sex'),
#             (499,    'Age'),
#             (552, 'Pclass'),
#             (552,    'Sex'),
#             (552,    'Age'),
#             (638, 'Pclass'),
#             (638,    'Sex'),
#             (638,    'Age'),
#             (811, 'Pclass'),
#             (811,    'Sex'),
#             (811,    'Age')],
#            names=['PassengerId', None])
# ```
#
#
# ```python
# # l'index des PassengerId est le niveau 0
# # on peut le faire monter dans les colonnes (le dépiler)
# df.stack().unstack(0)
# ->
#       PassengerId  552   638   499     395     811
# Survived           0     0     0       1       0
# Pclass             2     2     1       3       3
# Sex                male  male  female  female  male
# Age                27.0  31.0  25.0    24.0    26.0
# ```
#
# ```python
# # si on dépile le niveau 1, on dépile les colonnes Survived Pclass Sex et Age
# df.stack().unstack(1)
# # c'est la dataframe de d´epart
# ```
#
# ````
