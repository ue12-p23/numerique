# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version,-language_info.version, -language_info.codemirror_mode.version,
#       -language_info.codemirror_mode,-language_info.file_extension, -language_info.mimetype,
#       -toc
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
#     title: conditions et masques
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %% scrolled=true
from IPython.display import HTML
HTML(filename="_static/style.html");

# %% [markdown]
# # conditions et masques

# %%
import pandas as pd
import numpy as np

# %% [markdown] tags=["framed_cell"]
# ## conditions sur une dataframe
#
# ````{admonition} →
# dans les analyses de tables de données  
# il est fréquent de **sélectionner des données par des conditions**
#
# les conditions peuvent s'appliquer, selon le contexte
#
# * à tout un tableau
# * ou à toute une colonne
# * ou à toute une ligne
# * ou à tout un sous-tableau, sous-colonne, sous-ligne
#
# en `pandas`, comme en `numpy`, les fonctions sont **vectorisées**  
# par souci de rapidité du code
#
# il ne faut **jamais itérer avec un `for-python`** sur les valeurs d'une table  
# (les itérations se font dans le code des fonctions `numpy` et `pandas`)
#
# comme en `numpy`, une expression conditionnelle va s'appliquer à toute la structure
# et retourner une structure du même type  
# où seul le type des valeurs à changé puisque les conditions retournent des booléens `True` et `False`
#
# exemple:
#
# * `titanic['Age']` : un objet de type `Series` à valeurs entières
# * `titanic['Age'] > 12` : un objet de type `Series` à valeurs booléennes
#
# (voir ci-dessous)
#
# ````

# %% [markdown] slideshow={"slide_type": "slide"} tags=["framed_cell"] jp-MarkdownHeadingCollapsed=true
# ## conditions et masques
#
# ````{admonition} →
# regardons cet exemple en détail:  
# quels passagers avaient moins de 12 ans ?
#
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
#
# children = df['Age'] < 12 # l'opérateur < est vectorisé 
# children
#
# -> PassengerId
#     552    False  # <- le passager de PassengerId 552 a plus de 12 ans
#     638    False
#            ...
#     326    False
#     396    False
#     832     True  # <- celui-ci par contre a strictement moins de 12 ans
#     Name: Age, Length: 891, dtype: bool
# ```
#
# cette expression retourne des **booléens** - appelé **un masque**  
# dans une `pandas.Series` dont le type est naturellement `bool`  
# avec, pour chaque valeur de la colonne, la réponse au test
#
# en `pandas` comme en `numpy` pour combiner les conditions  
#
# * on utilise `&` (et) `|` (ou) et `~` (non)  
# ou les `numpy.logical_and`, `numpy.logical_or`, `numpy.logical_not`
#
# * et **surtout pas** `and`, `or` et `not` (opérateurs `Python` non vectorisés)
# * on **parenthèse toujours** les expressions
#
# ```python
# girls = (df['Age'] < 12) & (df['Sex'] == 'female')
# girls.sum()
# -> 32
# ```
#
# ```{attention}
#
# c'est **très important** de bien mettre des parenthéses  
# car les opérateurs bitwise (`&` et autres) ont des **précédences** (priorités)  
# qui sont non intuitives, et très différentes des opérateurs logiques (`and` et autres)
# ```
#
# on pourra ensuite utiliser ces tableaux de booléens  
#
# * pour leur appliquer des fonctions (comme `sum`)  
# * ou comme des masques pour sélectionner des sous-tableaux
#
#
# ````

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["framed_cell"]
# ## Intérêt de la vectorization
#
# ````{admonition} →
# :class: dropdown
#
# Calculez les temps d'exécution de ces deux codes.  
# Quel est le plus rapide ?
#
# ```python
# %%timeit
# df['Age'] < 12
# ```
#
# ou  
# ```python
# %%timeit
# for i in df.index:
#    df['Age'][i] < 12
# ```
#
# ```{exercise} vectorisation d'une fonction
#
# 1. En utilisant une fonction de génération de nombres aléatoires  
# *comme `np.random.randint`, `np.random.randn`...*
#    1. créez une `pd.Series` de valeurs aléatoires, de la taille d'une colonne de la dataframe du Titanic et ayant le même index que la dataframe du Titanic
#    2. ajoutez cette `pd.Series` aux colonnes de la dataframe du Titanic
# 1. Implémentez la fonction `Python` `my_abs(x)` qui renvoie la valeur absolue de `x` sans utiliser la fonction `abs`
# 1. Appliquez la fonction `my_abs` à la colonne que vous avez ajoutée dans la dataframe du Titanic  
# 1. Que se passe-t-il ? Pourquoi ?
# 1. La solution est créer une version vectorisée de votre fonction `my_abs`  
#    1. faites le (en utilisant `np.vectorize`)
#    1. remplacez (dans la dataframe) la colonne par la colonne à laquelle vous appliquez la fonction vectorisée
# ```
#
# ````
# %%
# prune-cell

from IPython import display

# on lit la dataframe du titanic
df_ = pd.read_csv('./titanic.csv').set_index('PassengerId')

# 1.1 créez une pd.Series
s = pd.Series(np.random.randn(len(df_)), index=df_.index)

# 1.2 ajoutez cette pd.Series aux colonnes de la dataframe du Titanic
df_['rand-col-end'] = s

# 1.2-bis
# on peut aussi l'insérer à une position donnée (ici à l'indice 0)
df_.insert(0, 'rand-col', s)
display.display(df_[['rand-col']].head())

# 2. Implémentez la fonction Python my_abs(x) qui renvoie la valeur absolue de x
def my_abs (x):
    if x < 0:
        return -x
    return x

# 3. Appliquez la fonction my_abs à la colonne que vous avez ajoutée dans la dataframe du Titanic
# et 4. Que se passe-t-il ? Pourquoi ?
try:
    my_abs(df_['rand-col'])
except ValueError as e:
    print(f"   {e}")
    # ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()
    print("on veut l'appliquer élément par élément alors qu'elle n'est pas vectorisée")

# 5.1 faites le (en utilisant np.vectorize)
my_vect_abs = np.vectorize(my_abs)

# 5.2 remplacez (dans la dataframe) la colonne par la colonne à laquelle vous appliquez la fonction vectorisée
df_['rand-col'] = my_vect_abs(df_['rand-col'])
df_[['rand-col']].head()

# %%
girls = (df['Age'] < 12) & (df['Sex'] == 'female')
girls.sum()

# %% [markdown] tags=["framed_cell"] slideshow={"slide_type": "slide"}
# ## `value_counts()`
#
# ````{admonition} →
# comment calculer le nombre d'enfants ?  
# par exemple nous pouvons sommer les `True` avec `pandas.Series.sum`
#
# ```python
# children = df['Age'] < 12
# children.sum()
# -> 68
# ```
#
# ou utiliser la méthode `value_counts()`  
# qui compte les occurrences dans une colonne  
#
# ```python
# children = df['Age'] < 12
# children.value_counts()
# -> False    823
#    True      68
#    Name: Age, dtype: int64
# ```
#
# la méthode vous indique la colonne `Age`  
# le type `int64` est le type des quantités
#
# ainsi parmi les passagers dont on connait l'âge  
# `68` passagers,  ont moins de `12` ans  
# on reviendra tout de suite sur les données manquantes
# ````

# %% [markdown]
# ## valeurs manquantes

# %% [markdown] tags=["framed_cell"]
# ### contexte général
#
# ````{admonition} →
# souvent, certaines colonnes ont des valeurs manquantes...  
# dans l'exemple du Titanic, ce sont les valeurs qui ne sont pas renseignées dans le `csv`  
#
# on a souvent besoin de les trouver, les compter, et si nécessaire les éliminer
#
# `NA` signifie Non-Available et `NaN` Not-a-Number
#
# sur les `DataFrame` et les `Series`  
# la méthode `isna()` construit **un masque**  
# du même type (DataFrame ou Series donc),
# et à valeurs booléennes  où
#
# * `True` signifie que la valeur est manquante
# * `False` que la valeur ne l'est pas
#
# il existe son contraire qui est `notna()`  
# il existe aussi des synonymes `isnull()` et `notnull()` - **préférez** `isna`
#
# ````

# %% [markdown] tags=["framed_cell"]
# ### valeurs manquantes dans une colonne
#
# ````{admonition} →
# regardons les valeurs manquantes d'une colonne
#
# ```python
# df['Age'].isna()
# ->  PassengerId
#     552    False
#     638    False
#     499    False
#     261     True
#     395    False
#            ...
#     396    False
#     832    False
#     Name: Age, Length: 891, dtype: bool
# ```
#
# l'age du passager d'`Id` 261 est manquant  
# on peut le vérifier dans le fichier en format `csv`:
#
# ```
# 261,0,3,"Smith, Mr. Thomas",male,,0,0,384461,7.75,,Q
#                                 ^^
# ```
#
# combien d'ages sont manquants ?  
# *on reviendra sur les valeurs manquantes*
#
# ```python
# df['Age'].isna().sum()
# -> 177
# ```
#
# on peut tout aussi bien utiliser le `sum` de `numpy` ou de `Python`
#
# ```python
# np.sum(df['Age'].isna())
# -> 177
# ```
# ```python
# sum(df['Age'].isna())
# -> 177
# ```
#
# ````

# %% [markdown] tags=["framed_cell"]
# ### valeurs manquantes sur une dataframe
#
# ````{admonition} →
# la méthode `isna()` s'applique aussi à une dataframe  
# et elle retourne une **dataframe de booléens** où - sans surprise :  
#
# * `True` signifie que la valeur est manquante
# * `False` que la valeur ne l'est pas
#
# regardons les valeurs manquantes d'une dataframe
#
# ```python
# df.isna()
#
# ->              Survived  Pclass   Name    Sex  ...  Ticket   Fare  Cabin  Embarked
# PassengerId                                  ...
# 552             False   False  False  False  ...   False  False   True     False
# 638             False   False  False  False  ...   False  False   True     False
# 499             False   False  False  False  ...   False  False  False     False
# 261             False   False  False  False  ...   False  False   True     False
# 395             False   False  False  False  ...   False  False  False     False
# ...               ...     ...    ...    ...  ...     ...    ...    ...       ...
# 463             False   False  False  False  ...   False  False  False     False
# 287             False   False  False  False  ...   False  False   True     False
# 326             False   False  False  False  ...   False  False  False     False
# 396             False   False  False  False  ...   False  False   True     False
# 832             False   False  False  False  ...   False  False   True     False
#
# [891 rows x 11 columns]
# ```
#
# vous remarquez une dataframe de la **même taille** que `df`
# ````

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=["framed_cell"]
# ### compter les valeurs manquantes
#
# ````{admonition} →
# comme en `numpy` je peux appliquer une fonction - ici `sum()` - en précisant l'`axis`  
# `0` on applique la fonction dans l'axe des lignes (le défaut)  
# `1` on applique la fonction dans l'axe des colonnes  
# l'objet retourné est une série contenant le résultat de la fonction
#
# exemple avec la somme (`sum`) des valeurs manquantes sur l'axe des lignes `axis=0`  
# qui `sum` les lignes entre elles - le résultat est par colonne donc
#
# ```python
# df.isna().sum()       # les deux formes sont
# df.isna().sum(axis=0) # équivalentes
#
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2
# dtype: int64
# ```
#
# ```{admonition} note
# :class: attention
#
# pour souligner une différence avec `numpy`: comparez le comportement
#
# * de `array.sum()`
# * et `df.sum()`  
# (on y revient ci-dessous)
# ```
#
# nous remarquons des valeurs manquantes dans les colonnes `Cabin`, `Age` et `Embarked`
# ````

# %% [markdown] tags=["framed_cell"]
# ### dans l'autre direction (axis=1)
#
# ````{admonition} →
# exemple de la somme des valeurs manquantes sur l'axe des colonnes
#
# ```python
# df.isna().sum(axis=1)
#
# ->  PassengerId
#     552    1
#     638    1
#     499    0
#     261    2
#     395    0
#           ..
#     463    0
#     287    1
#     326    0
#     396    1
#     832    1
#     Length: 891, dtype: int64
# ```
#
# le passager d'id `261` a deux valeurs manquantes
# ````

# %% [markdown] tags=["framed_cell"] editable=true slideshow={"slide_type": ""}
# ### les fonctions `numpy` d'agrégation
#
# ````{admonition} →
# les méthodes `numpy` d'agrégation (comme `sum` et `mean` et `min`...)  
# s'appliquent sur les `pandas.DataFrame` et les `pandas.Series`
#
# on précise l'`axis`
# * `0` pour l'axe des lignes (le défaut)  
# * `1` pour l'axe des colonnes  
#
# différence avec `numpy`, si on appelle sans préciser `axis`
#
# * avec **numpy**: on obtient le résultat **global**  
# * avec **pandas**: par défaut `axis=0`, on agrège sur l'axe des lignes
#
# **si on désire le résultat global**
# 1. soit on applique la fonction deux fois  
#    e.g. `df.isna().sum().sum()`
# 1. soit on peut passer par le sous-tableau `numpy`  
#   et là la fonction `numpy.sum()` donnera le résultat global
#
# la méthode `pandas.DataFrame.to_numpy` retourne le tableau `numpy.ndarray` de la DataFrame `pandas`
#
# ```python
# df.isna().to_numpy()
# -> array([[False, False, False, ..., False,  True, False],
#           [False, False, False, ..., False, False, False],
#           ...,
#           [False, False, False, ..., False,  True, False],
#           [False, False, False, ..., False,  True, False]])
# ```
#
# on somme
#
# ```python
# np.sum(df.isna().to_numpy())
# df.isna().to_numpy().sum()
# -> 866
# ```
#
# il y a `866` valeurs manquantes dans toute la data-frame
#
# ```{admonition} note
# :class: attention
#
# remarque: contrairement à ce qu'on avait vu en `numpy`, ici on ne pourrait pas faire `df.isna().sum(axis=(0, 1))`
# ```
# ````

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["framed_cell"]
# ````{exercise} valeurs uniques
#
#
# 1. lisez la data-frame du titanic `df`
#
# 2. utilisez la méthode `pd.Series.unique` pour compter le nombre de valeurs uniques  
# des colonnes `'Survived'`, `'Pclass'`, `'Sex'` et `'Embarked'`  
# vous pouvez utiliser un for-python pour parcourir la liste `cols` des noms des colonnes choisies
#
# 3. utilisez l'expression `df[cols]` pour sélectionner la sous-dataframe réduite à ces 4 colonnes  
#    et utilisez l'attribut `dtypes` des `pandas.DataFrame` pour afficher le type de ces 4 colonnes
#
# 4. que constatez-vous ?  
# quel type serait plus approprié pour ces colonnes ?
#
# ````

# %%
# prune-cell

# 1. lisez la data-frame du titanic `df`
df = pd.read_csv("titanic.csv")

# 2. utilisez la méthode `pd.Series.unique` pour compter le nombre de valeurs uniques  
cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
for c in cols:
    print(f"'{c}' unique values {df[c].unique()}")

# 3. utilisez l'expression `df[cols]` pour sélectionner la sous-dataframe réduite à ces 4 colonnes  
minidf = df[cols]
print(f'{minidf.dtypes=}')

for c in minidf.columns:
    print(f"column {c:>12} has type {minidf.dtypes[c]}")

# 4. que constatez-vous ?  
print('la colonne Survived devrait être de type booléen')
print('les trois autres colonnes devraient être de type catégories (nombre fini de valeurs possibles)')

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["framed_cell"]
# ````{exercise} conditions
#
# 1. lisez la data-frame des passagers du titanic
#
# 2. calculez les valeurs manquantes: totales, des colonnes et des lignes
#
# 3. calculez le nombre de classes du bateau
#
# 4. calculez le taux d'hommes et de femmes  
#    *indice: voyez les paramètres optionnels de `Series.value_counts()`*
#
# 5. calculez le taux de personnes entre 20 et 40 ans (bornes comprises)
#
# 6. calculez le taux de survie des passagers
#
# 7. calculez le taux de survie des hommes et des femmes par classes  
# on reverra ces décomptes d'une autre manière
#
# ````

# %%
# prune-cell

# 1. lisez la data-frame des passagers du titanic
df = pd.read_csv('titanic.csv', index_col='PassengerId')

# 2. calculez les valeurs manquantes: totales, des colonnes et des lignes
print(10*'-', 'par colonne')
print(df.isna().sum())
print(10*'-', 'par ligne')
print(df.isna().sum(axis=1))
print(10*'-', 'total')
print(df.isna().sum().sum())

# 3. calculez le nombre de classes du bateau
len(df['Pclass'].unique())

# 4. calculez le taux d'hommes et de femmes
df['Sex'].value_counts()/len(df)
# ou encore
df['Sex'].value_counts(normalize=True)#/len(df)

# 5. calculez le taux de personnes entre 20 et 40 ans (bornes comprises)
((df['Age'] >= 20) & (df['Age'] <= 40)).sum()/len(df)

# 6. calculez le taux de survie des passagers
(df.Survived == 1).mean()

# 7. calculez le taux de survie des hommes et des femmes par classes  
cls = 1

# nb d'hommes
pass_h_cls = ((df['Sex'] == 'male') & (df['Pclass'] == cls)).sum()

# nb de femmes
pass_f_cls = ((df['Sex'] == 'female') & (df['Pclass'] == cls)).sum()

# taux survie homme, femme de classe cls
(   ((df['Sex'] == 'male') & (df['Survived'] == 1) & (df['Pclass'] == cls)).sum()/pass_h_cls,
    ((df['Sex'] == 'female') & (df['Survived'] == 1) & (df['Pclass'] == cls)).sum()/pass_f_cls )

c1 = (df['Pclass'] == 1).sum()
c2 = (df['Pclass'] == 2).sum()
c3 = (df['Pclass'] == 3).sum()

(   ((df['Sex'] == 'female') & (df['Survived'] == 1) & (df['Pclass'] == 1)).sum()/c1,
    ((df['Sex'] == 'male') & (df['Survived'] == 1) & (df['Pclass'] == 1)).sum()/c1     );
