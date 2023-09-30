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
#     title: "indexation et acc\xE8s aux sous-tableaux"
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(filename="_static/style.html");

# %% [markdown]
# # indexation et accès aux sous-tableaux

# %%
import pandas as pd
import numpy as np # pandas reposant sur numpy on a souvent besoin des deux librairies

# %% [markdown] tags=["framed_cell"]
# ## introduction
#
# ````{admonition} →
# manipuler des **parties** (vues) de nos données  
# est une opération fréquente en traitement des données
#
# d'où l'importance de savoir localiser dans nos tables `pandas` des sous-parties  
# (élément, ligne, colonne, sous-séries, sous dataframes)  
# afin de leur appliquer une fonction
#
# `pandas` a mis ses efforts sur la gestion d'une indexation des lignes et des colonnes
#
# ils ont **privilégié** le repérage des éléments d'une dataframe **par les index**  `df['Age'][450]`  
# (les **noms** de colonnes et les **labels** de lignes)  
# et **pas** par les **indices** comme en `Python` ou en `numpy`
#
# Pourquoi ?
#
# * parce que quand vous utilisez `pandas`  
#   l'ordre dans lequel sont les données est généralement secondaire  
#   et on préfère faire référence aux données par leur identifiant (*index* donc)  
#
# * si vous n'avez pas besoin d'index particuliers  
#   i.e. si vos données se manipulent facilement à base d'indices  
#   autant rester avec des tableaux 2D `numpy`  
#   avec leurs indices de ligne et de colonne
#
# ```{note}
#
# bien sûr `pandas` va *aussi* vous permettre d'accéder à vos sous-tableaux  
# par indices, c'est juste moins pertinent la plupart du temps
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ## copier une dataframe ou une série
#
# ````{admonition} →
# pour dupliquer une dataframe ou une série (ligne ou colonne)  
# toujours la méthode classique `copy` des objets `Python`
#
# vous allez utiliser la méthode `pandas.DataFrame.copy` ou `pandas.Series.copy`
#
# construisons une dataframe
#
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# ```
#
# copions la
#
# ```python
# df2 = df.copy()
# ```
#
# modifions la copie
#
# ```python
# df2.loc[552, 'Age'] = 100
# # vérifions
# df2.head(1)
#              Survived  Pclass                         Name   Sex    Age  ...
# PassengerId
# 552                 0       2  Sharp, Mr. Percival James R  male  100.0  ...
# ```
#
# l'original n'est pas modifiée
#
# ```python
# df.head(1)
#              Survived  Pclass                         Name   Sex   Age  ...
# PassengerId
# 552                 0       2  Sharp, Mr. Percival James R  male  27.0  ...
# ```
#
# `df2` est une nouvelle dataframe  
# avec les mêmes valeurs que l'originale `df`  
# mais totalement indépendante
# ````

# %% [markdown] tags=["framed_cell"]
# ## créer une nouvelle colonne
#
# ````{admonition} →
# pour créer une nouvelle colonne  
# on la rajoute dans le dictionnaire des colonnes
#
# souvent on crée une nouvelle colonne  
# en faisant un calcul sur des colonnes existantes
#
# les opérations sur les colonnes peuvent utiliser la forme `df[nom_de_colonne]`
#
# dans la dataframe du titanic  
# créons une colonne des décédés (donc 1 - les survivants)
#
# ```python
# df['Deceased'] = 1 - df['Survived']
# ```
#
# nous avons rajouté la clé `'Deceased'` dans l'index des colonnes  
# `pandas` voit sa dataframe comme un dictionnaire des colonnes  
# ````

# %% [markdown] tags=["framed_cell"]
# ## rappels `python`, `numpy`
#
# ````{admonition} →
# pour accéder ou modifier des sous-parties de dataframe
# nous ***pourrions être tentés***:
#
# * d'utiliser les syntaxes classiques d'accès aux éléments d'un tableau par leur indice  
# comme vous le feriez en Python
#
# ```python
# L = [10, 20, 30, 40, 60]
# L[0] = "Hello !"
# print(L) # ['Hello !', 20, 30, 40, 60]
# L[1:3] = [200, 300, 500]
# L
# -> L[1:3] = [200, 300, 500]
# ```
#
# * ou d'utiliser l'accès à un tableau par une paires d'**indices**  
# comme vous le feriez en `numpy`
#
#     créons une matrice `numpy` (4, 4)  
#     et modifions une sous-matrice
#
# ```python
# mat = np.arange(12).reshape((4, 3))
# mat[0:2, 0:2] = 999
# mat
# -> [[999, 999,   2],
#     [999, 999,   5],
#     [  6,   7,   8],
#     [  9,  10,  11]])
# ```
#
# * ou encore enfin, en passant par la colonne puis la ligne  
#   ça fonctionne, on l'a même peut-être déjà fait  
#   mais **ATTENTION** il ne **FAUT PAS** faire comme ça !
#
# ```python
# df['Age'][552]
# 27.0
# ```
#
# ***mais ATTENTION  
# ce n'est pas comme ça que ça fonctionne en `pandas`***
#
# ````

# %% [markdown]
# ## localiser en `pandas`

# %% [markdown] tags=["framed_cell"]
# ### ligne,colonne *vs* colonne, ligne
#
# ````{admonition} →
# la première **grosse différence** entre `numpy` et `pandas`  
# est que
#
# * un tableau `numpy` de dimension 2  
#   est organisé en *ligne, colonne*  
#   c'est-à-dire que `tab[i]` renvoie **une ligne**
#
# * mais on a vu précédemment que sur une dataframe  
#   `df[truc]` renvoie **une colonne**  
#
# donc déjà on sait qu'on ne pourra pas écrire quelque chose comme  
# `df[ligne, colonne]` **NON**
# ````

# %% [markdown] tags=["framed_cell"]
# ### localisation avec `loc` et `iloc`
#
# ````{admonition} →
# première chose à retenir donc, les accès dans la dataframe  
# se font **au travers de 2 accessoires `loc`** et `iloc`  
# qui prennent cette fois-ci leurs arguments *dans le bon sens*
#
# `df.loc[index_ligne, index_colonne]` **OUI**  
# `df.iloc[indice_ligne, indice_colonne]` **OUI**  
#
#
# la différence entre les deux est que `loc` se base sur les **index**  
# alors que `iloc` (retenir: *i* pour *integer*) se base sur les **indices**
#
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df.head(2)
# ->              Survived  Pclass                         Name  ...   Fare  Cabin  Embarked
# PassengerId                                                 ...
# 552                 0       2  Sharp, Mr. Percival James R  ...  26.00    NaN         S
# 638                 0       2          Collyer, Mr. Harvey  ...  26.25    NaN         S
#
# df.tail(1)
# ->              Survived  Pclass                             Name  ...   Fare  Cabin  Embarked
# PassengerId                                                     ...
# 832                 1       2  Richards, Master. George Sibley  ...  18.75    NaN         S
#
# # accès par l'index
# # pour les lignes: la valeur de 'PassengerId'
# # pour les colonnes: les noms des colonnes
# df.loc[552, 'Name']
# -> 'Sharp, Mr. Percival James R'
#
# # accès par indice (plus rare en pratique)
# # attention la colonne d'index ne compte pas
# # i.e. la colonne d'indice 0 est 'Survived'
# df.iloc[0, 2]
# -> 'Sharp, Mr. Percival James R'
#
# # pareil avec un indice négatif
# df.iloc[-1, 2]
# -> 'Richards, Master. George Sibley'
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ### sélection multiple
#
# ````{admonition} →
# une fois ceci assimilé, `pandas` va offrir des techniques usuelles  
# pour sélectionner plusieurs lignes (ou colonnes)  
# 1. sélection multiple explicite
# 1. slicing
#
#
# commençons par la sélection multiple:  
#
# * si on ne précise pas les colonnes, on les obtient toutes  
# * on peut mentionner simplement plusieurs index (ou indices)  
#   que l'on passe **dans une liste**
#
#
# quelques exemples
#
# ```python
# # comme avec un tableau numpy,
# # si on ne précise pas les colonnes
# # on les obtient toutes
# df.loc[552]
# -> une série qui matérialise la première ligne
#
# # on peut passer des listes à loc/iloc
# # pour sélectionner explicitement
# # plusieurs lignes / colonnesa
# df.loc[[552, 832]]
# -> une dataframe avec deux lignes correspondant
#    aux deux passagers 552 et 832
#    
# df.loc[[552, 832], ['Name', 'Pclass']]
# -> la même dataframe mais réduite à deux colonnes  
#
# # à nouveau pour les indices de colonnes
# # la colonne d'index ne compte pas
# df.iloc[[0, -1], [2, 1]]
# -> la même
#
# # pour sélectionner plusieurs colonnes
# # le plus simple c'est quand même cette forme
# df[['Name', 'Pclass']]
# -> 2 colonnes, toutes les lignes
#
# # mais bien sûr on peut aussi faire
# df.loc[:, ['Name', 'Pclass']]
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ### slicing `pandas` et bornes
#
# ````{admonition} →
# on va accéder à des sous-dataframe  
# en étendant l'opération d'indexation `[i]` à des slices `[start:stop:step]`  
# comme en `python` et `numpy`
#
# **ATTENTION** pour le *slicing*  
# il y a une **grande différence** entre `loc` et `iloc`  
#
# * **avec `loc`: la slice contient les bornes**  
# * alors que avec `iloc` la borne supérieure est exclue  
#   comme c'est l'habitude en Python
#
# ````

# %% [markdown] tags=["framed_cell"]
# ### slicing avec `loc` par index
#
# ````{admonition} →
#
# on peut slicer sur les index  
# **MAIS ATTENTION** pour les **index** `stop` est compris  
#
# **exemple**  
# regardons les index (lignes et colonnes)  
#
# ```python
# # les 5 premiéres lignes
# df.index[:5]
# -> Int64Index([552, 638, 499, 261, 395], dtype='int64', name='PassengerId')
#
# # les 5 premières colonnes
# df.columns[:5]
# -> Index(['Survived', 'Pclass', 'Name', 'Sex', 'Age'], dtype='object')
#
# # le slicing avec .loc est inclusif
# df.loc[ 638:261, 'Pclass': 'Age']
# -> retourne une dataframe avec
#    3 lignes (638 et 261 inclus)
#    4 colonnes ('Pclass' et 'Age' inclus)
# ```
#
# ````

# %% [markdown] tags=["framed_cell"]
# ### slicing avec `iloc` par indices
#
# ````{admonition} →
# on peut `slicer` sur les indices  
# `df.iloc[start:stop:step, start:stop:step]`
#
# ce cas est simple car il est conforme aux habitude Python/numpy  
# la borne supérieure `stop` est exclue  
# et donc en particulier le nombre d'items sélectionnés  
# coincide avec `stop-start`
#
# **exemple**  
# si on prend les lignes d'indice `1` à `7`  
# et les colonnes d'indice `1` à `4`  
# on obient 6 lignes et 3 colonnes
#
# ```python
# df.iloc[1:7, 1:4].shape
# -> (6, 3)
# ```
#
# ```{tip}
# avec la méthode `get_loc()` sur un objet Index  
# on peut facilement obtenir l'indice d'un index (de colonne ou de ligne)  
# ```
#
# ```python
# # l'indice de la colonne 'Pclass'
# df.columns.get_loc('Pclass')
#
# -> 1
# ```
#
# ```python
# # l'indice de la ligne 261
# df.index.get_loc(261)
#
# -> 3
# ```
#
# ````

# %% [markdown] tags=["framed_cell"]
# ### localiser des lignes et des colonnes
#
# ````{admonition} →
# ***ou sous-lignes et sous-colonnes***
#
# avec le *slicing*, par indice et index, on peut obtenir des lignes et des colonnes  
# ou des sous-lignes et des sous-colonnes
#
# on obtient des objets de type `pandas.Series`
#
# on peut slicer, par index, pour obtenir une ligne
#
# ```python
# df.loc[552, :] # première ligne (toutes les colonnes)
# df.loc[552, :].shape
# -> (11,)
# ```
#
# on peut slicer, par index,  pour obtenir une colonne
#
# ```python
# df.loc[:, 'Survived'] # première colonne (toutes les lignes)
# df.loc[:, 'Survived'].shape
# -> (891,)
# ```
#
# on peut slicer, par indice, **pour obtenir une ligne**
#
# ```python
# df.iloc[0, :] # première ligne (toutes les colonnes)
# df.iloc[0, :].shape
# -> (11,)
# ```
#
# notez qu'on peut alors omettre les colonnes puisqu'on les prend toutes
#
# ```python
# df.iloc[0] # première ligne (toutes les colonnes)
# df.iloc[0].shape
# -> (11,)
# ```
#
# on peut slicer, par indice,  **pour obtenir une colonne**
#
# ```python
# df.iloc[:, 0] # première colonne (toutes les lignes)
# df.iloc[:, 0].shape
# -> (891,)
# ```
#
# ````

# %% [markdown] tags=["framed_cell"]
# ````{exercise} sélections multiples et slicing
#
# 1. lisez le titanic et mettez les `PassengerId` comme index des lignes
# 2. localisez l'élément d'index `40`  
#   a. Quel est le type de l'élément ?  
#   b. localisez le nom du passager d'index `40` ?
# 3. quel est le nom de la personne qui apparaît en avant-dernier dans le fichier
# 4. localisez les 3 derniers éléments de la ligne d'index `40`
# 5. localisez les 4 derniers éléments de la colonne `Cabin`
# 6. fabriquez une dataframe contenant
#   * les infos des 10 dernières lignes du fichier
#   * pour les colonnes `Name`, `Pclass` et `Survived`
# ````

# %%
# prune-cell

# 1. lisez le titanic et mettez les `PassengerId` comme index des lignes
df = pd.read_csv('titanic.csv', index_col='PassengerId')
df = pd.read_csv('titanic.csv').set_index('PassengerId')

# 2. localisez l'élément d'index `40`  
# 2.a. Quel est le type de l'élément ?  
# 2.b. localisez le nom du passager d'index `40` ?

p40 = df.loc[40]
print(f"{p40} is of type {type(p40)}")
print(f"their name is {p40.loc['Name']}")
print(f"their name is also {df.loc[40, 'Name']}")

# 3. quel est le nom de la personne qui apparaît en avant-dernier dans le fichier
df.iloc[-2].loc['Name']

# 4. localisez les 3 derniers éléments de la ligne d'index `40`
df.loc[40].iloc[-3:]

# 5. localisez les 4 derniers éléments de la colonne `Cabin`
df['Cabin'].iloc[-4:]

# 6. fabriquez une dataframe contenant
# les infos des 10 dernières lignes du fichier
# pour les colonnes `Name`, `Pclass` et `Survived`
df.iloc[-10:][['Name', 'Pclass', 'Survived']]

# %% [markdown]
# ## indexation par un masque

# %% [markdown] tags=["framed_cell"]
# ### rappel sur les conditions
#
# ````{admonition} →
# nous avons vu comment appliquer des conditions  
# à une colonne ou à une data-frame  
# et comment utiliser ce tableau de booléens pour des décomptes
#
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df_survived = (df['Survived'] == 1)
# df_survived.sum()/len(df)
# ->  0.3838383838383838
# ```
#
# ```{admonition} rappel
# :class: seealso
#
# on a vu comment combiner ces conditions  
# vous ne pouvez **pas** utiliser `and`, `or` et `not` python (pas vectorisés)  
# et devez utiliser `&`, `|` et `~`  
# ou `np.logical_and`, `np.logical_or` et `np.logical_not`
# ```
#
# taux de survie des passagers femmes de première classe
#
# ```python
# ( 
#     (
#           (df['Sex'] == 'female') # les femmes
#         & (df['Pclass'] == 1)     # de première classe
#         & (df['Survived'] == 1)   # ayant survécu
#     ).sum() # la somme
#     /
#     (
#           (df['Sex'] == 'female') # les femmes
#         & (df['Pclass'] == 1)     # de première classe
#     ).sum() # la somme
# )
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ### sélection par masque booléen
#
# ````{admonition} →
# les objets comme nous venons d'en construire  
# e.g. `df['Sex'] == 'female'`  
# sont des **séries à valeur booléennes**
#
# une **série à valeur booléennes** s'appelle **un masque** (comme en `numpy`)
#
# pour accéder à des sous-parties d'une dataframe  
# on va simplement **indexer** une dataframe **par un masque**  
# i.e. on va isoler les lignes de la dataframe où la valeur du booléen est vraie
#
# et pour ça on écrit simplement  
#
# ```python
# df [ df['Sex'] == 'female' ]
# # ou encore
# df.loc[ df['Sex'] == 'female' ]
#
# -> pandas retourne une nouvelle dataframe ne contenant que les femmes
#
# ```
#
# ```{admonition} note
# :class: seealso
#
# ici le masque est une série qui a **le même index** que la dataframe  
# et une valeur booléenne, qui va indiquer si la ligne en question  
# doit être sélectionnée ou non
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ### `df[mask]` décortiqué
#
# ````{admonition} →
# faisons le *masque* des passagers de sexe féminin
#
# ```python
# # le code
# mask = df['Sex'] == 'female'
# mask
# ->  PassengerId
#     552    False
#     638    False
#     499     True
#     261    False
#     395     True
#            ...
#     463    False
#     287    False
#     326     True
#     396    False
#     832    False
#     Name: Sex, Length: 891, dtype: bool
# ```
#
# vous obtenez une `pandas.Series` de `bool`  
# sa taille est le nombre de lignes de votre dataframe  
# indiquant le résultat de la condition pour chaque les passagers  
# le passager d'`Id` `499` est une femme
#
# pour extraire la sous-dataframe des femmes  
# on **indexe** notre dataframe, par cet objet de type `Series` de booléens
#
# seules sont conservées les lignes, dont les booléens sont vrais
#
# dans l'expression `df[mask]`  
# dans les crochets on n'a plus ni une slice, ni une liste  
# mais un objet de type `Series`, qui s'apparente à une colonne,  
# de booléens, que l'on appelle un **masque**
#
# pour un code concis et lisible  
# il est recommandé d'écrire directement la version abrégée
#
# ```python
# df[df['Sex'] == 'female']
# # ou encore, moins lourd amha
# df[df.Sex == 'female']
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ````{exercise} combinaison d'expressions booléennes
#
# 1. en une seule ligne sélectionner la sous-dataframe des passagers  
# qui ne sont pas en première classe  
# et dont l'age est supérieur ou égal à 70 ans
#
# 2. Combien trouvez-vous de passagers ?
#
# 3. Accédez à la valeur `Name` du premier de ces passagers
#
# 2. Faites la même expression que la question 1  
# en utilisant les fonctions `numpy.logical_and`, `numpy.logical_not`
#
#
# ````

# %%

# 1. en une seule ligne sélectionner la sous-dataframe des passagers  
# qui ne sont pas en première classe  
# et dont l'age est supérieur ou égal à 70 ans
selection = df[ (df.Pclass != 1) & (df.Age >= 70) ]
selection

# 2. Combien trouvez-vous de passagers ?
len(selection)

# 3. Accédez à la valeur `Name` du premier de ces passagers
selection.iloc[0].loc['Name']

# 4. Faites la même expression que la question 1  
# en utilisant les fonctions `numpy.logical_and`, `numpy.logical_not`

selection2 = df [ np.logical_and ( np.logical_not( df.Pclass == 1), df.Age >= 70)]

# %% [markdown]
# ## règles des modifications

# %% [markdown] tags=["framed_cell"]
# ### sélections de parties de dataframe
#
# ````{admonition} →
# une opération sur une dataframe `pandas` renvoie une **sous-partie** de la dataframe
#
# **le problème**
#
# * savoir si cette sous-partie **réfère** la dataframe initiale ou est une **copie** de la data-frame initiale
# * ...ça dépend du contexte
#
# vous devez vous en soucier ?
#
# * **oui**, dès que vous **voulez modifier** des sous-parties de dataframe
# * tant que vous ne faites que lire, tout va bien
#
# en effet
#
# * si c'est une **copie**  
#  votre modification ne sera **pas prise en compte** sur la dataframe d'origine  
#  (voire pire elle sera prise en compte un peu *par hasard* mais  **vous ne pouvez pas compter sur le résultat**)
#
# * si c'est une **référence partagée** (une vue)  
# vos modifications dans la sélection, seront bien **répercutées** dans les données d'origine
#
# **donc**  
# savoir si une opération retourne une copie ou une référence, **c'est important !**  
# et dépend toujours du contexte
#
# **à retenir**
#
# * en utilisant les méthodes **`pandas.DataFrame.loc[line, column]`** et `pandas.DataFrame.iloc[line, column]`  
# on ne **crée pas de copie** mais des **références partagées**  
# c'est ***la bonne façon de faire***
#
#
#
#
# * dès que vous utiliser un **chaînage d'indexation** pour modifier  
# que ce soit `df[l][c]` ou `df.loc[l][c]` ou `df.iloc[l][c]`  
#  **vous ne pouvez pas compter sur le résultat**  
# ça fonctionne par hasard  
# ***à éviter absolument***
#
# (pour les avancés) ce *problème* s'appelle le *chained indexing*  
# <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy>
# ````

# %% [markdown] tags=["framed_cell"]
# ### modification d'une copie
#
# ````{note}
# cette section est un peu avancée; pour les groupes de débutants, retenez simplement de toujours utiliser `.loc` (ou `.iloc` selon le contexte) pour créer des sélections de vos dataframes, si l'objectif est d'en modifir le contenu
# ````
#
# ````{admonition} →
#
# **par chainage d'indexations**
#
# prenons une dataframe et accèdons à une colonne  
# en utilisant la syntaxe classique d'accès à une colonne comme à une clé d'un dictionnaire
#
# la colonne des survivants `'Survived'`
#
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df['Survived']
# ```
#
# on obtient une colonne de type `pandas.Series`  
# accédons à l'élément d'index `1` de la colonne  
#
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df['Survived'][1]
# -> 0
# ```
#
# Pouvons-nous utiliser cette manière d'accéder pour modifier l'élément ?  
# et ressusciter le passager d'index 1 en changeant son état de survie
#
# essayons, on obtient un message d'erreur:
#
# ```python
# df['Survived'][1] = 1
# ```
# ```
# A value is trying to be set on a copy of a slice from a DataFrame
#
# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   df['Survived'][1] = 1
#
# ```
#
# **non**
#
# * `df['Survived'][1]` est clairement une indexation par chaînage, on voit les `[][]`
# * ce n'est pas une référence
# * toutes les indexations par chaînage sont des copies
# * elle ne doivent pas être utilisées pour des modifications
#
# il est tout à fait possible que `df['Survived'][1]` soit passé à 1, mais c'est **par hasard**  
# et vous **devez utiliser** `loc` ou `iloc` pour modifier
#
# > **gros warning** $\Rightarrow$ **code faux**
#
#
# ```python
# df.loc[1, 'Survived'] = 1
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ### faire des copies explicites
#
# ````{admonition} →
# vous ne voulez pas modifier la dataframe d'origine ?  
# faites une copie **explicite** de la sous-dataframe
#
# ```python
# df2 = df[ ['Survived', 'Pclass', 'Sex'] ].copy() # copie explicite
# df2.loc[1, 'Survived']     # 1
# df2.loc[1, 'Survived'] = 0 # on le passe à 0
# df2.loc[1, 'Survived']     # 0 maintenant
# df.loc[1, 'Survived']      # toujours 1 dans la dataframe d'origine df
# ```
#
# si l'idée est de ne modifier qu'une copie d'une dataframe  
# utilisez `copy` pour maîtriser ce que vous faites  
# et coder ainsi explicitement et proprement
#
# ```{note}
# `pandas`  développe un copy-on-write  
# <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>
# ```
# ````
