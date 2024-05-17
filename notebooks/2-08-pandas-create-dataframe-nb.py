# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
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
#     title: manipulations de base
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(filename="_static/style.html")

# %% [markdown]
# # création de dataframe
#
# *ne pas faire en cours, lire en autonomie*

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# de très nombreuses voies sont possibles pour créer une dataframe par programme  
# en voici quelques-unes à titre d'illustration  
# voyez la documentation de `pd.DataFrame?` pour les détails  

# %% [markdown] tags=["framed_cell"]
# ### à partir du dict Python des colonnes
#
# ````{admonition} →
# avec la méthode `pandas.DataFrame`  
# on peut créer un objet de type `pandas.DataFrame`
#
#
# le dictionnaire des colonnes
#
# ```python
# cols_dict = {'names' : ['snail', 'pig', 'elephant', 'rabbit',
#                         'giraffe', 'coyote', 'horse'],
#              'speed' : [0.1, 17.5, 40, 48, 52, 69, 88],
#              'lifespan' : [2, 8, 70, 1.5, 25, 12, 28], }
# ```
#
#
# création de la `pandas.DataFrame`
#
# ```python
# df = pd.DataFrame(cols_dict)
# df
#
# ->  names     speed   lifespan
# 0    snail    0.1     2.0
# 1    pig      17.5    8.0
# 2    elephant 40.0    70.0
# 3    rabbit   48.0    1.5
# 4    giraffe  52.0    25.0
# 5    coyote   69.0    12.0
# 6    horse    88.0    28.0
# ```
# ````

# %%
# le code
import pandas as pd
import numpy as np
cols_dict = {'names' : ['snail', 'pig', 'elephant', 'rabbit',
                        'giraffe', 'coyote', 'horse'],
             'speed' : [0.1, 17.5, 40, 48, 52, 69, 88],
             'lifespan' : [2, 8, 70, 1.5, 25, 12, 28], }

df = pd.DataFrame(cols_dict)
df

# %% [markdown] tags=["framed_cell"]
# ### à partir du `dict` des colonnes et d'une `list` (d'index) des lignes
#
# ````{admonition} →
# avec la méthode `pandas.DataFrame`
#
# le `dictionnaire` des id des colonnes  
# la `liste` des id des lignes
#
# ```python
# cols_dict = {'speed' : [0.1, 17.5, 40, 48, 52, 69, 88],
#              'lifespan' : [2, 8, 70, 1.5, 25, 12, 28], }
#
# line_ids =  ['snail', 'pig', 'elephant', 'rabbit',
#              'giraffe', 'coyote', 'horse']
# ```
#
# création de la `pandas.DataFrame`
#
# ```python
# df = pd.DataFrame(cols_dict, index = line_ids)
# df
# ->       speed   lifespan
# snail    0.1     2.0
# pig      17.5    8.0
# elephant 40.0    70.0
# rabbit   48.0    1.5
# giraffe  52.0    25.0
# coyote   69.0    12.0
# horse    88.0    28.0
# ```
#
# on peut ne pas lui passer la liste des id des lignes
# ````

# %%
cols_dict = {'speed' : [0.1, 17.5, 40, 48, 52, 69, 88],
             'lifespan' : [2, 8, 70, 1.5, 25, 12, 28], }

line_ids =  ['snail', 'pig', 'elephant', 'rabbit',
             'giraffe', 'coyote', 'horse']

df = pd.DataFrame(cols_dict, index = line_ids)
df.values

# %% [markdown] tags=["framed_cell"]
# ### à partir d'un `numpy.ndarray`
#
# ````{admonition} →
# avec la méthode `pandas.DataFrame`
#
# à partir d'un `numpy.ndarray` qui décrit la *table désirée*  
# attention à la forme
#
# et attention au `type`  
# le type des éléments d'un `numpy.ndarray` est homogène  
# (si vous mélangez des `float` et des `str` vous n'avez plus que des string à-la-`numpy`...)
#
# le `numpy.ndarray`
#
# ```python
# nd = np.array([[ 0.1,  2. ],
#                [17.5,  8. ],
#                [40. , 70. ],
#                [48. ,  1.5],
#                [52. , 25. ],
#                [69. , 12. ],
#                [88. , 28. ]])
#
# ```
#
# la `pandas.DataFrame`
#
# ```python
# df = pd.DataFrame(nd)
# df
# ->    0     1
# 0    0.1   2.0
# 1   17.5   8.0
# 2   40.0  70.0
# 3   48.0   1.5
# 4   52.0  25.0
# 5   69.0  12.0
# 6   88.0  28.0
# ```
#
# **remarquez**, sans index
#
# * les index des `2` colonnes sont leurs indices `0` à `1`
# * les index des `7` lignes sont leurs indices `0` à `6`
#
# on peut passer les index (colonnes et/ou lignes)  
# au constructeur de la `pandas.DataFrame`
#
# ```python
# df = pd.DataFrame(nd,
#                   index=['snail', 'pig', 'elephant',
#                          'rabbit', 'giraffe', 'coyote', 'horse'],
#                   columns = ['speed', 'lifespan'])
# df
# ->       speed   lifespan
# snail    0.1     2.0
# pig      17.5    8.0
# elephant 40.0    70.0
# rabbit   48.0    1.5
# giraffe  52.0    25.0
# coyote   69.0    12.0
# horse    88.0    28.0
# ```
# ````

# %%
# le code
nd = np.array([[ 0.1,  2. ],
               [17.5,  8. ],
               [40. , 70. ],
               [48. ,  1.5],
               [52. , 25. ],
               [69. , 12. ],
               [88. , 28. ]])

df = pd.DataFrame(nd)
df

# %%
# le code
nd = np.array([[ 0.1,  2. ],
               [17.5,  8. ],
               [40. , 70. ],
               [48. ,  1.5],
               [52. , 25. ],
               [69. , 12. ],
               [88. , 28. ]])

df = pd.DataFrame(nd,
                  index=['snail', 'pig', 'elephant',
                         'rabbit', 'giraffe', 'coyote', 'horse'],
                  columns = ['speed', 'lifespan'])
df['Names'] = df.index
df.values

# %% [markdown]
# ### **exercice** : création de df et type des éléments

# %% [markdown]
# 1. créer un `numpy.ndarray` à partir de la liste suivante
# ```python
# animals = [['snail', 0.1, 2.0],
#            ['pig', 17.5, 8.0],
#            ['elephant', 40.0, 70.0],
#            ['rabbit', 48.0, 1.5],
#            ['giraffe', 52.0, 25.0],
#            ['coyote', 69.0, 12.0],
#            ['horse', 88.0, 28.0]]
# ```
# 1. Affichez le type des éléments de la table  
# Que constatez-vous ? (U = Unicode)
#
# 1. Créez une `pandas.DataFrame` à partir de la table précédente  
# avec pour noms de colonnes `'names'`, `'speed'` et `'lifespan'`
#
# 1. affichez la valeur et le type du `'lifespan'` de l'éléphant  
# Que constatez-vous ?  
# (`object` signifie ici `str`)
#
# 1. affichez la valeur et le type du `'names'` de l'éléphant  
# Que constatez-vous ?
#
# 1. avec `loc` ou `iloc`, modifiez la valeur `elephant` par `'grey elephant'`  
# affichez la valeur et le type du `'names'` de l'éléphant  
# un constat ?
#
# 1. affichez le type des colonnes  
# utilisez l'attribut `dtypes` des `pandas.DataFrame`
#
# 1. avec la méthode `pandas.DataFrame.to_numpy`  
# affichez le tableau `numpy` sous-jacent de votre data-frame  
# affichez le type du tableau  
# que constatez-vous ?
#
# 1. modifiez les colonnes `'speed'` et `'lifespan'` de manière à leur donner le type `float`  
# (utilisez `pandas.Series.astype` voir les **rappels** en fin de cellule)
#
# **rappel**
#
# * `astype`  
# la méthode `pandas.Series.astype`, à laquelle vous indiquez un type `float`  
# crée (si c'est possible) une nouvelle `pandas.Series` dont les éléments sont de type `float`
#
# * rajouter ou modifier une colonne dans une `pandas.DataFrame`  
# revient à modifier ou rajouter une clé à un `dict`
#
# **explication**
#
# * quand les types des colonnes `numpy` ne sont pas homogènes  
# `numpy` met un tableau de caractères `Unicode` de la *plus grande taille*
#
# * quand les types des colonnes `pandas` ne sont pas homogènes  
# sans indication, `pandas` met `str` `Python`
#
# * quand dans une data-frame `pandas` on mélange des types de colonnes - genre `float` et `str`  
# `pandas` et son tableau `numpy` sous-jacent indiqueront `O` ou `object`  
# pour **mixed data types in columns**

# %% [markdown]
# ***
