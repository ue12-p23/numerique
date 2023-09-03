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
#     title: TP sur le tri d'une dataframe
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(url="https://raw.githubusercontent.com/ue12-p23/numerique/main/notebooks/_static/style.html")

# %% [markdown]
# # TP sur le tri d'une dataframe

# %% [markdown]
# **Notions intervenant dans ce TP**
#
# * affichage des données par `plot`
# * tri de `pandas.DataFrame` par ligne, par colonne et par index
#
# **N'oubliez pas d'utiliser le help en cas de problème.**

# %% [markdown]
# 1. importez les librairies `pandas`et `numpy`

# %%
# votre code

# %% [markdown]
# 2. importez la librairie `matplotlib.pyplot` avec le nom `plt` 

# %%
# votre code

# %%
# prune-cell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# 3. lors de la lecture du fichier de données `titanic.csv`  
#    1. gardez uniquement les colonnes `cols` suivantes `'PassengerId'`, `'Survived'`, `'Pclass'`, `'Name'`, `'Sex'`, `'Age'` et `'Fare'`
#
#    1. mettez la colonne `PassengerId` comme index des lignes
#    1. besoin d'aide ? faites `pd.read_csv?`

# %%
# votre code

# %%
# prune-cell
cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare' ]
df = pd.read_csv('titanic.csv', index_col='PassengerId', usecols=cols)

# %% [markdown]
# 4. en utilisant la méthode `pd.DataFrame.plot`  
#    plottez la dataframe (pas la série) réduite à la colonne des ages  
#    utilisez le paramètre de `style` `'rv'` (`r` pour rouge et `v` pour le style: points triangulaires)
#    

# %%
# votre code

# %%
# prune-cell
df[['Age']].plot(style='rv')

# %% [markdown]
# 5. on va organiser les lignes d'une dataframe suivant l'ordre d'une colonne    
#    en utilisant la méthode `df.sort_values()`:
#    1. créez une nouvelle dataframe  dont les lignes sont triées  
#       dans l'ordre croissant des `'Age'` des passagers
#    2. pour constater qu'elles sont triées, affichez les 4 premières lignes de la dataframe  
#       la colonne des `Age` est triée  
#       les lignes ont changé de place dans la table
#    3. remarquez que l'indexation a été naturellement conservée 
#

# %%
# votre code

# %%
# prune-cell
# on trie dans l'axe des lignes donc `axis=0`
df_sorted = df.sort_values(by='Age', ascending=True, axis=0)
df_sorted.head(4)

# %% [markdown]
# 6. 1. plottez la colonne des ages de la dataframe triée  
#       (n'oubliez pas le style 'bv' pour dire que vous voulez des points  
#       sinon plot trace les lignes qui relient les points)
#    1. Que constatez-vous ?

# %%
# votre code

# %%
# prune-cell
df_sorted[['Age']].plot(style='b.')

# %% [markdown]
# 7. 1. les abscisses de votre plot 2D sont les index  
#     vous tracez donc le point $(804, 0.42)$ puis le point $(756, 0.67)$ ...  
#     alors que vous voudriez tracer le point $(0, 0.42)$ puis le point $(1, 0.67)$ ...  
#     c'est à dire pas avec les 'PassengersId' mais avec des indices de lignes
#    1. une solution: enlever les index de la dataframe  
#       qui a alors des indices utilisez la méthode `reset_index`
#       sur la dataframe restreinte à la colonne des 'Age'
#       

# %%
# votre code

# %%
# prune-cell
df_sorted.reset_index()[['Age']].plot(style='b.')

# %% [markdown]
# ## tri des lignes *égales* au sens d'un premier critère d'une dataframe

# %% [markdown]
# 0. rechargez la dataframe

# %%
# votre code

# %%
# prune-cell
cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare' ]
df = pd.read_csv('titanic.csv', index_col='PassengerId', usecols=cols)

# %% [markdown]
# 2. utilisez `df.sort_values()` pour trier la dataframe suivant la colonne (`'Pclass'`)  
#    et trier les lignes identiques (passagers de même classe) suivant la colonne (`'Age'`)

# %%
# votre code

# %%
# prune-cell 2.
df_sorted = df.sort_values(by=['Pclass', 'Age'])
df_sorted.head(3)

# %% [markdown]
# 3. sélectionnez, dans la nouvelle dataframe, la sous-dataframe dont les ages ne sont pas définis  
#    (utiliser la méthode `isna` sur une série pour créer un masque booléens et appliquer ce masque à la dataframe   

# %%
# votre code

# %%
# prune-cell 3.
df_sorted_isna = df_sorted[df_sorted['Age'].isna()]

# %% [markdown]
# 4. combien manque-il d'ages ?

# %%
# votre code

# %%
# prune-cell
len(df_sorted_isna)

# %% [markdown]
# 5. où sont placés ces passagers dans la data-frame globale triée ?  
# en début (voir avec `head`) ou en fin (voir avec `tail`) de dataframe ?

# %%
# votre code

# %%
# prune-cell
df_sorted.tail() # à la fin

# %% [markdown]
# 6. trouvez le paramètre de `sort_values()`  
# qui permet de mettre ces lignes en début de dataframe lors du tri

# %%
# votre code

# %%
# prune-cell
df_sorted.sort_values(by='Age', ascending=True, axis=0, na_position='first').head()

# %% [markdown]
# 7. produire une nouvelle dataframe en ne gardant que les ages connus,
#    et triée selon les ages, puis les prix de billet

# %%
# prune-cell 7.
df[df.Age.notna()].sort_values(by=['Age', 'Fare'])

# %% [markdown] {"tags": ["level_intermediate"]}
# ## tri d'une dataframe selon l'index

# %% [markdown] {"tags": ["level_intermediate"], "cell_style": "center"}
# en utilisant `pandas.DataFrame.sort_index` il est possible de trier une dataframe  
# dans l'axe de ses index de ligne (ou même de colonnes)  
# utilisez le même genre de dataframe qu'à l'exercice précédent

# %% [markdown] {"tags": ["level_intermediate"], "cell_style": "center"}
# 1. reprenez la dataframe du Titanic  
#    utilisez la méthode des dataframe `sort_index` pour la trier dans l'ordre des index 

# %%
# votre code

# %%
# prune-cell
cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare' ]
df = pd.read_csv('titanic.csv', index_col='PassengerId', usecols=cols)
df.sort_index()

# %% [markdown]
# ***
