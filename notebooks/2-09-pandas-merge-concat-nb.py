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
#     title: manipulations de base
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(url="https://raw.githubusercontent.com/ue12-p23/numerique/main/notebooks/_static/style.html")

# %% [markdown]
# # agrégations des données

# %%
import pandas as pd
import numpy as np

# %% [markdown] tags=["framed_cell"] jp-MarkdownHeadingCollapsed=true
# ````{admonition} →
# parfois on obtient les données par plusieurs canaux  
# qu'il faut agréger dans une seule dataframe
#
# les outils à utiliser pour cela sont multiples  
# pour bien choisir, il est utile de se poser en priorité  
# la question de savoir si les différentes sources à assembler  
# concernent les **mêmes colonnes** ou au contraire les **mêmes lignes**  (*)
#
#
# illustrations:
#
# * on recueille les données à propos du coronavirus, qui sont disponibles par mois  
#   chaque fichier a la même structure - disons 2 colonnes: *deaths*, *confirmed*  
#   l'assemblage consiste donc à agréger les dataframes **en hauteur**
#
# * on recueille les notes des élèves d'une classe de 20 élèves  
#   chaque prof fournit un fichier excel avec les notes de sa matière  
#   chaque table contient 20 lignes  
#   il faut cette fois agréger les dataframes **en largeur**
#
# ```{note}
# cette présentation est simpliste, elle sert uniquement à fixer les idées
# ```
# ````

# %% [markdown] tags=["framed_cell"]
# ### en hauteur `pd.concat()`
#
# ````{admonition} →
# pour l'accumulation de donnée utilisez la fonction `pandas` suivante
#
# * la fonction `pd.concat([df1, df2, ..])`  
#   qui a vocation à accumuler des données en hauteur  
# ````

# %%
# exemple 1
# les deux dataframes ont les mêmes colonnes
# (ici on crée les dataframe à partir d'un dict décrivant les colonnes)
df1 = pd.DataFrame(
    data={
        'name': ['Bob', 'Lisa', 'Sue'],
        'group': ['Accounting', 'Engineering', 'HR']})

df2 = pd.DataFrame(
    data={
        'name': ['John', 'Mary', 'Andrew'],
        'group': ['HR', 'Accounting', 'Engineering',]})

# %% cell_style="split"
df1

# %% cell_style="split"
df2

# %%
# nous ne gardons pas les index de chaque sous-dataframe
pd.concat([df1, df2], ignore_index=True)
# pd.concat([df1, df2], axis=0) # by default concat rows

# %%
# nous indexons les dataframes par la colonne 'name'
pd.concat([df1.set_index('name'), df2.set_index('name')])

# %% [markdown] tags=["framed_cell"]
# ### en largeur `pd.merge()`
#
# ````{admonition} →
# pour la réconciliation de données, voyez cette fois
#
# * la fonction `pd.merge(left, right)`  
#   ou sous forme de méthode `left.merge(right)`  
#
# * et à la méthode `left.join(right)`
#   une version simplifiée de `left.merge()`
#
#   il est possible d'aligner des dataframes sur les valeurs de plusieurs colonnes
#
# ````

# %% [markdown]
# ### alignements
#
# dans les deux cas, `pandas` va *aligner* les données  
# par exemple on peut concaténer deux tables qui ont les mêmes colonnes  
# même si elles sont dans le désordre
#
# l'usage typique de `merge()`/`join()`  
# est l'équivalent d'un JOIN en SQL  
# pour ceux à qui ça dit quelque chose  
# sans indication, `merge()` calcule les **colonnes communes**  
# et se sert de ça pour aligner les lignes
#

# %%
# exemple 1
# les deux dataframes ont exactement une colonne en commun
df1 = pd.DataFrame(
    data={
        'name': ['Bob', 'Lisa', 'Sue'],
        'group': ['Accounting', 'Engineering', 'HR']})  # une seule colonne

df2 = pd.DataFrame(
    data={
        'name': ['Lisa', 'Bob', 'Sue'],
        'hire_date': [2004, 2008, 2014]})

# %% cell_style="split"
df1

# %% cell_style="split"
df2

# %%
df1.merge(df2)

# %%
pd.merge(df1, df2)

# %%
# exemple 2
# cette fois il faut aligner l'index de gauche
# avec la colonne 'name' à droite

df1 = pd.DataFrame(
    index = ['Bob', 'Lisa', 'Sue'],  # l'index
    data={'group': ['Accounting', 'Engineering', 'HR']})  # une seule colonne

df2 = pd.DataFrame(
    data = {'name': ['Lisa', 'Bob', 'Sue'],
            'hire_date': [2004, 2008, 2014]})

# %% cell_style="split"
df1

# %% cell_style="split"
df2

# %%
# du coup ici sans préciser de paramètres
# ça ne fonctionnerait pas
df1.merge(df2, left_index=True, right_on='name')

# %%
# ou encore
pd.merge(df1, df2, left_index=True, right_on='name')

# %% [markdown] tags=["level_intermediate"] jp-MarkdownHeadingCollapsed=true
# ### `concat()` *vs* `merge()`
#
# les deux fonctionnalités sont assez similaires sauf que
#
# * `merge` peut aligner les index ou les colonnes  
#   alors que `concat` ne considère que les index
#
# * `merge` est une opération binaire  
#    alors que `concat` est n-aire  
#    ce qui explique d'ailleurs la différence de signatures  
#    `concat([d1, d2])` *vs* `merge(d1, d2)`
#
# * seule `concat()` supporte un paramètre `axis=`

# %% [markdown]
# ### **exercice** - collage de datatables
#
# voici 3 jeux de données qu'on vous demande d'assembler  
# pour décrire à la fin 4 caractéristiques à propos de 5 élèves

# %% cell_style="split"
df1 = pd.read_csv('pupils1.csv')
df1

# %% cell_style="split"
df2 = pd.read_csv('pupils2.csv')
df2

# %%
df3 = pd.read_csv('pupils3.csv')
df3

# %%
# votre code

# %% [markdown] tags=["level_intermediate"]
# ### **exercice** - intermédiaire
#
# l'énoncé est le même, sauf que cette fois on a choisi
# d'indexer toutes les tables par la colonne `name`

# %% cell_style="split" tags=["level_intermediate"]
df1i = pd.read_csv('pupils1.csv',
                  index_col='name')
df1i

# %% cell_style="split" tags=["level_intermediate"]
df2i = pd.read_csv('pupils2.csv',
                  index_col='name')
df2i

# %% tags=["level_intermediate"]
df3i = pd.read_csv('pupils3.csv', index_col='name')
df3i

# %% tags=["level_intermediate"]
# votre code

# %%
