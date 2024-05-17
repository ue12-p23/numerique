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
#     title: TP sur le tri d'une dataframe
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %% {"scrolled": true}
from IPython.display import HTML
HTML(filename="_static/style.html")

# %% [markdown]
# # TP on the moon

# %% [markdown]
# **Notions intervenant dans ce TP**
#
# * suppression de colonnes avec `drop` sur une `DataFrame`
# * suppression de colonne entièrement vide avec `dropna` sur une `DataFrame`
# * accès aux informations sur la dataframe avec `info`
# * valeur contenues dans une `Series` avec `unique` et `value_counts` 
# * conversion d'une colonne en type numérique avec `to_numeric` et `astype` 
# * accès et modification des chaînes de caractères contenues dans une colonne avec l'accesseur `str` des `Series`
# * génération de la liste Python des valeurs d'une série avec `tolist`
#    
# **N'oubliez pas d'utiliser le help en cas de problème.**
#
# **Répartissez votre code sur plusieurs cellules**

# %% [markdown]
# 1. importez les librairies `pandas` et `numpy`

# %%
# votre code

# %%
# prune-cell
import pandas as pd
import numpy as np

# %% [markdown]
# 2. 1. lisez le fichier de données `objects_on_the_moon.csv`
#    2.  affichez sa taille et regardez quelques premières lignes

# %%
# votre code

# %%
# prune-cell
df = pd.read_csv('objects_on_the_moon.csv')
print(df.shape)

# %%
# prune-cell
df.head(2)

# %% [markdown]
# 3. 1. vous remarquez une première colonne franchement inutile  
#      utiliser la méthode `drop` des dataframes pour supprimer cette colonne de votre dataframe  
#      `pd.DataFrame.drop?` pour obtenir de l'aide

# %%
# votre code

# %% {"scrolled": true}
# prune-cell
df.drop(['Unnamed: 0'], axis=1, inplace=True)
# ou aussi
#df.drop(columns=['Unnamed: 0'], inplace = True)

df.head(2)

# %% [markdown]
# 4. 1. appelez la méthode `info` des dataframes  
#    (`non-null` signifie `non-nan` i.e. non manquant)
#    1. remarquez une colonne entièrement vide

# %%
# votre code

# %%
# prune-cell
df.info()

# %% [markdown]
# 5. 1. utilisez la méthode `dropna` des dataframes  
#      pour supprimer *en place* les colonnes qui ont toutes leurs valeurs manquantes  
#      (et pas uniquement la colonne `'Size'`)
#    2. vérifiez que vous avez bien enlevé la colonne `'Size'`

# %%
# votre code

# %%
# prune-cell
df.dropna(how='all', axis=1, inplace=True)
'Size' in df.columns

# %% [markdown]
# 6. 1. affichez la ligne d'`index` $88$, que remarquez-vous ?
#    2. toutes ses valeurs sont manquantes  
#      utilisez la méthode `dropna` des dataframes  
#      pour supprimer *en place* les lignes qui ont toutes leurs valeurs manquantes
#      (et pas uniquement la ligne d'index $88$)

# %%
# votre code

# %%
# prune-cell
df.loc[88]

# %%
# prune-cell
df.dropna(how='all', axis=0, inplace=True)
df.shape

# %% [markdown]
# 7. 1. utilisez l'attribut `dtypes` des dataframes pour voir le type de vos colonnes
#    2. que remarquez vous sur la colonne des masses ?

# %%
# votre code

# %%
# prune-cell
df.dtypes

# %% [markdown]
# 8. 1. la colonne des masses n'est pas de type numérique mais de type `object`  
#       (ici des `str`)   
#    1. utilisez la méthode `unique` des `Series`pour en regarder le contenu
#    2. que remarquez vous ?

# %%
# votre code

# %%
# prune-cell
df['Mass (lb)'].unique()

# %% [markdown]
# 9. 1. conservez la colonne `'Mass (lb)'` d'origine  
#       (par exemple dans une colonne de nom `'Mass (lb) orig'`)  
#    1. utilisez la fonction `pd.to_numeric` pour convernir  la colonne `'Mass (lb)'` en numérique    
#    (en remplaçant  les valeurs invalides par la valeur manquante)
#    1. naturellement vous vérifiez votre travail en affichant le type de la série `df['Mass (lb)']`

# %%
# votre code

# %%
# prune-cell
df['Mass (lb) orig'] = df['Mass (lb)']

# %%
# prune-cell
df['Mass (lb)'] = pd.to_numeric(df['Mass (lb)'], errors='coerce')

# %%
# prune-cell
df['Mass (lb)'].dtype

# %% [markdown]
# 10. 1. cette solution ne vous satisfait pas, vous ne voulez perdre aucune valeur  
#        (même au prix de valeurs approchées)  
#     1. vous décidez vaillamment de modifier les `str` en leur enlevant les caractères `<` et `>`  
#        afin de pouvoir en faire des entiers
#     - *hint*  
#        les `pandas.Series` formées de chaînes de caractères sont du type `pandas` `object`  
#        mais elle possèdent un accesseur `str` qui permet de leur appliquer les méthodes python des `str`  
#        (comme par exemple `replace`)
#         ```python
#         df['Mass (lb) orig'].str
#         ```
#         remplacer les `<` et les `>` par des '' (chaîne vide)
#      3. utilisez la méthode `astype` des `Series` pour la convertir finalement en `int` 

# %%
# votre code

# %%
# prune-cell
df['Mass (lb) clean'] = df['Mass (lb) orig'].str.replace('<', '').str.replace('>', '').astype(int)

# %%
# prune-cell

# REMARQUE - pour les avancés
# il existe en Pandas un type qui contient les entiers ET nan
# donc si par exemple on n'avait pas nettoyé la dernière ligne ci-dessus
# on aurait donc une valeur nan dans la colonne des masses
# et dans ce cas-là on ne peut pas faire .astype(int)
# MAIS on peut le faire avec ce type:

# le type magique se construit comme ceci
int_with_nan = dtype=pd.Int64Dtype()

# df['Mass (lb) clean'] = 
df['Mass (lb) orig'].str.replace('<', '').str.replace('>', '').astype(int_with_nan)

# %%
# prune-cell
df['Mass (lb) clean'].dtype

# %% [markdown]
# 11. 1. sachant `1 kg = 2.205 lb`  
#    créez une nouvelle colonne `'Mass (kg)'` en convertissant les lb en kg  
#    arrondissez les flottants en entiers en utilisant `astype`

# %%
# votre code

# %%
# prune-cell
df['Mass (kg)'] = (df['Mass (lb) clean'] / 2.205).astype(int)

# %% [markdown]
# 12. 1. Quels sont les pays qui ont laissé des objets sur la lune ?
#     2. Combien en ont-ils laissé en pourcentage (pas en nombre) ?  
#      *hint* regardez les paramètres de `value_counts`

# %%
# votre code

# %% {"scrolled": true}
# prune-cell
df['Country'].unique()

# %% {"scrolled": true}
# prune-cell
df['Country'].value_counts(normalize=True)

# %% [markdown]
# 13. 1. Quel est le poid total des objets sur la lune en kg ?
#     2. quel est le poids total des objets laissés par les `United States`  ?

# %%
# votre code

# %%
# prune-cell
df['Mass (kg)'].sum()

# %% {"scrolled": true}
# prune-cell
df.loc[df['Country'] == 'United States', 'Mass (kg)'].sum()

# %%
# prune-cell
# plus tard on fera plutôt
df.groupby(by=['Country'], axis=0)['Mass (kg)'].sum()

# %% [markdown]
# 14. 1. quel pays a laissé l'objet le plus léger ?  
#      *hint* comme il existe une méthode `min` des séries, il existe une méthode `argmin` 

# %%
# votre code

# %%
# prune-cell
df.loc[df['Mass (kg)'].argmin(), 'Country']

# %% [markdown]
# 15. 1. y-a-t-il un Memorial sur la lune ?  
#      *hint*  
#      en utilisant l'accesseur `str` de la colonne `'Artificial object'`  
#      regardez si une des description contient le terme `'Memorial'`
#     2. quel pays qui a mis ce mémorial ?  

# %%
# votre code

# %%
# prune-cell
np.any(df['Artificial object'].str.contains('Memorial'))

# %%
# prune-cell
df.loc[df['Artificial object'].str.contains('Memorial'), 'Country']

# %% [markdown]
# 16. 1. faites la liste Python des objets sur la lune  
#      *hint*  
#      utilisez la méthode `tolist` des séries

# %%
# votre code

# %% {"scrolled": true}
# prune-cell
ao = df['Artificial object'].tolist()
ao

# %% [markdown]
# ***
