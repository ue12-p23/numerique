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
#     title: "intro \xE0 seaborn"
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(filename="_static/style.html")

# %% [markdown]
# # courte introduction à seaborn
#
# une librairie de visualisation plus évoluée que matplotlib pour faire de l'exploration de données

# %% editable=true slideshow={"slide_type": ""} tags=[]
import seaborn as sns

# %% [markdown]
# ## quelques goodies

# %%
# assez pratique, les jeux de données courants

# celui-ci vous le connaissez :)
titanic = sns.load_dataset('titanic')

# et nous ici on va utiliser celui-ci
tips = sns.load_dataset('tips')

# %%
# voyons les pingouins
df = tips

df.head(5)

# %%
# at aussi, tout à fait optionnel
# mais seaborn vient avec des styles

sns.set_style('darkgrid')

# %% [markdown]
# ## les grandes familles de plot
#
# nous allons voir quelques-uns des plot exposés par seabord
#
# - `relplot` pour afficher des **relations** statistiques
# - `distplot` pour afficher la **distribution** d'une ou deux variables
# - `catplot` pour afficher la *distribution* de valeurs **catégorielles*
#
# et aussi
#
# - `jointplot` une version un peu plus élaborée de `relplot`
# - `pairplot` pour étudier en une seule figure les corrélations entre plusieurs colonnes
#
# voyons cela sur quelques exemples

# %% [markdown]
# ## `relplot()`
#
# dans la table des pingouins, on choisit les deux colonnes `total_bill` et `tip` pour voir leur corrélation

# %%
# commençons par une forme
# pas trop intéressante

sns.relplot(data=df, x='total_bill', y='tip');

# %% [markdown]
# jusque-là, rien de bien original; mais en fait avec `seaborn` on peut utiliser davantage que ces deux dimensions `x` et `y`, et notamment (on va voir des exemples tout de suite)
#
# - `hue` pour choisir la couleur 
# - `style` pour choisir la forme (genre x ou o) 
# - `size` pour la taille des points
#
# et même d'autres plus intéressantes
#
# - `col` : par exemple avec une colonne catégorielle à trois valeurs, choisir cette colonne avec le paramètre `col` va construire 3 figures situées côte à côte
# - `row` : pareil mais les figures sont situées l'une au-dessus de l'autre
#
# ce qui en tout, permet de faire en principe des visualisations à 7 dimensions (`x`, `y`, `hue`, `style`, `size`, `col` et `row`
#
# comme promis voici quelques exemples

# %%
# hue=
#
# la même visu mais qui fait ressortir 
# le déjeuner et le diner en couleurs

sns.relplot(data=df, x='total_bill', y='tip', 
            hue='time',   # time vaut 'Lunch' ou 'Dinner'
           );

# %%
# col=
#
# toujours la même donnée, mais cette fois
# on met le déjeuner à gauche et le diner à droite
sns.relplot(data=df, x='total_bill', y='tip', 
            col='time',
           );

# %%
# etc etc
# on peut tout combiner de cette façon...
# et donc en tout on peut mettre en évidence
# jusque 7 dimensions

sns.relplot(data=df, 
            x='total_bill', y='tip', 
            col='time', row='day',
            hue='sex', size='size', style='smoker',
           );

# %% [markdown]
# ````{exercise} révision masques
# :class: dropdown
#
# vérifiez que les données contiennent une seule entrée pour le jeudi soir, et aucune pour les samedi et dimanche à midi
#
# :::{admonition} solution
# :class: dropdown
#
# ```python
# df[ df.day.isin(['Sat', 'Sun']) & (df.time == 'Lunch')]
#
# df[(df.day == 'Thur') & (df.time == 'Dinner')]
# :::
#
# ````

# %% [markdown]
# ### affichage des incertitudes

# %% [markdown]
# signalons enfin, pour le même genre de figures, que `seaborn` permet aussi de visualiser les variations pour les données multiples 

# %%
# un exemple de données où on a plusieurs valeurs (signal) pour le méme X (ici timepoint)

fmri = sns.load_dataset("fmri")
fmri.head(10)

# %%
# si on ajoute `kind=line` on indique qu'on veut un "lineplot" 
# et non pas un "scatterplot" comme tout à l'heure
# dans ce cas seaborn va nous montrer 
# les intervalles de confiance autour de la moyenne

sns.relplot(data=fmri, x="timepoint", y="signal", kind="line");

# %% [raw]
# pour en savoir plus: <https://seaborn.pydata.org/tutorial/relational.html>

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## `displot()`
#
# avec displot on peut représenter la distribution d'une variable numérique; en reprenant les données sur les tips

# %%
df = tips
df.head(2)

# %%
# on peut voir qu'il vaut mieux faire
# le service du soir

sns.displot(
    data=df,
    x='tip',
    hue='time',
    kind='kde');

# %%
# ou encore, la même chose mais en cumulatif

sns.displot(
    data=df,
    x='tip',
    hue='time',
    kind='ecdf');

# %% [raw]
# pour en savoir plus: <https://seaborn.pydata.org/tutorial/distributions.html>

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## `catplot()`
#
# par exemple, la même donnée mais avec d'autres représentations 

# %%
# en x une valeur catégorielle

sns.catplot(
    data=df,
    x='time', hue='time',
    y='tip',
    kind='box');

# %%
# la même chose avec `kind=swarm'

sns.catplot(
    data=df,
    x='time', hue='time',
    y='tip',
    kind='swarm');

# %% [raw]
# pour en savoir plus: <https://seaborn.pydata.org/tutorial/categorical.html>

# %% [markdown]
# ## `jointplot()`
#
# cet outil est très pratique pour fabriquer en un seul appel des vues croisées entre plusieurs colonnes; par exemple

# %%
# toujours la corrélation entre 
# les pourboires et le montant de l'addition

sns.jointplot(
    data=df,
    x='total_bill',
    y='tip',
    hue='time');

# %% [markdown]
# ## `pairplot()`

# %% [markdown]
# va nous montrer la corrélation entre toutes les colonnes numériques  
# ici nous en avons 3:

# %%
df.dtypes

# %%
# ce qui donne un diagramme carré de 3x3 figures:

# sur la diagonale on retrouve un displot de cette colonne
# et dans les autres cases un relplot entre les deux colonnes
# on peut visualiser les colonnes de catégories pour
# par exemple la couleur

sns.pairplot(
    data=df,
    hue='time');

# %% [markdown]
# ## exercice
#
# ````{exercise}
#
# à partir des données du titanic, affichez ce `pairplot`
#
# ```{image} media/3-05-exo-pairplot.png
# :width: 600px
# :align: center
# ```
#
# ```{admonition} attention
# la table du titanic, telle qu'exposée par `seaborn`, n'a pas exactement les mêmes noms/types de colonnes que notre `titanic.csv`
# ```
# ````

# %%
# prune-begin

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("titanic.csv")
df.dtypes

# %%
df.drop(columns='PassengerId,SibSp,Parch,Ticket,Cabin'.split(','), inplace=True)

# %%
# les colonnes numériques (merci stackoverflow)

num_cols = df.select_dtypes(include='number').columns
num_cols

# %%
df['Survived'] = df['Survived'].astype('category')
# df['Pclass'] = df['Pclass'].astype('category')

# %%
# les colonnes numériques (merci stackoverflow)

num_cols = df.select_dtypes(include='number').columns
num_cols

# %%
sns.pairplot(
    data=df,
    diag_kind='kde',
    hue='Sex',
);
plt.savefig("media/3-05-exo-pairplot.png")

# %%
# prune-end

# %% [markdown]
# ## conclusion
#
# ce (très) rapide survol devrait vous convaincre de l'utilité de cette librairie, qui permet de gagner beaucoup de temps pour l'analyse visuelle de vos données
