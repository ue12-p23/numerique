---
jupyter:
  jupytext:
    cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
    notebook_metadata_filter: 'all, -jupytext.text_representation.jupytext_version,
      -jupytext.text_representation.format_version,

      -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,

      -language_info.file_extension, -language_info.mimetype, -toc'
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
---

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat



# Python numérique: numpy, pandas et matplotlib

## contenu de ce module:

le corpus principal porte sur:

* numpy: le tableau homogène , pour le calcul scientifique;
* pandas: la dataframe (similaire à une table SQL), pour le traitement de données;
* matplotlib: pour les affichages de données scientifiques.
* un module optionnel, et assez court, qui contient des rappels essentiels sur le Python dit "de base"


## rappels

pour être sûr que vous avez tout ce qu'il faut pour travailler


### obtenir le cours

pour commencer 

* vous clonez le dépôt git du cours sur votre ordinateur avec `git clone`
* vous installez les éventuelles dépendances avec
  ```bash
  pip install -r requirements.txt
  ```

<!-- #region cell_style="split" slideshow={"slide_type": "slide"} -->
### lancer Python

1. exécuter un programme déjà fait  
  `$ python monprogramme.py`
1. lancer un interpréteur interactif  
  `$ python`  
  ou encore mieux  
  `$ ipython`
1. mode 'mixte' dans des notebooks  
  `$ jupyter lab # (ou jupyter notebook)` 
<!-- #endregion -->

<!-- #region cell_style="split" -->
**illustration**

ces usages ont été vus dans le cours d'introduction, et [dans la vidéo associée](https://youtu.be/i_ZcP7iNw-U)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": ""} -->
### nos cas d'usage

vous reconnaissez les programmes impliqués dans les différents scénarios:

* le `terminal`: la façon la plus simple de lancer d'autres programmes
* l'interpréteur `Python` : le programme qui exécute du code Python
* interpréteur `IPython` : une surcouche qui ajoute de la souplesse
  * complétion, aide en ligne, déplacement/édition dans l'historique
* les notebooks `jupyter`: nos petits cahiers cours/exercices
<!-- #endregion -->

### sachez à qui vous parlez

**convention**

lorsque c'est ambigu, on préfixera :

* la commande à taper dans un terminal, avec un `$`

  ```bash
  $ python
  ```

* la commande à taper dans un interpréteur Python, avec `>>>`

  ```python
  >>> a = 100
  ```

ça va sans dire, et mieux encore en le disant, mais si vous tapez une commande python dans le terminal - ou inversement - évidemment ça va mal se passer..
