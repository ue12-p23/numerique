---
jupyter:
  jupytext:
    cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
    notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
      -jupytext.text_representation.format_version,-language_info.version, -language_info.codemirror_mode.version,
      -language_info.codemirror_mode,-language_info.file_extension, -language_info.mimetype,
      -toc
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


## mode d'emploi / rappels

pour être sûr que vous avez tout ce qu'il faut pour travailler  
on suppose [les installations faites lors du cours d'introduction](https://ue12-p23-intro.readthedocs.io/en/main/1-01-install-fasttrack-nb.html#les-installations) (bash - vscode - conda - jupyter - jupytext - git)


### convention (sachez à qui vous parlez)

lorsque c'est ambigu, on préfixera :

````{admonition} dans le terminal
:class: seealso

parfois on préfixe la commande à taper **avec un `$`** pour indiquer que ça s'adresse au terminal

  ```bash
  $ python
  ```

  dans ce cas **attention** à ne pas copier-coller le signe `$` !
````

````{admonition} dans l'interpréteur python
:class: seealso

si on préfixe **avec `>>>`** c'est pour indiquer que la commande est du Python - on peut alors la copier-coller dans `python` ou `ipython` ou `jupyter` 

  ```python
  >>> a = 100
  ```

  et pareil il ne faut pas copier le `>>>` évidemment

````

ça va sans dire, et mieux encore en le disant, mais si vous tapez une commande python dans le terminal - ou inversement - évidemment ça va mal se passer..


### obtenir le cours

si tout cela est bien en place il ne vous reste plus alors que deux choses à faire:

* cloner ce dépôt
  ````{admonition} exercice
  on fait comment déjà ?
  ```{admonition} solution
  :class: dropdown
  ```bash
  git clone https://github.com/ue12-p23/numerique.git
  ```
  ````
* installer les dépendances
  ````{admonition} exercice
  on fait comment déjà ?
  ```{admonition} solution
  :class: dropdown
  ```bash
  pip install -r requirements.txt
  ```
  ````


### lancer Python

les notebooks du cours se trouvent .. dans le dossier `notebooks`

1. principalement on lira les notebooks avec jupyter avec (ne pas taper le `$` hein!)   
   `$ jupyter lab`
1. exécuter un programme déjà fait  
  `$ python monprogramme.py`
1. lancer un interpréteur interactif  
  `$ python`  
  ou encore mieux  
  `$ ipython`


### vidéo d'illustration

ces usages ont été vus dans le cours d'introduction, et [dans la vidéo associée](https://youtu.be/i_ZcP7iNw-U)


## environnements virtuels

enfin, et pour les geeks:

lorsqu'on travaille sur plusieurs projets, il est possible de créer un environnement virtuel afin d'isoler les dépendances installées: cela évitera qu'une modification apportée sur un projet impacte les autres projets par effet de bord.

si vous voulez essayer d'utiliser ce système, [voyez cette page dans le cours d'introduction](https://ue12-p23-intro.readthedocs.io/en/main/3-03-optionals-nb.html#multiples-environnements-python)

````{attention}
Comme indiqué dans cette page, il est **crucial** de s'assurer que votre "prompt" bash (ce qui est affiché avant le `$` lorsque vous avez la main dans le terminal) vous indique en permanence dans quel environnement vous vous trouvez.
````


***
