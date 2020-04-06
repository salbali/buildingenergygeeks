---
title: Diagramme de l'air humide
summary: "Comment lire le diagramme de l'air humide et y relever les propriétés de l'air"
permalink: airhumide_dah.html
keywords: bâtiment, énergie, humidité
tags: [traitement]
sidebar: sidebar_ebe
topnav: topnav_ebe
folder: ebeclim
---

## Vidéo

[Diapos au format PDF](/pdf/airhumide1 - dah.pdf)

<iframe src="https://player.vimeo.com/video/99807194?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>


## Formulaire

Un volume d'air humide peut être représenté par un point sur le diagramme psychrométrique, où on peut lire jusqu'à 7 grandeurs physiques intensives :

| $$T$$ | °C | Température |
| $$\mathrm{HR}$$ | - | Humidité relative |
| $$r$$ ou $$w$$ | kg$$_{eau}$$/kg$$_{as}$$ | Teneur en eau |
| $$h$$ | J/kg$$_{as}$$ | Enthalpie spécifique |
| $$v_s$$ | m$$^3$$/kg$$_{as}$$ | Volume spécifique |
| $$T_h$$ | °C | Température d'air humide |
| $$T_r$$ | °C | Température de rosée |

Il suffit de 2 de ces grandeurs (n'importe lesquelles) pour placer un point sur le diagramme et déduire toutes les autres.

Alternativement, on peut aussi calculer ces propriétés : **la lecture du diagramme de l'air humide ne fait que remplacer les formules suivantes**

* L'humidité relative est le rapport entre la pression partielle de vapeur $$p_{vap}$$ et de la pression de saturation $$p_{sat}$$, qui est fonction de la température (en °C)

$$ \mathrm{HR} = \frac{p_{vap}}{p_{sat}} $$

$$ \mathrm{log}_{10}(p_{sat}) = 2,7858 + \frac{7,5 \, T}{237,3 + T} $$

* L'expression de la teneur en eau et du volume spécifique se déduisent de la loi des gaz parfaits, où $$R_{as}=287$$ J.kg$$^{-1}$$.K$$^{-1}$$ est la constante spécifique de l'air :

$$ r = 0,622 \frac{p_{vap}}{p_{atm}-p_{vap}} $$

$$ v_s = \frac{R_{as} \, T}{p_{atm}-p_{vap}} $$

* L'enthalpie spécifique de l'air humide est la somme de l'enthalpie de l'air sec et de celle de la vapeur d'eau :

$$h = c_{as}\, T + r \, (l_v+c_v\, T) $$

## Exercice

<ul id="profileTabs" class="nav nav-tabs">
    <li class="active"><a class="noCrossRef" href="#enonce" data-toggle="tab">Enoncé</a></li>
    <li><a class="noCrossRef" href="#correction" data-toggle="tab">Correction</a></li>
</ul>

<div class="tab-content">

<div role="tabpanel" class="tab-pane active" id="enonce" markdown="1">

Une masse d'air a une enthalpie spécifique $$h = 67$$ J/kg$$_{as}$$ et une teneur en eau $$r = 0.014$$ kg$$_{eau}$$/kg$$_{as}$$.

* Calculer sa température, son humidité relative et son volume spécifique.

* Confirmer ces valeurs en les lisant sur le diagramme de l'air humide

</div>

<div role="tabpanel" class="tab-pane" id="correction" markdown="1">

Vous allez bien y arriver tous seuls non ?

</div>

</div>
