---
title: Analogie électrique
summary: "Représenter les transferts thermiques comme des courants électriques pour pouvoir faire des calculs simples"
permalink: thermique_analogie.html
keywords: thermique, bâtiment, énergie, analogie, électrique
tags: [thermique]
sidebar: sidebar_ebe
topnav: topnav_ebe
folder: ebeclim
---

## Vidéos

### Vidéo 1 : représentation d'une paroi

Introduction au principe de l'analogie électrique et de la mise en parallèle et en série des composants du bâtiment. [Diapos au format PDF](/pdf/thermique1 - analogie électrique.pdf)

<iframe src="https://player.vimeo.com/video/141894652?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### Vidéo 2: représentation d'un bâtiment

On peut utiliser l'analogie électrique à l'échelle d'un bâtiment pour représenter les transferts par conduction, convection et rayonnement. [Diapos au format PDF](/pdf/thermique2 - analogie électrique.pdf)

<iframe src="https://player.vimeo.com/video/142221212?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Formulaire

La chaleur traversant une unité de surface de mur est proportionnelle à l'écart de température de part et d'autre du mur $$\Delta T$$, et à son coefficient de transfert $$U$$ [W/(m$$^2$$.K)] :

$$ \varphi = U . \Delta T = \dfrac{1}{R} . \Delta T$$

Cette chaleur surfacique $$\varphi$$ a pour dimension [W/m$$^2$$]. La chaleur totale $$\phi$$ perdue à travers un mur de surface $$S$$, en [W], vaut donc :

$$ \phi = S . \varphi = S . U . \Delta T$$

On peut définir un coefficient de déperdition totale comme $$ D = S.U $$ [W/K]

La résistance thermique $$R_i$$ d'un composant d'épaisseur $$e_i$$ et de conductivité thermique $$\lambda_i$$ vaut :

$$R_i=\dfrac{e_i}{\lambda_i}$$

Un mur à plusieurs couches se comporte comme plusieurs résistances placées en série : leurs résistances s'additionnent :

$$ R = \sum_i R_i = \sum_i \dfrac{e_i}{\lambda_i}$$

Un mur composé de plusieurs surfaces (mur bas, vitres...) se comporte comme plusieurs résistances en parallèle: la chaleur traverse chaque surface $$S_i$$ (m$$^2$$) en même temps avec plus ou moins de facilité. Les coefficients de déperdition $$D$$ de chaque surface s'additionnent :

$$ D = S.U = \sum_i S_i.U_i $$

{% include warning.html content="Pour additionner des coefficients de déperdition, il faut les pondérer par leur surface: on additionne des [W], pas des [W/m$$^2$$]" %}

|  | Variable | Dimension |  |
|-------|--------|---------|
| $$R$$ | Résistance | (m$$^2$$.K)/W | s'additionne en série |
| $$U$$ | Coefficient de transfert | W/(m$$^2$$.K) |  |
| $$D$$ | Coefficient de déperdition | W/K | s'additionne en parallèle |

Cette méthode peut être appliquée pour représenter les transferts à l'échelle de tout un bâtiment, y compris en incluant des transferts par renouvellement d'air (voir vidéo 2).

## Exercice

On veut calculer les déperditions thermiques d'un local isolé.

<ul id="profileTabs" class="nav nav-tabs">
    <li class="active"><a class="noCrossRef" href="#enonce" data-toggle="tab">Enoncé</a></li>
    <li><a class="noCrossRef" href="#correction" data-toggle="tab">Correction</a></li>
</ul>

<div class="tab-content">

<div role="tabpanel" class="tab-pane active" id="enonce" markdown="1">

Les parois du local sont composées de :

* 44 m$$^2$$ de mur en béton ($$e_b=15$$ cm, $$\lambda_b=2,3$$ W/(m.K)) avec une couche d'isolant ($$e_{iso}=10$$ cm, $$\lambda_{iso}=0,04$$ W/(m.K))
* 8 m$$^2$$ de double vitrage ($$U_v=3,3$$ W/(m$$^2$$.K))
* La résistance surfacique intérieure est $$h_i = 0,11$$ (m$$^2$$.K)/W, et la résistance extérieure est $$h_e = 0,07$$ (m$$^2$$.K)/W

Le local a également un taux de renouvellement d'air depuis l'extérieur de 9 m$$^3$$/h.

Quelle puissance doit-on fournir pour maintenir le local à 19°C, si la température extérieure est de 2°C ?

</div>

<div role="tabpanel" class="tab-pane" id="correction" markdown="1">

Les déperditions thermiques totales du local sont la somme de trois parties : les pertes par le mur béton+isolant, les pertes par les vitres et les pertes par renouvellement d'air. Il suffit de calculer ces trois parties et de les additionner.

**1. Mur béton isolé**

La résistance thermique totale doit tenir compte des deux couches du mur et des résistances surfaciques

$$R_1 = h_i + \dfrac{e_b}{\lambda_b} + \dfrac{e_{iso}}{\lambda_{iso}} + h_e = 0,11 + \dfrac{0,15}{2,3} + \dfrac{0,10}{0,04} + 0,07 = 2,75$$ (m$$^2$$.K)/W

La déperdition $$D_1$$ est ensuite l'inverse de cette résistance, multiplié par la surface de mur:

$$D_1 = \dfrac{S_1}{R_1} = \dfrac{44}{2,75} = 16,03$$ W/K

**2. Vitres**

Le coefficient $$U_v$$ fourni pour le vitrage n'inclut probablement pas les résistances surfaciques. Il faut les ajouter à la résistance de ce composant pour obtenir le coefficient de déperdition total des vitres $$D_2$$:

$$D_2 = \dfrac{S_2}{h_i+\frac{1}{U_v}+h_e} = \dfrac{8}{0.11+\frac{1}{3,3}+0,07} = 16,56$$ W/K

**3. Renouvellement d'air**

Si le débit d'air est donné en (m$$^3$$/h), le coefficient de déperdition qui y est associé se calcule facilement:

$$D_3 = 0,34 \, Q_v = 3,06$$ W/K

**Total**

La puissance totale $$\phi$$ perdue par le local est la somme des trois composantes de déperditions, multipliée par l'écart de température intérieur-extérieur:

$$\phi = \underbrace{(D_1+D_2+D_3)}_{W/K}.\underbrace{\Delta T}_{K} = (16,03 + 16,56 + 3,06 ). (19-2) = 606$$ W

</div>

</div>
