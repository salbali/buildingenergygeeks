---
title: Inertie thermique
summary: "En régime non permanent, comment estimer le temps de réponse d'un bâtiment"
permalink: thermique_inertie.html
keywords: thermique, bâtiment, énergie, inertie
tags: [thermique]
sidebar: sidebar_ebe
topnav: topnav_ebe
folder: ebeclim
---

## Vidéo

Explication de l'inertie thermique et de la différence entre une isolation par l'intérieur ou par l'extérieur. [Diapos au format PDF](/pdf/thermique3 - inertie.pdf)

<iframe src="https://player.vimeo.com/video/142244633?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Formulaire

La chaleur accumulée par une paroi à la température $$T_i$$ est définie par rapport à une température de référence $$T_0$$, souvent prise à 0°C. Pour un mur d'épaisseur $$e_i$$, masse volumique $$\rho_i$$ et capacité thermique $$c_{p,i}$$, elle peut être exprimée par unité de surface de la paroi en [J/m$$^2$$] :

$$Q_i = \rho_i . e_i . c_{p,i} (T_i-T_0) $$

L'inertie totale d'une paroi de $$n$$ couches de matériaux peut être estimée par la somme de ces énergies accumulées dans chaque couche :

$$Q = \sum_{i=1}^n Q_i = \sum_{i=1}^n \rho_i . e_i . c_{p,i} (T_i-T_0) $$

{% include warning.html content="Attention aux unités ! Faites toujours une analyse dimensionnelle pour véfifier que vous n'avez pas oublié une masse volumique ou une épaisseur." %}

<table>
<tr>
<th> <img src="images/ebe/thermique_transitoire.png" style="width: 250px;"> </th>
<th style="font-weight: normal">
En régime non transitoire, on peut calculer la variation d'une température dans le temps en fonction des valeurs R et C d'un élément de paroi. Ces valeurs dépendent de l'épaisseur de discrétisation et des propriétés du matériau. Pour l'exemple ci-contre, on résoud l'équation :

$$ C \dfrac{\partial T}{\partial t} = \frac{1}{R} (T_1-T) + \frac{1}{R} (T_2-T) $$

</th>
</tr>
</table>

## Exercices

<ul id="profileTabs" class="nav nav-tabs">
    <li class="active"><a class="noCrossRef" href="#enonce" data-toggle="tab">Enoncé</a></li>
    <li><a class="noCrossRef" href="#correction" data-toggle="tab">Correction</a></li>
</ul>

<div class="tab-content">

<div role="tabpanel" class="tab-pane active" id="enonce" markdown="1">

On considère une paroi constituée de trois couches :

| Couche 1 (béton) | Couche 2 (isolant) | Couche 3 (enduit) |
|-------|--------|---------|
| $$e_b=15$$ cm | $$e_i=4$$ cm | $$e_e=1,5$$ cm |
| $$\lambda_b=1,5$$ W/m.K | $$\lambda_i=0,04$$ W/m.K | $$\lambda_e=1,5$$ W/m.K |
| $$c_b=920$$ J/kg.K | $$c_i=920$$ J/kg.K | $$c_e=920$$ J/kg.K |
| $$\rho_b=2700$$ kg/m$$^3$$ | $$\rho_i=75$$ kg/m$$^3$$ | $$\rho_e=2700$$ kg/m$$^3$$ |

La température extérieure est $$T_e=2°C$$ et la température intérieure est $$T_i=20°C$$. Le coefficient d'échange superficiel extérieur est $$h_e=16,7$$ W/m$$^2$$.K et le coefficient intérieur est $$h_i=9,1$$ W/m$$^2$$.K

**Exercice 1 : estimation de l'inertie**

On suppose que l'isolant est du côté intérieur par rapport au béton.

* Calculer la résistance thermique de chaque couche et la densité de flux traversant le mur.
* Calculer la température des interfaces entre les couches, et la température moyenne de chaque couche.
* Calculer le volant thermique total de la paroi.

Refaire ces calculs pour le cas où l'isolant est à l'extérieur par rapport au béton

**Exercice 2 : comportement dynamique**

On veut simuler le comportement dynamique de cette paroi en réponse à des sollicitations extérieures.

* Discrétiser chaque couche en un nombre suffisants de points et donner la valeur des résistances et capacités en chaque point (cela revient à une discrétisation de type différences finies)
* Etablir les équations de l'évolution dans le temps de chaque point. On pourra prendre un schéma explicite si le pas de temps est suffisamment faible.
* Résoudre et tracer l'évolution de la température à l'interface béton/isolant, en supposant une température initiale constante de $$T=10°C$$

</div>

<div role="tabpanel" class="tab-pane" id="correction" markdown="1">

**Exercice 1 : estimation de l'inertie**

Dans le cas de l'isolation intérieure :

* Résistance thermique totale : $$R=1.28$$ m$$^2$$.K/W
* Flux thermique à travers le mur : $$\phi = 14.07$$ W/m$$^2$$
* Températures aux interfaces: $$T = \left[ 2.84, 4.25, 18.31, 18.45\right] $$
* Chaleur totale: $$Q=1.21$$ MJ

Dans le cas de l'isolation extérieure :

* Résistance thermique totale : $$R=1.28$$ m$$^2$$.K/W
* Flux thermique à travers le mur : $$\phi = 14.07$$ W/m$$^2$$
* Températures aux interfaces: $$T = \left[ 2.84, 16.91, 18.31, 18.45\right] $$
* Chaleur totale: $$Q=6.45$$ MJ

</div>

</div>
