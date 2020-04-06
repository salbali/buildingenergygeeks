---
title: Ventilation
summary: "Faire un bilan de pression pour estimer les débits d'air entre les zones thermique"
permalink: thermique_ventilation.html
keywords: thermique, bâtiment, énergie, ventilation
tags: [thermique]
sidebar: sidebar_ebe
topnav: topnav_ebe
folder: ebeclim
---

## Vidéo

Comment prédire les écoulements d’air sous l’effet du vent et des différences de température. [Diapos au format PDF](/pdf/thermique6 - aéraulique.pdf)

<iframe src="https://player.vimeo.com/video/142891349?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Formulaire

Une petite ouverture de surface $$S$$ et de coefficient de décharge $$C_d$$ sépare deux ambiances aux pressions $$P_1$$ et $$P_2$$. Le débit d'air $$Q_v$$ [m$$^3$$/s] à travers l'ouverture est donné par :

$$ P_1 - P_2 = \frac{\rho Q_v^2}{2S^2C_d^2} $$

En présence de vent, la pression totale de l'air est la somme de la pression atmosphérique $$P_{atm}$$ et d'un terme de pression dynamique, qui est fonction du coefficient de pression $$C_p$$ et de la vitesse du vent $$V$$.

$$ P = P_{atm} + C_p \frac{1}{2} \rho V^2 $$

La masse volumique de l'air $$\rho$$ [kg/m$$^3$$] peut être approchée par l'équation suivante, où la température $$T$$ est en Kelvin :

$$\rho = \frac{353}{T}$$

<img src="images/ebe_thermique_tirage.png" style="width: 600px;">

L'effet de tirage thermique vient de cette relation entre masse volumique et température. Sur la figure ci-dessus, deux ambiances sont aux températures $$T_e$$ et $$T_i$$. La différence de pression entre elles est une fonction de la hauteur $$z$$ :

$$ P_e(z) - P_i(z) = \left( P_e(0)-\rho_e \, g \, z\right) - \left( P_i(0)-\rho_i \, g \, z \right) = P_e(0)-P_i(0)-\rho_e \, g \, z \, \frac{T_i-T_e}{T_i}$$

*L'axe neutre* est la hauteur $$z_n$$ pour laquelle cette différence de pression est nulle :

$$z_n = \dfrac{P_e(0)-P_i(0)}{g(\rho_i-\rho_e)} $$

## Exercices

<ul id="profileTabs" class="nav nav-tabs">
    <li class="active"><a class="noCrossRef" href="#enonce" data-toggle="tab">Enoncé</a></li>
    <li><a class="noCrossRef" href="#correction" data-toggle="tab">Correction</a></li>
</ul>

<div class="tab-content">

<div role="tabpanel" class="tab-pane active" id="enonce" markdown="1">

<img src="images/ebe_thermique_aeraulique.png" style="width: 500px;">

Une porte de hauteur $$H=2$$m et de largeur $$w=1$$m sépare deux pièces aux températures $$T_1=15°C$$ et $$T_2=21°C$$.

On suppose (sans le démontrer) que la vitesse de l'air à l'interface, en fonction de la hauteur $$z$$, est donnée par :

$$ V(z) = \sqrt{2\, \frac{P_1(z)-P_2(z)}{\rho}} $$

où $$\rho$$ est la masse volumique du côté où rentre l'air (qui dépend de si on se trouve en dessous ou au dessus de l'axe neutre).

Intégrer $$V(z)$$ sur toute la hauteur $$H$$ pour calculer le débit massique total d'air $$Q_m$$ [kg/s] passant de la pièce 1 vers la pièce 2. On suppose que la porte est une somme infinie de petites ouvertures de coefficient de décharge $$C_d=0.4212$$.

</div>

<div role="tabpanel" class="tab-pane" id="correction" markdown="1">

En dessous de l'axe neutre $$z_n$$, l'air passe de la pièce 1 vers la pièce 2 avec un débit total :

$$Q_{m,12} = C_d \int_0^{z_n} \rho_1 \, V(z) \, w \, \mathrm{d}z $$

Au dessus de l'axe neutre $$z_n$$, l'air passe de la pièce 2 vers la pièce 1 avec un débit total :

$$Q_{m,21} = C_d \int_{z_n}^{H} \rho_2 \, V(z) \, w \, \mathrm{d}z $$

Le débit net total de 1 à 2 vaut $$Q_{m,12}-Q_{m,21}$$

</div>

</div>
