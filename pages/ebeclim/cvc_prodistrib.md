---
title: Production et distribution
summary: "La production et la distribution dans les réseaux hydrauliques pour le chauffage"
permalink: cvc_prodistrib.html
keywords: bâtiment, énergie, chauffage, ventilation, climatisation
tags: [chauffage]
sidebar: sidebar_ebe
topnav: topnav_ebe
folder: ebeclim
---

## Vidéo

<iframe src="https://player.vimeo.com/video/320511676?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Formulaire

La régulation de la puissance d'un émetteur de chaleur peut se faire soit en faisant varier la température de l'eau qui y est envoyée, soit en faisant varier son débit. Dans les deux cas, c'est par une vanne 3 voie qu'est effectué le réglage.

Si la vanne est en **mode mélange**, le débit dans les émetteurs est fixé par un circulateur en aval de la vanne. Ce débit est un mélange d'eau chaude venant de la chaudière et d'eau recirculée depuis l'émetteur, donc plus froide.

<table>
<tr>
<th> <img src="images/ebe/chauffage_melange.png" style="width: 350px;"> </th>
<th style="font-weight: normal">
La puissance (W) de l'émetteur est donnée par son débit (kg/s) et la différence de température entrée-sortie :
$$ P = \dot{m} \, c \, (T_1-T_2) $$
Le débit dans l'émetteur est la somme de celui fourni par la chaudière et de celui revenant dans le by-pass :
$$ \dot{m} = \dot{m}_0 + \dot{m}_{BP}$$
Le flux thermique sortant de la vanne est la somme des deux flux thermiques qui y rentrent :
$$ \dot{m} \, c \, T_1 = \dot{m}_0 \, c \, T_0 + \dot{m}_{BP} \, c \, T_2$$
</th>
</tr>
</table>

Si la vanne est en **mode décharge**, la température de l'eau arrivant vers les émetteurs est égale à celle de la boucle primaire. C'est son débit qui pourra varier selon l'ouverture de la vanne.

<table>
<tr>
<th> <img src="images/ebe/chauffage_decharge.png" style="width: 350px;"> </th>
<th style="font-weight: normal">
La puissance (W) de l'émetteur est donnée par son débit (kg/s) et la différence de température entrée-sortie :
$$ P = \dot{m} \, c \, (T_0-T_2) $$
Le débit dans l'émetteur est plus faible que celui fourni par la chaudière du fait de l'eau dirigée vers le by-pass :
$$ \dot{m} = \dot{m}_0 - \dot{m}_{BP}$$
</th>
</tr>
</table>
