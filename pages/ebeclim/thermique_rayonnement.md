---
title: Rayonnement
summary: "Comment inclure le soleil et les échanges radiatifs dans les calculs thermiques"
permalink: thermique_rayonnement.html
keywords: thermique, bâtiment, énergie, rayonnement
tags: [thermique]
sidebar: sidebar_ebe
topnav: topnav_ebe
folder: ebeclim
---

## Apports solaires (courtes longueurs d'onde)

Première vidéo sur le rayonnement : comment calculer les apports solaires directs et diffus sur une paroi à partir des données météo. [Diapos au format PDF](/pdf/thermique4 - rayonnement CLO.pdf)

<iframe src="https://player.vimeo.com/video/142485394?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

La chaleur totale $$\Phi$$ que le soleil apporte à une surface $$S$$ d'absorptivité $$\alpha$$ est la somme de l'ensoleillement direct $$E_{dir}$$, diffus $$E_{dif}$$ et réfléchi sur les surfaces voisines $$E_{ref}$$. Chacun de ces termes se calcule avec un peu de trigonométrie en fonction de l'angle de la paroi et la position du soleil.

$$ \Phi_{CLO} = \alpha . S . (E_{dir}+E_{dir}+E_{ref}) $$

$$\Phi$$ est exprimée ici en W et non en W/m$$^2$$, puisqu'on a multiplié la somme des ensoleillements par la surface $$S$$.

## Echanges radiatifs entre parois (grandes longueurs d'onde)

### Vidéo

Deuxième vidéo sur le rayonnement, où on aborde le calcul des températures des parois sous l’effet des échanges radiatifs, et la description de la température radiante moyenne. [Diapos au format PDF](/pdf/thermique5 - rayonnement GLO.pdf)

<iframe src="https://player.vimeo.com/video/142616863?color=ff9933&portrait=0" width="640" height="480" frameborder="1" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### Résolution

Pour résoudre un problème d'échange radiatif (grandes longueurs d'onde) entre parois, il faut d'abord connaître tous les facteurs de forme $$F_{ij}$$. Les voici dans deux cas particuliers courants:

<table>
<tr>
<th> <img src="images/ebe_view_factor_1.png" style="width: 150px;"> </th>
<th style="font-weight: normal">
Deux surfaces parallèles de mêmes dimensions:

$$ F_{12} = \frac{1}{\pi x y} \left[ \ln \frac{x_1^2 y_1^2}{x_1^2+y_1^2-1} + 2x\left(y_1 \arctan \frac{x}{y_1}-\arctan x \right) + 2y\left(x_1 \arctan \frac{y}{x_1}-\arctan y \right)  \right] $$

$$ \mathrm{avec} \: x_1=\sqrt{1+x^2} \: ; \: y_1=\sqrt{1+y^2} \: ; \: x=W/H \: ; \: y=L/H $$

</th>
</tr>
</table>

<table>
<tr>
<th> <img src="images/ebe_view_factor_2.png" style="width: 150px;"> </th>
<th style="font-weight: normal">
Deux rectangles adjacents perpendiculaires:

$$ F_{12} = \frac{1}{\pi w} \left[ h \arctan \left(\frac{1}{h} \right) + w \arctan \left(\frac{1}{w} \right) - \sqrt{h^2+w^2} \arctan \left(\frac{1}{\sqrt{h^2+w^2}} \right) + \frac{1}{4} \ln \left( a \, b^{w^2} \, c^{h^2}\right) \right]$$

$$ \mathrm{avec} \: a = \frac{(1+h^2)(1+w^2)}{1+h^2+w^2} \: ; \: b = \frac{w^2(1+h^2+w^2)}{(1+w^2)(h^2+w^2)} \: ; \: c = \frac{h^2(1+h^2+w^2)}{(1+w^2)(h^2+w^2)} $$

$$  h=H/L \: ; \: w=W/L $$
</th>
</tr>
</table>

On peut ensuite résoudre le problème soit en établissant un schéma électrique équivalent comme montré sur la vidéo, soit en posant un système d'équations linéaires comme décrit ci-dessous.

Le système peut être résolu si chaque paroi a soit une température connue, soit un flux net de chaleur connu (par exemple une paroi adiabatique). On écrit une de ces deux équations pour chaque paroi $$i$$ :

* Soit la température $$T_i$$ est connue :

$$J_i =\varepsilon_i \sigma T_i^4 + \rho_i\sum_{j=1}^n J_j F_{ij}$$

* Soit le flux net $$\Phi_i$$ est connu :

$$\frac{\Phi_i}{S_i} = J_i - \sum_{j=1}^n J_j F_{ij}$$

On peut alors poser un système de la forme $$Ax=b$$ où $$A$$ est une matrice $$n \times n$$ ($$n$$=nombre de parois) et $$x$$ est le vecteur des radiosités. Une fois celles-ci connues, on peut utiliser cette formule pour calculer les flux nets et températures initialement inconnues:

$$\frac{\Phi_i}{S_i} = \frac{\varepsilon_i}{1-\varepsilon_i}(\sigma T_i^4-J_i) $$

Une mise en pratique de cette méthodologie est proposée avec l'exercice ci-dessous.

## Exercice

<ul id="profileTabs" class="nav nav-tabs">
    <li class="active"><a class="noCrossRef" href="#enonce" data-toggle="tab">Enoncé</a></li>
    <li><a class="noCrossRef" href="#correction" data-toggle="tab">Correction</a></li>
</ul>

<div class="tab-content">

<div role="tabpanel" class="tab-pane active" id="enonce" markdown="1">

On considère une pièce parallélépipédique aux dimensions suivantes :

<img src="images/ebe_thermique_rayonnement_exo.png" style="width: 250px;">

* La surface $$S_5$$ (mur vertical gauche) est une baie vitrée à la température $$T_5 = 8°C$$.
* La surface $$S_0$$ est un radiateur couvrant la moitié de la hauteur du mur de droite, à la température $$T_0 = 60°C$$.
* La surface $$S_2$$ (le sol) est adiabatique.
* Toutes les autres parois sont à la température $$T = 20°C$$.

Calculer les pertes radiatives de la pièce par la fenêtre, le flux net radiatif cédé par le radiateur et la température du sol. L'émissivité de chaque surface est $$\varepsilon=0,85$$ et la réflectivité est $$\rho=0,15$$

</div>

<div role="tabpanel" class="tab-pane" id="correction" markdown="1">

Toutes les parois à la température $$T = 20°C$$ peuvent être considérées comme une seule surface, qu'on désignera avec l'indice 3. Le problème revient donc à un échange radiatif entre 4 surfaces: il faut écrire 4 équations.

Il faut d'abord trouver tous les facteurs de forme. On a besoin d'utiliser les grandes formules ci-dessus 3 fois en tout, pour trouver les valeurs suivantes :

$$ F_{25}=0.1174 \: ; \: F_{20}=0.081 \: ; \: F_{50}= 0.0477$$

Tous les autres facteurs de forme peuvent être déduits des formules simples: $$S_iF_{ij}=S_jF_{ji}$$ et $$\sum_j F_{ij}=1$$.

On écrit ensuite les 4 équations correspondant à chaque surface. Sur la surface 0 (le radiateur), la température est connue, ce qui donne :

$$J_0 =\varepsilon_0 \sigma T_0^4 + \rho_i\sum_{j=1}^n J_j F_{0j}$$

Sur la surface 2 (le sol), le flux net est connu: la surface est adiabatique donc $$\Phi_2=0$$. On peut donc écrire l'équation:

$$0 = J_2 - \sum_{j=1}^n J_j F_{2j}$$

Les surfaces 3 et 5 ont le même type de condition aux limites que la surface 0 (température connue) donc leur équation est similaire. On aboutit finalement au système linéaire suivant dont la solution est un vecteur contenant les radiosités:

$$\begin{bmatrix} 1-\rho_0F_{00} & -\rho_0F_{02} & -\rho_0F_{03} & -\rho_0F_{05} \\ -F_{20} & 1-F_{22} & -F_{23} & -F_{25}  \\
-\rho_3F_{30} & -\rho_3F_{32} & 1-\rho_3F_{33} & -\rho_3F_{35} \\ -\rho_5F_{50} & -\rho_5F_{52} & \rho_5F_{53} & 1-\rho_5F_{55}
\end{bmatrix}
\begin{bmatrix} J_0 \\ J_2 \\ J_3 \\ J_5 \end{bmatrix} =
\begin{bmatrix} \varepsilon_0\sigma T_0^4 \\ 0 \\ \varepsilon_3\sigma T_3^4 \\ \varepsilon_5\sigma T_5^4 \end{bmatrix}
$$

En résolvant ce système, on obtient les valeurs suivantes en W/m$$^2$$ pour les radiosités:

$$ J_0= 656.61 \: ; \: J_2=433.23 \: ; \: J_3 = 420.45 \: ; \: J_5=366.34 $$

La dernière étape est d'utiliser cette formule:

$$\frac{\Phi_i}{S_i} = \frac{\varepsilon_i}{1-\varepsilon_i}(\sigma T_i^4-J_i) $$

pour trouver les valeurs demandées dans l'énoncé: la température du sol $$T_2$$, le flux net du radiateur $$\Phi_0$$ et le flux net sur la vitre $$\Phi_5$$:

$$T_2= 22.5°C \: ; \: \Phi_0=711.53 W \: ; \: \Phi_5=-410.27 W$$

</div>

</div>
