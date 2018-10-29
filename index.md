### Welcome to the InverseBuilding page

This page is a series of tutorials showing how to solve various types of inverse problems, or model calibration problems, in Python. The physical application mostly revolves around heat transfer in building physics, hence the repo's name.

It is not a book on the theoretical fundations of inverse methods, but a set of cases aimed at giving clear and documented examples on how to implement some of them in Python. Each tutorial is an IPython notebook, and all the code is available in the page's [GitHub repository](https://github.com/srouchier/InverseBuilding).

Here is the list of the courses available so far (or that will soon be)
* [01_HeatConductivity](http://nbviewer.jupyter.org/github/srouchier/InverseBuilding/blob/master/01_HeatConductivity/Notebook_HeatConductivity.ipynb): using Bayesian Inference and the [pymc](https://pymc-devs.github.io/pymc/) package to estimate the thermal conductivity of a wall
* [02_HeatFlow](http://nbviewer.jupyter.org/github/srouchier/InverseBuilding/blob/master/02_HeatFlow/Notebook_HeatFlow.ipynb): linear and transient inverse heat conduction problem; calculating a heat flow from temperature measurements
* [03_MoistureTransfer](http://nbviewer.jupyter.org/github/srouchier/InverseBuilding/blob/master/03_MoistureTransfer/Notebook_MoistureTransfer.ipynb): using scipy.optimize for the characterisation of heat and moisture transfer properties of a material
* [04_RC_Box](http://nbviewer.jupyter.org/github/srouchier/InverseBuilding/blob/master/04_RC_Box/04_RC.ipynb): calibration of a stochastic and a deterministic RC model 

You can check out my other ongoing projects on [my Wordpress page](https://simonrouchier.wordpress.com/)

### References
* Rouchier S., Woloszyn M., Kedowide Y., Bejat T. (2015) Identification of the hygrothermal properties of a building envelope material by the Covariance Matrix Adaptation evolution strategy, Journal of Building Performance Simulation, DOI: 10.1080/19401493.2014.996608
