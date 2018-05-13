# Introduction

EsgLiL is a parallel and distributed framework for Monte Carlo simulations. It includes a collection of models for generating Monte scenarios. While the framework is general, the models are oriented to uses in finance and insurance.

# Motivation

Monte Carlo scenario generators are mostly propietary. Those open-source, like QuantLib, require non-trivial installation and require some time to understand its class model. The objective of this project is to provide an easy-to-use interface, familiar to the math used to write those models.

# Installation

You can download EsgLiL stand-alone from gitlab or as part of its parent project, PyLiL. We recommend that you create a virtual environment (with conda or virtualenv). 
EsgLiL has been tested with Python 3.6 and all the versions described in the requirements.txt and environment.yml files. 

Example using Anaconda under Linux (make sure to run this commend from the top EsgLiL folder, where environment.yml is):

    
    conda env create -f environment.yml
    
Example not using Anaconda:

After creating an environment with Pyhon 3.6 (look at virtualenv documentation for this step), run ```pip install -r requirements.txt``` to quickly install all those dependencies.


# Quick tutorial

# Structure [work in progress]
EsgLiL provides a graph approach to creating monte carlo simulations. The starting point is the basic random number simulator. The second step, if needed, is a transformation to a target distribution. Then, a correlation structure across dimensions. The next step is the application of a 
stochastic model as defined by its stochastic differential equations. Then, an asset price layer which feeds from the stochastic processes in the previous step.

These layers can be assembled as needed in a model which takes care of the joint simulation of all these variables. Some popular combinations will be provided in a small model zoo module.

Finally, all calculations can be done in parallel or distributed mode, thanks to its Dask integration.

# Want to contribute?

We are keen on receiving contributions. Please get in touch at py_lil[at]outlook.com with us to discuss how to do it.

We use the Gitflow workflow, for more details read here: http://nvie.com/posts/a-successful-git-branching-model/, https://www.atlassian.com/git/tutorials/comparing-workflows#gitflow-workflow


# License

EsgLil is distributed under the GNU AGPLv3 license.
