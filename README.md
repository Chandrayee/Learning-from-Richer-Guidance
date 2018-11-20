# Learning-from-Richer-Guidance
Companion codes for the paper 'Learning from Richer Human Guidance: Augmenting comparison-based learning with Feature Queries', HRI 2018, paper link here: https://dl.acm.org/citation.cfm?id=3171284
Use this code to interactively learn probability distribution over reward functions using comparison and feature queries.

## Modules 
1. baseline.py
2. cirlf_noeps.py
3. cirlf_eps.py. 

Baseline is one-step comparison queries as in the work Dorsa Sadigh and reuses some of her codes found here: https://github.com/dsadigh/driving-preferences

cirlf_eps.py and cirlf_noeps.py are two versions of comparison-based learning with feature queries

The programs use pywren to parallelize the runs in python. More information about pywren here: http://pywren.io/

Modules and packages necessary for running the above codes are world.py, utils.py, visual.py and pywren

Most of the plots in the paper are generated using allplots.py

## Running
Run baseline.py and cirlf_noeps.py separately for simulation experiments

Run userstudy1.py -m [method 1 or 2 or 3] for running one user study where method 1 is 20 baseline queries,
method 2 is 20 comparison+feature queries and method 3 is the validation part (more details on validation in the paper)
