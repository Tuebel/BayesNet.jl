[![Run Tests](https://github.com/Tuebel/BayesNet.jl/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Tuebel/BayesNet.jl/actions/workflows/run_tests.yml)
[![Documenter](https://github.com/Tuebel/BayesNet.jl/actions/workflows/documenter.yml/badge.svg)](https://github.com/Tuebel/BayesNet.jl/actions/workflows/documenter.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Tuebel.github.io/BayesNet.jl)

# About
This code has been produced during while writing my Ph.D. (Dr.-Ing.) thesis at the institut of automatic control, RWTH Aachen University.
If you find it helpful for your research please cite this:
> T. Redick, „Bayesian inference for CAD-based pose estimation on depth images for robotic manipulation“, RWTH Aachen University, 2024. doi: [10.18154/RWTH-2024-04533](https://doi.org/10.18154/RWTH-2024-04533).

# BayesNet.jl
Minimal implementation of a Bayesian Network which is a directed acyclic graph (DAG) where each variable conditionally depends on a set of parent variables ([Wikpedia](https://en.wikipedia.org/wiki/Bayesian_network)).
Type stable and by unrolling recursions and sequentializing the DAG.
