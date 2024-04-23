[![Run Tests](https://github.com/rwth-irt/BayesNet.jl/actions/workflows/run_tests.yml/badge.svg)](https://github.com/rwth-irt/BayesNet.jl/actions/workflows/run_tests.yml)
[![Documenter](https://github.com/rwth-irt/BayesNet.jl/actions/workflows/documenter.yml/badge.svg)](https://github.com/rwth-irt/BayesNet.jl/actions/workflows/documenter.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://rwth-irt.github.io/BayesNet.jl)

# BayesNet.jl
Minimal implementation of a Bayesian Network which is a directed acyclic graph (DAG) where each variable conditionally depends on a set of parent variables ([Wikpedia](https://en.wikipedia.org/wiki/Bayesian_network)).
Type stable and by unrolling recursions and sequentializing the DAG.