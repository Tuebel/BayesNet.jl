# BayesNet.jl
Minimal implementation of a Bayesian Network which is a directed acyclic graph (DAG) where each variable conditionally depends on a set of parent variables ([Wikpedia](https://en.wikipedia.org/wiki/Bayesian_network)).

By convention, each node in the graph represents a variable and has a unique name associated to it.
Exceptions can be made for specific node implementations, i.e. a modifier which post-processes the result of its child node.
A node is defined by the following abstract type:
```@doc
AbstractNode
```
All implementations expect the following methods are expected to be implemented or the fields of the default implementation to be available:
```julia
children(node) = node.children
model(node) = node.model
name(::YourNode{name}) where {name} = name
rng(node) = node.rng
```
For the specific behavior specify one or many of the following methods for your type:
```julia
rand_barrier(node::YourNode, variables::NamedTuple, dims...)
evaluate_barrier(node::YourNode, variables::NamedTuple)
logdensityof_barrier(node::YourNode, variables::NamedTuple)
bijector_barrier(node::YourNode, variables::NamedTuple)
```


# Naming Convention
Naming of parent-child relationship is reversed in a Bayesian network compared to DAGs.
The probability of a child variable ``y`` given a parent variable ``x`` is ``p(y|x)``.
However, node ``x`` is the parent of node ``y`` in the resulting graph ``x → y``.

Programming is done more intuitively using the graph & node notation, thus we use parent ``x → y``.

```
@license BSD-3 https://opensource.org/licenses/BSD-3-Clause
Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
All rights reserved. 
```