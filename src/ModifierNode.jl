# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
    ModifierNode
Wraps another node and represents the same variable as the `wrapped` node.
For the model create a new type e.g. ModifierModel and implement:
* `rand(rng,::AbstractRNG, model::ModifierModel, x)`, where `x` is a randomly drawn value from the wrapped node.
* `logdensityof(model::ModifierModel, x, ℓ)`, where `x` is the value of the evaluated variable and `ℓ` the logdensity returned by the wrapped model

When traversing the graph, only the wrapped node is returned. 
"""
struct ModifierNode{name,child_names,N<:AbstractNode{name,child_names},R<:AbstractRNG,M} <: AbstractNode{name,child_names}
    wrapped_node::N
    rng::R
    model::M
end

children(node::ModifierNode) = (node.wrapped_node,)

function rand_barrier(node::ModifierNode, variables::NamedTuple, dims...)
    wrapped_value = rand_barrier(node.wrapped_node, variables, dims...)
    rand(rng(node), node(variables), wrapped_value)
end

function logdensityof_barrier(node::ModifierNode, variables::NamedTuple)
    wrapped_ℓ = logdensityof_barrier(node.wrapped_node, variables)
    logdensityof(node(variables), varvalue(node, variables), wrapped_ℓ)
end

bijector_barrier(node::ModifierNode, variables::NamedTuple) = bijector_barrier(node.wrapped_node, variables)

"""
    SumLogdensityModifier
Sums up the logdensities of the wrapped_node, e.g. if multiple data points are available for a variable.
Random generation is not modified.
"""
struct SumLogdensityModifier end
SumLogdensityModifier(args...) = SumLogdensityModifier()
Base.rand(::AbstractRNG, model::SumLogdensityModifier, x) = x
DensityInterface.logdensityof(::SumLogdensityModifier, x, ℓ) = sum(ℓ)
