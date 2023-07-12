# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# TODO not intuitive, makes the ObservationNode hacky and I would like to get rid of it.

"""
    ModifierNode
Wraps another node and represents the same variable as the `wrapped` node.
For the model create a new type e.g. ModifierModel and implement:
* `rand(rng,::AbstractRNG, model::ModifierModel, x)`, where `x` is a randomly drawn value from the wrapped node.
* `logdensityof(model::ModifierModel, x, ℓ)`, where `x` is the value of the evaluated variable and `ℓ` the logdensity returned by the wrapped model

When traversing the graph, the children of the wrapped node are returned. 
"""
struct ModifierNode{name,child_names,N<:AbstractNode{name,child_names},R<:AbstractRNG,M} <: AbstractNode{name,child_names}
    wrapped_node::N
    rng::R
    model::M
end

# Forward methods to wrapped node
children(node::ModifierNode) = children(node.wrapped_node)
childvalues(node::ModifierNode, nt::NamedTuple) = childvalues(node.wrapped_node, nt)
model_dims(node::ModifierNode) = model_dims(node.wrapped_node)

function rand_barrier(node::ModifierNode, variables::NamedTuple, dims...)
    wrapped_value = rand_barrier(node.wrapped_node, variables, dims...)
    rand(rng(node), node(variables), wrapped_value)
end

function logdensityof_barrier(node::ModifierNode, variables::NamedTuple)
    wrapped_ℓ = logdensityof_barrier(node.wrapped_node, variables)
    logdensityof(node(variables), varvalue(node, variables), wrapped_ℓ)
end

bijector_barrier(node::ModifierNode, variables::NamedTuple) = bijector_barrier(node.wrapped_node, variables)
