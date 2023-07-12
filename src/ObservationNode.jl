# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
    ObservationNode
Allows to condition another node on data.
Handles inserting additional dims for broadcasting and summing up the per observation loglikelihoods accordingly.
Currently only BroadcastedNode and ModifierNode are tested.
"""
struct ObservationNode{name,child_names,N<:AbstractNode{name,child_names},O} <: AbstractNode{name,child_names}
    wrapped_node::N
    observation::O
end

"""
    |(node::BroadcastedNode, observation)
Syntactic sugar for conditioning a broadcasted `node` on an `observation`
"""
Base.:|(node::BroadcastedNode, observation) = ObservationNode(node, observation)
Base.:|(node::ModifierNode, observation) = ObservationNode(node, observation)

# Forward wrapped fields
children(node::ObservationNode) = children(node.wrapped_node)
model(node::ObservationNode) = model(node.wrapped_node)
rng(node::ObservationNode) = rng(node.wrapped_node)

# Construct as leaf
"""
    ObservationNode(name, rng, distribution, params...)
Construct the node as leaf (no children) by broadcasting the `distribution` over the `params`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims of the `params`.
"""
ObservationNode(name::Symbol, rng::AbstractRNG, distribution, params...) = ObservationNode(name, rng, BroadcastedDistribution(distribution, params...), param_dims(params...), (), ())

function logdensityof_barrier(node::ObservationNode{name,child_names}, variables::NamedTuple) where {name,child_names}
    # BroadcastedNode inserts dims for compatibility of child nodes
    wrapped_values = childvalues(node.wrapped_node, variables)
    child_values = map(wrapped_values) do wv
        insert_observation_dims(wv, model_dims(node), size(node.observation))
    end
    child_vars = NamedTuple{child_names}(child_values)
    variables = merge(variables, child_vars)
    # Do not use the value from variables but the observation instead
    # Use the broadcasted distribution for the reduction of the model_dims
    ℓ = node_logdensityof(node, variables)
    # First dims match the number of observations, last dims the number of samples
    sum_and_dropdims(ℓ, observation_dims(model_dims(node), size(node.observation)))
end

# Do not use node(variables) as it would invoke childvalues and insert dims again
node_logdensityof(node::ObservationNode{<:Any,child_names,<:BroadcastedNode}, variables) where {child_names} = logdensityof(node.wrapped_node(values(variables[child_names])...), node.observation)

# Do not use logdensityof_barrier(variables) as it would invoke childvalues and insert dims again
function node_logdensityof(node::ObservationNode{<:Any,child_names,<:ModifierNode}, variables) where {child_names}
    # Observation node wraps another node wrapper
    # BUG breaks if modifier wraps modifier
    modifier_node = node.wrapped_node
    internal_node = modifier_node.wrapped_node
    wrapped_ℓ = logdensityof(internal_node(values(variables[child_names])...), node.observation)
    logdensityof(modifier_node(variables), node.observation, wrapped_ℓ)
end

# Do not call rand on the wrapped_node but directly insert the observation
rand_barrier(node::ObservationNode, variables::NamedTuple, dims...) = node.observation

"""
insert_observation_dims(A, model_dims::Dims, observation_size::Dims)
    `model_dims` are the dimensions of the broadcasted model, aka of the wrapped node.
    `observation_dims` are the dimensions of the observation
"""
function insert_observation_dims(A, model_dims::Dims{M}, observation_size::Dims{N}) where {M,N}
    # How many dims have been added for the observation
    n_observation_dims = N - M
    fill_ones = ntuple(_ -> 1, n_observation_dims)
    # Of a single sample which has been broadcasted
    single_size = size(A)[1:M]
    # How many samples have been drawn
    multi_size = size(A)[M+1:end]
    reshape(A, single_size..., fill_ones..., multi_size...)
end
# CUDA compatibility - do not wrap in vector
insert_observation_dims(A::Real, model_dims::Dims, observation_size::Dims) = A

"""
    observation_dims(model_dims::Dims, observation_size::Dims)
Extract the dims to sum over in the logdensityof_barrier. 
"""
observation_dims(model_dims::Dims{M}, observation_size::Dims{N}) where {N,M} = (1:(N-M)...,)

model_dims(node::ObservationNode) = model_dims(node.wrapped_node)
