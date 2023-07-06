# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
    DeterministicNode
This node only takes part in the generative `rand` process but is not random.
Instead a deterministic function `fn` is provided, e.g. rendering an image for a pose.
It does not change the joint logdensity of the graph by returning a logdensity of zero.
"""
struct DeterministicNode{name,child_names,M,C<:Tuple{Vararg{AbstractNode}}} <: AbstractNode{name,child_names}
    fn::M
    children::C
end

# Leaf node
DeterministicNode(name::Symbol, fn::M) where {M} = DeterministicNode{name,(),M,Tuple{}}(fn, ())

# Parent node
DeterministicNode(name::Symbol, fn::M, children::C) where {M,C<:Tuple{Vararg{AbstractNode}}} = DeterministicNode{name,nodename.(children),M,C}(fn, children)

rand_barrier(node::DeterministicNode, variables::NamedTuple, _...) = evaluate_barrier(node, variables)
evaluate_barrier(node::DeterministicNode, variables::NamedTuple) = node.fn(childvalues(node, variables)...)

# Do not change the joint probability - log probability of 0
logdensityof_barrier(node::DeterministicNode, variables::NamedTuple) = varvalue(node, variables) |> eltype |> zero

bijector_barrier(node::DeterministicNode, variables::NamedTuple) = ZeroIdentity()