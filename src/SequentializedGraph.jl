# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
    SequentializedGraph
A graph that can be executed in sequence compared to traversing the graph.
Since Bayesian networks are DAGs, they can always be sequentialized.
This allows to implement type stable functions by unrolling the loop over the sequence.
"""
const SequentializedGraph = NamedTuple{<:Any,<:Tuple{Vararg{AbstractNode}}}

"""
    sequentialize(node)
Finds the shortest path where each node is executed exactly once via depth search.
The result is an ordered NamedTuple. 
"""
sequentialize(node::AbstractNode) =
    traverse(node, (;)) do node, _
        node
    end
sequentialize(graph::SequentializedGraph) = graph

"""
    rand(graph, [variables, dims...])
Type stable implementation to generate random values from the variables of the sequentialized graph.
The `variables` parameter allows to condition the model and will not be re-sampled.
"""
Base.rand(graph::SequentializedGraph, variables::NamedTuple, dims::Integer...) = rand_unroll(values(graph), variables, dims...)
Base.rand(graph::SequentializedGraph, dims::Integer...) = rand(graph, (;), dims...)

# unroll required for type stability
@unroll function rand_unroll(graph, variables, dims::Integer...)
    @unroll for node in graph
        if !(nodename(node) in keys(variables))
            value = rand_barrier(node, variables, dims...)
            variables = merge_value(variables, node, value)
        end
    end
    variables
end

"""
    evaluate(graph, variables)
Type stable version to only the deterministic nodes in the `graph` given the random `variables`.
All required random variables are assumed to be available.
"""
evaluate(graph::SequentializedGraph, variables::NamedTuple) = evaluate_unroll(values(graph), variables)

# unroll required for type stability
@unroll function evaluate_unroll(graph, variables)
    @unroll for node in graph
        value = evaluate_barrier(node, variables)
        variables = merge_value(variables, node, value)
    end
    variables
end

"""
    logdensityof(graph, nt)
Type stable implementation to calculate the logdensity for a set of variables for the sequentialized graph.
"""
DensityInterface.logdensityof(graph::SequentializedGraph{names}, nt::NamedTuple) where {names} =
    reduce(add_logdensity, map(values(graph)) do node
        logdensityof_barrier(node, nt)
    end)
# still don't get why reduce(.+, map...) is type stable but mapreduce(.+,...) not

# Support for empty models (e.g. no prior)
DensityInterface.logdensityof(graph::SequentializedGraph{()}, nt::NamedTuple) = 0

"""
    bijector(node)
Infer the bijectors of the sequentialized graph.
"""
function Bijectors.bijector(graph::SequentializedGraph)
    variables = rand(graph)
    map(x -> bijector_barrier(x, variables), graph)
end

"""
    parents(graph::SequentializedGraph, node_name)
Returns a SequentializedGraph for the parents of the `node_name` from the sequentialized `graph`.
"""
function parents(graph::SequentializedGraph{names}, node_name) where {names}
    # only nodes right of node_name might be parents
    node_idx = findfirst(x -> x == node_name, names)
    right = graph[names[node_idx:end]]
    # unrelated nodes might be in the list
    parents = ()
    for node in right
        # Add direct parent
        if any(x -> x == node_name, childnames(node))
            parents = (parents..., node)
            # only add once
            continue
        end
        # Add parent of parent
        for parent_node in parents
            if any(x -> x == nodename(parent_node), childnames(node))
                parents = (parents..., node)
                # only add once
                break
            end
        end
    end
    parent_names = nodename.(parents)
    NamedTuple{parent_names}(parents)
end

parents(graph::SequentializedGraph{names}, nodes::AbstractNode...) where {names} =
    reduce(nodes; init=(;)) do accumulated, node
        nt = parents(graph, nodename(node))
        # Merge only nodes which are not present in the evaluation model yet
        diff_nt = Base.structdiff(nt, accumulated)
        merge(accumulated, diff_nt)
    end