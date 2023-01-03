module BayesNet

using Bijectors
using DensityInterface
using KernelDistributions
using Random
using Unrolled

# General graph functions and sequentialization for type stability
include("Math.jl")
include("Graph.jl")
include("SequentializedGraph.jl")

include("BroadcastedNode.jl")
include("DeterministicNode.jl")
include("ModifierNode.jl")
include("SimpleNode.jl")

export AbstractNode
export BroadcastedNode
export DeterministicNode
export ModifierNode
export SequentializedGraph
export SimpleNode

export add_logdensity
export children
export evaluate
export isleaf
export name
export parents
export prior
export rng
export sequentialize

end # module BayesNet
