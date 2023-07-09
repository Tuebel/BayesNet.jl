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
export ModifierNode, SumLogdensityModifier
export SequentializedGraph
export SimpleNode

export add_logdensity
export children
export evaluate
export isleaf
export nodename
export parents
export prior
export sequentialize

using Reexport
@reexport import DensityInterface: logdensityof

end # module BayesNet
