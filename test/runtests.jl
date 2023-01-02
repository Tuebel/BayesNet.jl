# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base: materialize
using BayesNet
using CUDA
using DensityInterface
using KernelDistributions
using Random
using Test

# Setup a list of rngs to loop over
cpurng = Random.default_rng()
Random.seed!(cpurng, 42)
curng = CUDA.default_rng()
Random.seed!(curng, 42)
# TODO replace in test files
rngs = [cpurng, curng]

# BUG CUDA does weird stuff (wrong calculations, minimum is always 0.0) only on my laptop during Pkg.test() not when include("runtests.jl")
CUDA.allowscalar(false)

include("simple.jl")
include("sequentialized.jl")
include("broadcasted.jl")
include("deterministic.jl")
include("modifier.jl")