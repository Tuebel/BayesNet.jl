# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using BayesNet
using DensityInterface
using KernelDistributions
using Random
using Test

rngs = (Random.default_rng(),)

# Minimal implementation to test whether the values get modified and the rest of the graph is traversed
struct SimpleModifierModel end
# Construct with same args as wrapped model
SimpleModifierModel(args...) = SimpleModifierModel()
Base.rand(::AbstractRNG, model::SimpleModifierModel, value) = 10 * value
DensityInterface.logdensityof(::SimpleModifierModel, ::Any, ℓ) = ℓ + one(ℓ)

@testset "Simple ModifierNode, RNG: $rng" for rng in rngs
    a = SimpleNode(:a, rng, KernelUniform)
    b = SimpleNode(:b, rng, KernelExponential)
    c = SimpleNode(:c, rng, KernelNormal, (a, b))
    d = SimpleNode(:d, rng, KernelNormal, (c, b))
    d_mod = ModifierNode(d, rng, SimpleModifierModel)

    nt = rand(d_mod)
    @test logdensityof(d, nt) == logdensityof(d_mod, nt) - 1
    bij = bijector(d_mod)
    @test bij isa NamedTuple{(:a, :b, :c, :d)}
    @test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))
end

@testset "SumLogdensityModifier, RNG: $rng" for rng in rngs
    N = 42
    a = SimpleNode(:a, rng, KernelNormal, 1.0f0, 2.0f0)
    b = SimpleNode(:b, rng, KernelExponential)
    c = BroadcastedNode(:c, rng, KernelNormal, (a, b))
    c_mod = ModifierNode(c, rng, SumLogdensityModifier)
    model = sequentialize(c_mod)

    # Type stability of rand
    sample = @inferred rand(model)
    sample_N = @inferred rand(model, N)
    @test length(sample_N.c) == N

    # Condition on data, aka override the randomly generated values with the observed ones
    data = (; c=rand(model, N).c)
    @test length(data.c) == N
    conditioned_sample = (; sample..., data...)
    @test length(conditioned_sample.a) == 1
    @test length(conditioned_sample.b) == 1
    @test length(conditioned_sample.c) == N

    # Sum of a scalar is a scalar
    ℓ_sample = @inferred logdensityof(model, sample)
    @test ℓ_sample isa Float32
    # Must be the same as unmodified version
    @test logdensityof(model, sample) == logdensityof(c, sample)

    # Sum of vector is also a scalar
    ℓ_conditioned = @inferred logdensityof(model, conditioned_sample)
    @test ℓ_conditioned isa Float32
    # Compare to unmodified version adds the prior probability to each likelihood value
    # However, the prior should only be added once - more data → less influence of prior
    @test logdensityof(c, conditioned_sample) isa AbstractVector
    ℓ_prior = logdensityof(a, sample) + logdensityof(b, sample)
    @test logdensityof(model, conditioned_sample) ≈ sum(logdensityof(c, conditioned_sample)) - (N - 1) * ℓ_prior
end