# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using BayesNet
using DensityInterface
using KernelDistributions
using Random
using Test

DATA_SIZES = ((), (1,), (3,), (3, 4))
a = BroadcastedNode(:a, rng, KernelNormal, 1.0f0, 2.0f0)
b = BroadcastedNode(:b, rng, KernelExponential, [1.0f0, 2.0f0])
c = BroadcastedNode(:c, rng, KernelNormal, (a, b))

@testset "ObservationNode single sample, RNG: $rng, DATA_SIZE: $data_size" for rng in rngs, data_size in DATA_SIZES
    # Condition on data, aka override the randomly generated values with the observed ones
    data = rand(rng, BroadcastedDistribution(KernelNormal, 0, [2.0f0, 3.0f0]), data_size...)
    @test size(data) == (2, data_size...)
    c_obs = c | data
    model = sequentialize(c_obs)

    # Type stable rand with correct values and dimensions
    sample = @inferred rand(model)
    @test sample.c == data
    @test sample.a isa Float32
    @test sample.b isa AbstractArray{Float32,1}
    @test size(sample.b) == (2,)

    # Single sample
    ℓ_sample = @inferred logdensityof(model, sample)
    @test ℓ_sample isa Float32
    # Compare to unmodified version adds the prior probability to each likelihood value
    # However, the prior should only be added once - more data → less influence of prior
    ℓ_prior = logdensityof(a, sample) + logdensityof(b, sample)
    @test ℓ_sample ≈ sum(logdensityof(c, sample)) - (prod(data_size) - 1) * ℓ_prior
end;

SAMPLE_SIZE = ((1,), (3,), (5, 6))
@testset "ObservationNode multiple observations, RNG: $rng, DATA_SIZE: $data_size, SAMPLE_SIZE: $sample_size" for rng in rngs, data_size in DATA_SIZES, sample_size in SAMPLE_SIZE
    # Condition on data, aka override the randomly generated values with the observed ones
    data = rand(rng, BroadcastedDistribution(KernelNormal, 0, [2.0f0, 3.0f0]), data_size...)
    @test size(data) == (2, data_size...)
    c_obs = c | data
    model = sequentialize(c_obs)

    # Type stable rand with correct values and dimensions
    sample_N = @inferred rand(model, sample_size...)
    @test sample_N.c == data
    @test sample_N.a isa AbstractArray{Float32,length(sample_size)}
    @test size(sample_N.a) == (sample_size...,)
    @test sample_N.b isa AbstractArray{Float32,length(sample_size) + 1}
    @test size(sample_N.b) == (2, sample_size...)
    # Sanity counter example
    sample_c = rand(c)
    @test sample_c.c != data

    # Type stable logdensityof with correct values and dimensions
    ℓ_sample = @inferred logdensityof(model, sample_N)
    @test ℓ_sample isa AbstractArray{Float32,length(sample_size)}
    @test size(ℓ_sample) == (sample_size...,)
    # Compare to unmodified version adds the prior probability to each likelihood value
    # However, the prior should only be added once - more data → less influence of prior
    ℓ_prior = logdensityof(a, sample_N) + logdensityof(b, sample_N)
    # TODO sort out reduction dims
    # ℓ_likel = sum_and_dropdims(logdensityof(c(sample_N[(:a, :b)]), reshape(data, 2, ntuple(_ -> 1, length(SAMPLES_SIZE))..., data_size...)), ((1+length(SAMPLES_SIZE)):(1+length(SAMPLES_SIZE)+length(data_size))...,))
    # @test logdensityof(model, sample_N) ≈ ℓ_prior .+ ℓ_likel
end;