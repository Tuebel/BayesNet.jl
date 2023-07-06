# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "SequentializedGraph, RNG: $rng" for rng in rngs
    # Build & sequentialize graph
    a = SimpleNode(:a, rng, KernelUniform)
    b = SimpleNode(:b, rng, KernelExponential)
    c = SimpleNode(:c, rng, KernelNormal, (a, b))
    d = SimpleNode(:d, rng, KernelNormal, (c, b))
    seq_graph = sequentialize(d)
    @test seq_graph == (; a=a, b=b, c=c, d=d)

    # Type stable bijectors
    bij = @inferred bijector(seq_graph)
    @test bij isa NamedTuple{(:a, :b, :c, :d)}
    @test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

    # Type stable rand
    nt = @inferred rand(seq_graph, (; a=1))
    @test nt.a == 1
    nt = @inferred rand(seq_graph)
    @test nt.a != 1
    ℓ = @inferred logdensityof(seq_graph, nt)
    @test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)

    # Test BroadcastedNode
    a_params = array_for_rng(rng, Float32, 2)
    a_params .= 1
    a = BroadcastedNode(:a, rng, KernelUniform, 0, a_params)
    b = BroadcastedNode(:b, rng, KernelExponential, 1.0f0)
    c = BroadcastedNode(:c, rng, KernelNormal, (a, b))
    d = BroadcastedNode(:d, rng, KernelNormal, (c, b))
    seq_graph = sequentialize(d)

    nt = @inferred rand(seq_graph)
    ℓ = @inferred logdensityof(seq_graph, nt)
    @test ℓ ≈ logdensityof(KernelExponential(), nt.b) + sum(logdensityof.(KernelUniform(), nt.a) + logdensityof.(KernelNormal.(nt.a, nt.b), nt.c) + logdensityof.(KernelNormal.(nt.c, nt.b), nt.d))

    nt = @inferred rand(seq_graph, 3)
    ℓ = @inferred logdensityof(seq_graph, nt)
    @test ℓ ≈ logdensityof.(KernelExponential(), nt.b) + sum(logdensityof.(KernelUniform(), nt.a) + logdensityof.(KernelNormal.(nt.a, reshape(nt.b, 1, 3)), nt.c) + logdensityof.(KernelNormal.(nt.c, reshape(nt.b, 1, 3)), nt.d); dims=(1,))'
end