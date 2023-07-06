# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "DeterministicNode, RNG: $rng" for rng in rngs
    a_params = array_for_rng(rng, Float32, 3)
    a_params .= 1
    a = BroadcastedNode(:a, rng, KernelExponential, a_params)
    b_params = array_for_rng(rng, Float32, 2)
    b_params .= 1
    b = BroadcastedNode(:b, rng, KernelExponential, b_params)

    fn(a, ::Any) = a
    c = DeterministicNode(:c, fn, (a, b))

    nt = rand(c)
    @test nt.c isa AbstractArray{Float32,1}
    @test size(nt.c) == (3,)
    ℓ = logdensityof(c, nt)
    @test ℓ isa Float32
    @test ℓ == logdensityof(a, nt) + logdensityof(b, nt)

    nt = rand(c, 2)
    @test nt.c isa AbstractArray{Float32,2}
    @test size(nt.c) == (3, 2)
    ℓ = logdensityof(c, nt)
    @test ℓ isa AbstractArray{Float32,1}
    @test ℓ == logdensityof(a, nt) + logdensityof(b, nt)

    # Test that evaluate only calls the DeterministicNode
    nt = evaluate(c, (; a=1, b=2, c=3))
    @test nt == (; a=1, b=2, c=1)

    # Type stable in SequentializedGraph?
    seq_graph = sequentialize(c)
    nt = @inferred evaluate(seq_graph, (; a=1, b=2, c=3))
    @test nt == (; a=1, b=2, c=1)

    # Deterministic as leaf
    d = DeterministicNode(:d, () -> fill(42, 42))
    s = @inferred rand(d)
    @test logdensityof(d, s) == 0
    @test s.d == fill(42, 42)
end