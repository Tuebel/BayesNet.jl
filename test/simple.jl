# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "SimpleNode, RNG: $rng" for rng in rngs
    # Leaf node
    a = SimpleNode(:a, rng, KernelUniform)
    @test isleaf(a)
    @test a(1, 2, 3) isa KernelUniform
    @test Base.broadcastable(a).x == a

    # Building graph
    b = SimpleNode(:b, rng, KernelExponential)
    c = SimpleNode(:c, rng, KernelNormal, (a, b))
    d = SimpleNode(:d, rng, KernelNormal, (c, b))
    @test isnothing(show(d))

    # rand
    nt = rand(d, (; a=1))
    @test nt.a == 1
    nt = rand(d)
    @test nt.a != 1
    @test nt.d isa Float32

    # logdensity
    ℓ = logdensityof(d, nt)
    @test ℓ isa Float32
    @test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)
    bij = bijector(d)

    # Bijectors
    @test bij isa NamedTuple{(:a, :b, :c, :d)}
    @test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

    # multiple samples not supported
    @test_throws MethodError rand(d, 2)

    # prior extraction
    prior_d = prior(d)
    @test prior_d == (; a=a, b=b, c=c)

    # parent extraction
    parent_a = parents(d, :a)
    @test parent_a == (; c=c, d=d)
    parent_c = parents(d, :c)
    @test parent_c == (; d=d)
    parent_ac = parents(d, a, c)
    @test parent_ac == (; c=c, d=d)
    parent_ba = parents(d, b, a)
    @test parent_ba == (; c=c, d=d)
end