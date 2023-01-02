# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "add_logdensity scalar" begin
    y = @inferred add_logdensity(1, 2)
    @test y == 3
    y = @inferred add_logdensity(1, [2])
    @test y == [3]
    y = @inferred add_logdensity([1, 2], 1)
    @test y == [2, 3]
    y = @inferred add_logdensity(1, CuArray([1, 2]))
    @test y == CuArray([2, 3])
    y = @inferred add_logdensity(CuArray([1, 2]), 1)
    @test y == CuArray([2, 3])
end

@testset "add_logdensity same array type" begin
    y = @inferred add_logdensity([1 2; 3 4; 5 6], [1 2])
    @test y == [2 4; 4 6; 6 8]
    y = @inferred add_logdensity([1 2], [1 2; 3 4; 5 6])
    @test y == [2 4; 4 6; 6 8]
    y = @inferred add_logdensity(CuArray([1 2; 3 4; 5 6]), CuArray([1 2]))
    @test y == CuArray([2 4; 4 6; 6 8])
    y = @inferred add_logdensity(CuArray([1 2]), CuArray([1 2; 3 4; 5 6]))
    @test y == CuArray([2 4; 4 6; 6 8])
end

@testset "add_logdensity different array type" begin
    y = @inferred add_logdensity([1 2; 3 4; 5 6], CuArray([1 2]))
    @test y == [2 4; 4 6; 6 8]
    y = @inferred add_logdensity([1 2], CuArray([1 2; 3 4; 5 6]))
    @test y == [2 4; 4 6; 6 8]
end