# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    add_logdensity(A, B)
Add logdensities from different devices by converting them to CPU Arrays explicitly.
"""
add_logdensity(A, B) = A .+ B
add_logdensity(A::Array, B::Array) = A .+ Array(B)
add_logdensity(A::Array, B::AbstractArray) = A .+ Array(B)
add_logdensity(A::AbstractArray, B::Array) = Array(A) .+ B