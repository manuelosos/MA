using Revise

using Graphs
using Graphs.SimpleGraphs
using Distributions
using LinearAlgebra
using Plots



using .DynamicsOnNetworkSimulation


function meanFieldComplete(x, R, Rt)

    res = zeros(size(x))

    for n in 1:size(R)[1]
        for m in 1:size(R)[1]
            if m == n 
                continue
            end
            tmp = x[m]*(R[m,n]* x[n] + Rt[m,n])

            res[n] += tmp
            res[m] -= tmp
        end
    end

    return res
end

"""Diffusion function for the Chemical Langevin Equation for a complete Network. """
function diffusionComplete(x, R, Rt, N)

    M = size(R)[1]

    diffusion = zeros((M,M^2-M)) # M is subtracted since diagonal elements receive no noise

    k = 1
    for i = 1:M
        for j = 1:M
            if i == j
                continue
            end
            tmp = x[j]* (R[j,i] * x[i] + Rt[j,i])
            if tmp < 0
                tmp = 0 # Clip negatve values to zero
            else
                tmp = sqrt(tmp/N) 
            end
            diffusion[j,k] = -tmp
            diffusion[i,k] = tmp
            k += 1
            
        end 
    end

    return diffusion
end





