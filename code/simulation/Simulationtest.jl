using Revise

using Graphs
using Graphs.SimpleGraphs
using Distributions
using LinearAlgebra
using Plots

#includet("Gillespie.jl")
#includet("SDE.jl")
includet("DynamicSimulation.jl")
includet("Statistics.jl")
includet("LangevinEquations.jl")


using .DynamicsOnNetworkSimulation

function simulateSIRS(r_si, rt_ir, rt_rs, network)

    num_states = 3
    r = [0 r_si 0
         0 0 0
         0 0 0]

    rt = [0 0 0
          0 0 rt_ir
          rt_rs 0 0 ]

    model = CNVM(num_states, r, rt, network)

    x_init = rand(1: num_states, nv(network))
    t, x_traj = gillespie(model, 5, x_init)

    cv = computeCvFromTrajectory(x_traj, num_states)

    return t, x_traj, cv
end

"""Computes the relative shares of each individual state of 
one superstate of the networkdynamics.
Arguments
    state : Superstate of the network
    n_states : Number of states that can occur.
     Assumes that the states are of kind [1:n_states]."""
function computeSharesOfState(state, n_states, normalized=true)
    shares = zeros(n_states)
    for i = 1:n_states
        shares[i] = count(x -> x==i, state)
    end
    if normalized == true
        return shares./length(state)
    end
    return shares
end


function standardCNVMTest()


    # CNVM Setup
    num_nodes = 500
    R = [0 0.8 0.2
        0.2 0 0.8
        0.8 0.2 0]
    Rt = [0 0.01 0.01
          0.01 0 0.01
          0.01 0.01 0]


    n_states = size(R)[1]
    #g = complete_graph(num_nodes)
    g = erdos_renyi(num_nodes, 0.03)
    

    model = CNVM(n_states, R, Rt, g)


    # General Simulation Setup
    x_init_gillespie = rand(1: n_states, num_nodes)
    
    
    x_init_sde = computeSharesOfState(copy(x_init_gillespie), n_states)
    x_init_sde = [0.2, 0.3, 0.5]
    t0 = 0
    t_max = 200
    n_time = 10*t_max
    n_runs = 5000

    # SDE
    
    noise_shape = n_states^2-n_states
    
    t_sde, x_sde = ensembleEulerMaruyama(
        x -> meanFieldComplete(x, R, Rt),
        #x-> zeros((n_states,n_states^2-n_states)),
        x -> diffusionComplete(x, R, Rt, num_nodes),
        n_states,
        noise_shape',
        x_init_sde,
        t0,
        t_max,
        n_time,
        n_runs
    )
    println("SDE Simulation done")

    
    # Gillespie Simulation

    
    t_gillespie, x_gillespie = ensembleGillespie(model, t_max, copy(x_init_gillespie), n_runs)
    println("Gillespie Simulation done")

    cv_gillespie = computeCvFromTrajectories(x_gillespie, n_states)

    t_gillespie, x_gillespie = unifyEnsembleTime(t_gillespie, cv_gillespie, t_sde)

    x_gillespie = x_gillespie ./ num_nodes



    #mean_sde, variance_sde = computeMeanVariance(x_sde)
    #mean_gillespie, variance_gillespie = computeMeanVariance(x_gillespie)

    savestr = "plots/"

    p1 = plotEnsemble(t_gillespie, x_gillespie, t_max, "Gillespie")
    p2 = plotEnsemble(t_sde, x_sde, t_max, "SDE", 0)

    savefig(p1,"$(savestr)plot_1.png" )
    savefig(p2,"$(savestr)plot_2.png" )
    
    #mean_difference = abs.(mean_sde .- mean_gillespie)
    #variance_difference = abs.(variance_sde .- variance_gillespie)


    #plotEnsembleDifference(t_sde, x_sde, x_gillespie,"Difference", "SDE", "Gillespie" )
    return
end


function ornsteinuhlenbeck(equilibrium, stiffness, diffusion, initial)

    μ = equilibrium
    θ = stiffness
    σ = diffusion
    a = initial

    a(x) = θ*(μ - x)

    b(x) = σ    

    return a, b

end


function testou2()
    x0 = [-3 -3]'
    t0 = 0
    t_max = 10
    n_t = 1000
    
    n_runs = 30
    
    equilibrium = [1 1]'
    stiffness = [2 1
                 1 2]
    diffusion = [0.5 0
                 0 0.5]

    

    drift, diffusion = ornsteinuhlenbeck(equilibrium, stiffness, diffusion, x0)

    t,x = eulerMaruyama(drift, diffusion, 2, (2,1), x0, t0, t_max, 10000)


    plot(x[:,1], x[:,2])
    #t, x = simulateEnsembleEulerMaruyama(drift, diffusion,2, x0, t0, t_max, n_t, n_runs)

end


function test()

    x0 = -4
    t0 = 0
    t_max = 10
    n_t = 2000
    
    n_runs = 30


    drift, diffusion = ornsteinuhlenbeck(1, 2, 2, x0)
    #drift, diffusion = ornsteinuhlenbeck([1 1], [[1 1] [1 1]], [[1 0] [0 1]], [-2 2])

    #t,x = simulateEnsembleEulerMaruyama(drift, diffusion,1, x0, t0, t_max, n_t, n_runs)
    #t,x = eulerMaruyama(drift, diffusion,(1,1), x0, t0, t_max, n_t)

    #plot(t,x)

    #plotEnsemble(x, t, t_max)
end


function testMFE()

    #CNVM
    R = [0.0   0.4; 
        1.0   0.0] 

    Rt = [0    0.03;
        0.01 0.0]

    N=100



    d = size(R)[1]
    noise_shape = d^2-d
    x0 = [0.3,0.7]
    t0=0
    t_max = 5
    n_time = 1000
    stepsize = 0.05

    n_runs = 20


    t, trajs = ensembleEulerMaruyama(
        x -> meanFieldComplete(x, R, Rt),
        x -> diffusionComplete(x, R, Rt, N),
        d,
        noise_shape,
        x0,
        t0,
        t_max,
        n_time,
        n_runs
    )

    plotEnsemble(trajs, t, t_max)

    end

standardCNVMTest()


