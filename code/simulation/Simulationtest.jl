using Revise

using Graphs
using Graphs.SimpleGraphs
using Distributions
using LinearAlgebra
using Plots

#includet("Gillespie.jl")
#includet("SDE.jl")
includet("DynamicsOnNetworkSimulation.jl")
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


function standardCNVMTest()
    
    num_nodes = 500
    num_states = 3
    t_max = 2
    n_traj = 20
    r = [0 2 2
        2 0 1
        1 2 0]

    rt = [0 0.1 0.1
          0.1 0 0.1
          0.1 0.1 0]


    g = complete_graph(num_nodes)

    model = CNVM(num_states, r, rt, g)

    x_init = rand(1: num_states, num_nodes)

    t_gillespie, x_gillespie = ensembleGillespie(model, t_max, x_init, n_traj)
    
    cv_gillespie = computeCvFromTrajectory(x_trajs, 3)

    t_gillespie, x_gillespie = unifyEnsembleTime(cv_gillespie, t_gillespie, t_max)

    plotEnsemble(x_gillespie, t_gillespie, t_max)
    print("done")

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
    t,x = eulerMaruyama(drift, diffusion,(1,1), x0, t0, t_max, n_t)

    plot(t,x)

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


testMFE()


