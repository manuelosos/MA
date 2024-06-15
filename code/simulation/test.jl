using Revise
using Graphs
using Graphs.SimpleGraphs
using Distributions
using LinearAlgebra
using Plots


includet("models.jl")


using .Models

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
    x_traj, t_traj = simulateSSA(model, 5, x_init)

    cv = computeCvFromTrajectory(x_traj, num_states)

    return x_traj, t_traj, cv
end


function standardCNVMTest()
    
    num_nodes = 500
    num_states = 3
    t_max = 20
    n_traj = 20
    r = [0 2 2
        2 0 1
        1 2 0]

    rt = [0 0.1 0.1
          0.1 0 0.1
          0.1 0.1 0]

    
    g = barabasi_albert(num_nodes, 2, 2)

    model = CNVM(num_states, r, rt, g)

    x_init = rand(1: num_states, num_nodes)

    x_trajs, t_trajs = simulateEnsembleSSA(model, t_max, x_init, n_traj)
    
    cvs = computeCvFromTrajectory(x_trajs, 3)

   # u_ttraj, u_xtraj = unifyEnsembleTime(cvs, t_trajs,100, t_max)


    plotEnsemble(cvs, t_trajs, t_max)

end


standardCNVMTest()