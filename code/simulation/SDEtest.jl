using Revise
using Distributions
using Plots


includet("Statistics.jl")


function brownianMotion(timesteps::Vector, d=1)
    
    n = length(timesteps)
    normal_dist = Normal()
    rand_incs = rand(normal_dist, (n, d))
    x_traj = zeros(n, d)

    for i = 2:n
        x_traj[i, :] .= x_traj[i-1, :] .+ sqrt(timesteps[i] - timesteps[i-1]) * rand_incs[i-1,:]

    end
    return x_traj
end


function eulerMaruyama(drift, diffu, diff_shape, x0, time::Vector)

    if diffusion_shape == 1 
        x0 = [x0]
    end

    n_steps = length(time)
    traj = zeros(n_steps, d)
    traj[1, :] = copy(x0)
    randinc = brownianMotion(time, d)

    
    for i = 1:n_steps-1

        Δt = time[i+1]-time[i]
        traj[i+1,:] .= traj[i,:] .+ drift(traj[i,:]).*Δt + diffu(traj[i,:]) * (randinc[i+1,:].-randinc[i,:])
        
    end 

    return traj
end


function eulerMaruyama(a, b, n_states, x0, t0, tmax, timesteps)

    time = [t0+i*(tmax-t0)/timesteps for i = 0:timesteps-1]


    return time, eulerMaruyama(a, b, n_states, x0, time)

end 


function simulateEnsembleEulerMaruyama(a, b, n_states, x0, t0, tmax, timesteps, n_runs)

    timesteps = [t0+i*(tmax-t0)/timesteps for i = 0:timesteps-1]

    x_traj_list = zeros(length(timesteps), n_states, n_runs)

    Threads.@threads for i = 1:n_runs
        x_traj_list[:,:, i] = eulerMaruyama(a, b, n_states, x0, timesteps)
    end
    return timesteps, x_traj_list
end


function ornsteinuhlenbeck(equilibrium, stiffness, diffusion, initial)

    μ = equilibrium
    θ = stiffness
    σ = diffusion
    a = initial

    a(x) = θ*(μ' .- x)

    b(x) = σ    

    return a, b

end 

function testou2()
    x0 = [-3 -3]
    t0 = 0
    t_max = 10
    n_t = 1000
    
    n_runs = 30
    
    equilibrium = [1 1]
    stiffness = [2 1
                 1 2]
    diffusion = [0.5 0
                 0 0.5]

    

    drift, diffusion = ornsteinuhlenbeck(equilibrium, stiffness, diffusion, x0)

    t,x = eulerMaruyama(drift, diffusion, 2, x0, t0, t_max, 10000)


    plot(x[:,1], x[:,2])
    #t, x = simulateEnsembleEulerMaruyama(drift, diffusion,2, x0, t0, t_max, n_t, n_runs)

end


function test()

    x0 = -4
    t0 = 0
    t_max = 10
    n_t = 1000
    
    n_runs = 30

    drift, diffusion = ornsteinuhlenbeck(1, 2, 2, x0)
    #drift, diffusion = ornsteinuhlenbeck([1 1], [[1 1] [1 1]], [[1 0] [0 1]], [-2 2])

    #t,x = simulateEnsembleEulerMaruyama(drift, diffusion,1, x0, t0, t_max, n_t, n_runs)
    t,x = eulerMaruyama(drift, diffusion,1, x0, t0, t_max, n_t)

    plot(t,x)

    #plotEnsemble(x, t, t_max)
end

#test()

collect(Iterators.flatten((3,(5,9))))


