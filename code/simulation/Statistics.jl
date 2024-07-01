function computeMeanVariance(traj_list)

    n_traj = size(traj_list)[3]
    n_states = size(traj_list)[2]
    n_points = size(traj_list)[1]
    mean = zeros(Float64,(n_points, n_states))
    variance = zeros(Float64,(n_points, n_states))

    for i = 1:n_traj
        for j = 1:n_states
        mean[:,j] .+= traj_list[:,j,i]
        end
    end
    mean ./= n_traj

    for i = 1:n_traj
        for j = 1:n_states
            variance[:,j] .+= (mean[:,j] .- traj_list[:,j,i]).^2

        end
    end

    variance ./= n_traj
    variance = sqrt.(variance)

    return mean, variance
end 


function computeMeanVarianceEnsembleDifference()

end


function plotEnsemble(
    ensemble_traj,
    ensemble_time,
    t_max
)
    n_traj = size(ensemble_traj)[3]
    n_states = size(ensemble_traj)[2]
    
    mean, variance = computeMeanVariance(ensemble_traj)

    for j = 1:n_states
        plt = plot(title="State $j", )
        for i = 1:n_traj
            plot!( ensemble_time, ensemble_traj[:,j,i], color="light gray")
        end

        plot!(ensemble_time, mean[:,j], color="red", legend=false)
        plot!(ensemble_time, mean[:,j] + variance[:,j], color="orange")
        plot!(ensemble_time, mean[:,j] - variance[:,j], color="orange")
        display(plt)    
    
    end

    
end