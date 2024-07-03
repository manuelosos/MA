function computeMeanVariance(traj_list)

    n_traj = size(traj_list)[3]
    n_states = size(traj_list)[2]
    n_points = size(traj_list)[1]
    mean = zeros(Float64, (n_points, n_states))
    variance = zeros(Float64, (n_points, n_states))

    for i = 1:n_traj
        for j = 1:n_states
            mean[:, j] .+= traj_list[:, j, i]
        end
    end
    mean ./= n_traj

    for i = 1:n_traj
        for j = 1:n_states
            variance[:, j] .+= (mean[:, j] .- traj_list[:, j, i]) .^ 2
        end
    end

    variance ./= n_traj
    variance = sqrt.(variance)

    

    return mean, variance
end


function plotEnsemble(
    ensemble_time,
    ensemble_traj,
    t_max,
    title="",
    plot_trajectories = 100
)
    n_traj = size(ensemble_traj)[3]
    n_states = size(ensemble_traj)[2]


    mean, variance = computeMeanVariance(ensemble_traj)

    plt = plot(layout=n_states)

    for j = 1:n_states
        plot!(title="State $j $title", subplot=j)

        for i = 1:min(n_traj, plot_trajectories)
            plot!(ensemble_time, ensemble_traj[:, j, i], color="light gray", subplot=j)
        end

        plot!(ensemble_time, mean[:, j], color="red", legend=false,subplot=j)
        plot!(ensemble_time, mean[:, j] + variance[:, j], color="orange",subplot=j)
        plot!(ensemble_time, mean[:, j] - variance[:, j], color="orange",subplot=j)

    end
    display(plt)
    return plt
end


function plotEnsembleDifference(time, traj_1, traj_2, 
    title="", traj_1_name="", traj_2_name=""
    )

    n_traj = size(traj_1)[3]
    n_states = size(traj_1)[2]

    mean_1, variance_1 = computeMeanVariance(traj_1)
    mean_2, variance_2 = computeMeanVariance(traj_2)

    for j = 1:n_states
        plt = plot(title="State $j $title")

        plot!(time, [mean_1[:, j] mean_2[:,j]], color=["red" "blue"],
         label=["$traj_1_name" "$traj_2_name"])
        plot!(time, mean_1[:, j] + variance_1[:, j], color="orange")
        plot!(time, mean_1[:, j] - variance_1[:, j], color="orange")

        plot!(time, mean_2[:, j] + variance_2[:, j], color="green")
        plot!(time, mean_2[:, j] - variance_2[:, j], color="green")

        display(plt)

    end




end