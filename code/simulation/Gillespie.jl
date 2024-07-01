
function gillespie(x_init::Vector,
                  t_max,
                  num_states,
                  neighbor_list,
                  r_imit,
                  r_noise,
                  prob_imit,
                  prob_noise,
                  degrees  
                    )

    num_nodes = length(degrees)

    # Setting up exponential RV X with E[X] = event rate
    next_event_rate = 1 / (r_imit * sum(degrees)+ r_noise * num_nodes)
    event_distance_distribution = Exponential(next_event_rate)

    noise_probability = r_noise * num_nodes * next_event_rate
    
    prob_table, alias_table = buildAliasTable(degrees)

    t=0
    x_traj = [copy(x_init)]
    t_traj = [0.]

    x = x_init

    while t < t_max
        t += rand(event_distance_distribution)
        noise = rand() < noise_probability ? true : false 

        if noise
            node = rand(1:num_nodes)
            new_state = rand(1:num_states)
            if rand() < prob_noise[x[node], new_state]
                x[node] = new_state
            end
        else
            node = sampleFromAlias(prob_table, alias_table)
            neighbors = neighbor_list[node]
            rand_neighbor = neighbors[rand(1:length(neighbors))]
            new_state = x[rand_neighbor]
            if rand() < prob_imit[x[node], new_state]
                x[node] = new_state
            end
        end

        push!(x_traj, copy(x))
        push!(t_traj, t)

    end

    return t_traj, x_traj
end


function buildAliasTable(weights)  
    # Algorithm from Kronmal and Peterson; see alias method for more information 
    table_prob = weights / sum(weights) * length(weights)
    table_alias = ones(Integer, length(weights))

    small_ids = [i for i=1:length(table_prob) if table_prob[i] < 1]
    large_ids = [i for i=1:length(table_prob) if table_prob[i] >= 1]

    while length(small_ids) > 0 && length(large_ids) > 0
        small_id = pop!(small_ids)
        large_id = pop!(large_ids)

        table_alias[small_id] = large_id
        table_prob[large_id] = table_prob[large_id] + table_prob[small_id] -1

        if table_prob[large_id] < 1
            push!(small_ids, large_id)
        else
            push!(large_ids, large_id)
        end
    end

    table_prob[large_ids] .= 1
    table_prob[small_ids] .= 1

    return table_prob, table_alias
end


function sampleFromAlias(table_prob::Vector{Float64}, table_alias::Vector{Integer})

    x = rand()
    idx = trunc(Int, x * length(table_prob))
    y = length(table_prob) * x - idx
    if y < table_prob[idx+1]
        return idx+1
    end

    return table_alias[idx+1]

end
