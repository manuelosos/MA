using DifferentialEquations
using Plots
using DifferentialEquations.EnsembleAnalysis

r_ab = 0.99
r_ba = 1
rt_ab = 0.01
rt_ba = 0.01
u0 = [0.2,0.8]

R = [0.0   0.99; 
     1.0   0.0] 

Rt = [0    0.01;
      0.01 0.0]

#R = [0.0 0.8 0.2;
#     0.2 0.0 0.8;
#     0.8 0.2 0.0]

#Rt = [0 0.01 0.01;
#      0.01 0 0.01;
#      0.01 0.01 0]
#u0 = [0.2,0.5,0.3]

M = size(R)[1]
N=10000

timespan = (0, 200)


function tmp_f(du, u, p, t)
    
    du[1] = (r_ba-r_ba)*u[1]*(1-u[1]) - rt_ab*u[1] + rt_ba*(1-u[1])   
    
    
end




function mfe_complete(du, u, p, t)

    for i in 1:M
        du[i] = 0
    end
    
    for n in 1:M
        for m in 1:M
            if m == n 
                continue
            end
            tmp = u[m]*(R[m,n]* u[n] + Rt[m,n])
            du[n] += tmp
            du[m] -= tmp
        end
    end

    return du
end

function noise_complete(du, u, p, t)
    du[1] = -1/sqrt(N)*sqrt(max(0, r_ab * u[1] * (1-u[1])))
    du[2] = 1/sqrt(N)*sqrt(max(0, r_ba * u[1] * (1-u[1])))
    du[3] = -1/sqrt(N)*sqrt(min(1, rt_ab * u[1]))
    du[4] = 1/sqrt(N)*sqrt(max(0, rt_ba * (1-u[1])))
end


function noise_erdos_renyi(du, u, p, t)
    du[1]=1

end


function get_fs(model)
    if model == "complete"
        return mfe_complete, noise_complete
    end

end


function ode(du, u, p, t, R, Rt)
    
    M = size(R)[1]

    du = (R * u + Rt * ones(M)) .* u
end


function main()


    model = "complete"

    f, g = get_fs(model)


    sde_prob = SDEProblem(tmp_f, g, [0.2], timespan, noise_rate_prototype = zeros(1,4))
    
    ensembleprob = EnsembleProblem(sde_prob)
    sde_ens_sol = solve(ensembleprob, EnsembleThreads(), trajectories = 100 )

    ode_prob = ODEProblem(f, u0, timespan )


    
    ode_sol = solve(ode_prob)

    

    

    summ = EnsembleSummary(sde_ens_sol, 0:0.01:timespan[2])
    plot(summ, labels = "Middle 95%")
    summ = EnsembleSummary(sde_ens_sol, 0:0.01:timespan[2]; quantiles = [0.25, 0.75])
    plot!(summ, labels = "Middle 50%", legend = true)
    
    plot!(ode_sol, labels = "MFE ODE")
end




function full_model(r, rt)


    R = transpose(r) - r 
    Rt = transpose(rt) - rt

    u0 = [0.2, 0.8]

    M = size(r)[1]
    #println("value")
    #print(u0)
    #println("Ru0 ",((transpose(r)-r)*u0) .* u0 )
    #println((R * u0 + Rt * ones(M)) .* u0)
    #return
    

    setindex!.(Ref(R), 0.0, 1:M, 1:M)
    
    timespan = (0, 400)

    #f = (du, u, p, t) -> ode(du, u, p, t, R, Rt)
    f(u, p, t) = (R * u + Rt * ones(M)) .* u
    print(f(u0,0,0))
    ode_prob = ODEProblem(f, u0, timespan )
    ode_sol = solve(ode_prob) #Euler(), dt= 0.01)

    plot!(ode_sol)

    


end
main()

#full_model([0 0.99; 1 0], [0 0.01; 0.01 0])