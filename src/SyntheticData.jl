using Distributions, LinearAlgebra, Plots, ProgressMeter
using TensorMethods 
using QuantumGateDesign

# include("main.jl")
# include("solvers.jl")



# H_mat = H_drift_mat(nqubits, nlevels, freq01_all, rotfreq, self_kerr, zz, Jkl)

# steps = 100
# ans_vec, sol_history, _, _ = td_exp_solver(H_mat, bc_params, init_vec, nqubits, 0, 1, steps)


# p_history = abs2.(sol_history)
# p_hat_history = zeros(steps + 1, 2^nqubits)
# for i in 1:steps + 1
#     println("Step $i")
#     p = abs2.(sol_history[i,:])
#     p_obs = M_spam*p 
#     p_distribution = Multinomial(samples, p_obs)
#     counts = rand(p_distribution)
#     p_hat = counts ./ samples 
#     p_hat_history[i,:] = p_hat 
# end

function column_stochastic(N::Int64, epsilon_list::Array{Float64})
    # Ident = Matrix(1.0*I, N, N)
    M = zeros(N, N)
    row, col = size(M)
    for i in 1:col
        col_vec = zeros(row)
        col_vec[1] = 1 - epsilon_list[i]
        remaining_elements = rand(row - 1)
        remaining_sum = sum(remaining_elements)
        remaining_elements *= epsilon_list[i]/remaining_sum 
        col_vec[2:row] = remaining_elements 
        col_vec[1], col_vec[i] = col_vec[i], col_vec[1]
        M[:,i] = col_vec 
    end
    return M 
end

function create_synthetic_data(p::AbstractArray, samples::Int64 = 10000, epsilon_Mspam::Union{Nothing, Vector{Float64}} = nothing)
    steps, vec_length = size(p)
    if epsilon_Mspam == nothing 
        epsilon_Mspam = 1E-1*rand(vec_length)
    elseif epsilon_Mspam != nothing 
        if length(epsilon_Mspam) != vec_length 
            error("Length of random epsilon vector must match with size of probability vector length")
        end
    end
    M_spam = column_stochastic(vec_length, epsilon_Mspam)
    p_hat_history = zeros(steps, vec_length)
    p_obs_history = zeros(steps, vec_length)
    for i in 1:steps 
        p_hat_history[i,:] = M_spam*p[i,:]
        p_distribution = Multinomial(samples, p_hat_history[i,:])
        counts = rand(p_distribution)
        p_obs_history[i,:] = counts ./ samples
    end
    return p_obs_history, p_hat_history 
end

# p_obs_history, p_hat_history = create_synthetic_data(p_history, 1000, [1E-2, 1E-2])



        




# plot(LinRange(t0, T, steps + 1), p_history, labels = ["True: |0>" "True: |1>"])
# plot!(LinRange(t0, T, steps + 1), p_obs_history, labels = ["Observed: |0>" "Observed: |1>"], legend =:outertop, legend_columns = 2)
#After getting the data, perform abs2.() and then draw from distribution using Multinomial(). Then normalize that new data
#Look at confusion matrix?