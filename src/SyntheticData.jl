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
    # p_meas = zeros(size(p))
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
        #For now make sure it's normalize by diving by sum(p)
        p[i,:] = p[i,:]/sum(p[i,:])
        p_obs_history[i,:] = M_spam*p[i,:]
        # println("sum of p_obs at step $i: ", sum(p_obs_history[i,:]))
        p_distribution = Multinomial(samples, p_obs_history[i,:])
        counts = rand(p_distribution)
        p_hat_history[i,:] = counts ./ samples
    end
    return p_hat_history, p_obs_history 
end

function create_synthetic_data_full(p::AbstractArray, samples::Int64 = 10000, epsilon_Mspam::Union{Nothing, Vector{Float64}} = nothing)
    n_levels, steps, init_conds = size(p)
    # p_meas = zeros(size(p))
    if epsilon_Mspam == nothing 
        epsilon_Mspam = 1E-1*randn(n_levels)
    elseif epsilon_Mspam != nothing 
        if length(epsilon_Mspam) != n_levels
            error("Length of random epsilon vector must match with size of probability vector length")
        end
    end
    M_spam = column_stochastic(n_levels, epsilon_Mspam)
    p_hat_history = zeros(size(p))
    p_obs_history = zeros(size(p))
    for i in 1:steps
        for j in 1:init_conds 
        #For now make sure it's normalize by diving by sum(p)
            p[:,i,j] = p[:,i,j]/sum(p[:,i,j])
            p_obs_history[:,i,j] = M_spam*p[:,i,j]
            # println("sum of p_obs at step $i: ", sum(p_obs_history[i,:]))
            p_distribution = Multinomial(samples, p_obs_history[:,i,j])
            counts = rand(p_distribution)
            p_hat_history[:,i,j] = counts ./ samples
        end
    end
    return p_hat_history, p_obs_history 
end

function create_synthetic_vector(p::AbstractVector, samples::Int64 = 10000, epsilon_M_spam::Union{Nothing, Vector{Float64}} = nothing)
    vec_length = length(p)
    
    if epsilon_Mspam == nothing 
        epsilon_Mspam = 1E-1*rand(vec_length)
    elseif epsilon_Mspam != nothing 
        if length(epsilon_Mspam) != vec_length 
            error("Length of random epsilon vector must match with size of probability vector length")
        end
    end
    M_spam = column_stochastic(vec_length, epsilon_Mspam)

    p_obs = M_spam*p 
    p_distribution = Multinomial(samples, p_obs)
    counts = rand(p_distribution)
    p_hat = counts ./ samples 

    return p_hat, p_obs 
end

#Calculate fidelity between 2 probability vectors
function fidelity_probability(a::AbstractVecOrMat{<: Number}, b::AbstractVecOrMat{<: Number})
    if typeof(a) == Vector{Float64} 
        fidelity = (sqrt.(a)'*sqrt.(b))^2
    elseif typeof(a) == Matrix{Float64}
        row, col = size(a)
        fidelity = 0 
        for i in 1:col 
            fidelity += (sqrt.(a[:,i])'*sqrt.(b[:,i]))^2
        end
        fidelity = fidelity/col
    end
    # fidelity = sum((sqrt.(a)'*sqrt.(b))^2)
    return fidelity 
end

#Calculate average over fidelities using fidelity_probability
function fidelity_probability_risk_neutral(Prob_array::Vector{Any}, target::AbstractArray{<: Number})
    fidelity = 0.0
    N = length(Prob_array)
    for i in 1:N 
        fidelity += fidelity_probability(Prob_array[i], target)
    end
    fidelity /= N 
    return fidelity 
end

function fidelity(a::AbstractVecOrMat{<: Number}, b::AbstractVecOrMat{<: Number})
    if typeof(a) == Vector{ComplexF64}
        fidelity = abs2.(a'*b)
    else
        row, col = size(a)
        fidelity = (1/col)^2*abs2(tr(a'*b))
    end
    return fidelity
end

#Calculate fidelity between two probabilty distributions using frobenius inner product
function fidelity_general(a::AbstractArray, b::AbstractArray)
    #normalize columns of each array a and b 
    if typeof(a) == Vector{Float64}
        a_norm = a/norm(a)
        b_norm = b/norm(b)
        fid = abs2(a_norm'*b_norm)
    else
        row_a, col_a = size(a)
        row_b, col_b = size(b)
        if (row_a, col_a) != (row_b, col_b)
            error("mismatch in dimensions of a and b!")
        end
        a_normal = zeros(row_a, col_a)
        b_normal = zeros(row_b, col_b)
        fid_sum = 0
        for i in 1:col_a 
            a_normal[:,i] = a[:,i]/norm(a[:,i])
            b_normal[:,i] = b[:,i]/norm(b[:,i])
            fid_sum += a_normal[:,i]'*b_normal[:,i]
        end
        fid = abs2(fid_sum)/(col_a^2)
    end
    return fid 
end
