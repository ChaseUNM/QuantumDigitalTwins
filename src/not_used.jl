using LinearAlgebra
using Plots
using TensorMethods
#Step 1
#Forward Quantum Simulations

#First initialize a separable state with N subsystems each with 2 energy levels
# N = 2
# init_vec = zeros(2^N)
# init_vec[1] = 1.0 + 0.0*im 

# #Create Hamiltonian, for now use constant Hamiltonian
# t0 = 0.0
# T = 10.0
# J = 1.0
# g = 1.0
# H = xxx(N, J, g)
# #Evolve system and get resulting probability vector p
# sol_vec = exp(-im*H*(T - t0))*init_vec 
# p = abs2.(sol_vec)



#Step 2
#SPAM Errors
#Create SPAM matrix M, M needs to be unitary in order to preserve norm, however can be near identity, choose M_spam to just the first 2x2 block is affected

#Create M_spam matrix with 2x2 unitary blocks, with each block having an epsilon constant
function create_mspam(N::Int64, epsilon_list::Array{Float64})
    mat_size = 2^N 
    M_spam = Matrix(1.0*I, mat_size, mat_size)
    for i in 1:2^(N-1)
        local_unitary = [sqrt(1 - epsilon_list[i]^2) epsilon_list[i]; -epsilon_list[i] sqrt(1 - epsilon_list[i]^2)]
        println("Modidying the 2x2 matrix : U[$(2*(i - 1)):$(2*i), $(2*(i - 1)):$(2*i)]")
        M_spam[2*(i - 1) + 1:2*i, 2*(i - 1) + 1:2*i] = local_unitary
    end
    return M_spam 
end

function create_mspam_kron(N::Int64, epsilon_list::Array{Float64})
    init_matrix = [sqrt(1 - epsilon_list[1]^2) epsilon_list[1]; -epsilon_list[1] sqrt(1 - epsilon_list[1]^2)]
    for i in 2:N
        M = [sqrt(1 - epsilon_list[i]^2) epsilon_list[i]; -epsilon_list[i] sqrt(1 - epsilon_list[i]^2)]
        init_matrix = kron(M, init_matrix)
    end
    return init_matrix 
end

# function column_stochastic(N::Int64, epsilon_list::Array{Float64})
#     Ident = Matrix(1.0*I, 2^N, 2^N)
#     M = zeros(2^N, 2^N)
#     row, col = size(M)
#     for i in 1:col
#         col_vec = zeros(row)
#         col_vec[1] = 1 - epsilon_list[i]
#         remaining_elements = rand(row - 1)
#         remaining_sum = sum(remaining_elements)
#         remaining_elements *= epsilon_list[i]/remaining_sum 
#         col_vec[2:row] = remaining_elements 
#         col_vec[1], col_vec[i] = col_vec[i], col_vec[1]
#         M[:,i] = col_vec 
#     end
#     return M 
# end



# M_spam = create_mspam(N, 1E-1*rand(2^(N-1)))
# M_spam_kron = create_mspam_kron(N, 1E-1*rand(N))
# #Check for unitary M_spam 
# println("Unitary error non-kron: ", norm(M_spam'*M_spam - I))
# println("Test Unitary kron: ", norm(M_spam_kron'*M_spam_kron - I))
# p_obs = M_spam*p 
# p_obs_kron = M_spam_kron*p
# comparison_plot = plot([p, p_obs, p_obs_kron], labels = ["True p" "Observed p" "Observed p kron"])
#Step 3
#finite-shot sampling 

#Step 4
#synthetic frequencies