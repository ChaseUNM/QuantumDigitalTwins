using LinearAlgebra
using TensorMethods 

function td_exp_solver(H, nlevels, bc_params, init_vec, N, t0, T, steps)
    h = (T - t0)/steps 
    magnet_history = zeros(steps + 1, N)
    energy_history = zeros(steps + 1)
    for j = 1:N
        m_mat = s_op_reverse([1 0; 0 -1], j, N)
        magnet_history[1,j] = real(init_vec'*m_mat*init_vec)
    end
    t0 = 0.0
    energy_history[1] = real(init_vec'*H*init_vec)
    sol_history = zeros(ComplexF64, steps + 1, 2^N)
    sol_history[1,:] = init_vec
    @showprogress 1 "Exponential solver" for i = 1:steps 
        init_vec = exp(-im*H*h)*init_vec
        sol_history[i + 1,:] = init_vec
        for j = 1:N 
            m_mat = s_op_reverse([1 0; 0 -1], j, N)
            magnet_history[i + 1,j] = real(init_vec'*m_mat*init_vec)
        end
        energy_history[i + 1] = real(init_vec'*H*init_vec)
        t0 += h
        updateH_mat!(H, nlevels, N, bc_params, t0) 
    end
    return init_vec, sol_history, magnet_history, energy_history 
end