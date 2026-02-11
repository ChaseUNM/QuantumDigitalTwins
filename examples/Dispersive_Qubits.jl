using LinearAlgebra, Plots, LaTeXStrings
using TensorMethods
using QuantumDigitalTwins

plot_pulse = true

#Setup hamiltonian
nqubits = 2
nlevels = fill(2, nqubits)
init_vec = zeros(ComplexF64, 2^nqubits)
init_vec[1] = 1.0 + 0.0*im
init_vec = init_vec/norm(init_vec)
freq01_all = [4.5, 5, 5.5, 6]
freq01_all = freq01_all[1:nqubits]*2pi

favg = sum(freq01_all)/length(freq01_all)
t0 = 0.0
T = 10.0

Jkl_coupling_strength = 5e-3*2pi # 5e-3  	# [GHz] Coupling strength for qubit CHAIN topology
Jkl = zeros(nqubits, nqubits)
for i in 1:nqubits
	for j in i+1:nqubits
		if j == i+1
			Jkl[i,j] = Jkl_coupling_strength
        end
    end
end 

self_kerr = zeros(nqubits)*2pi 
zz = zeros(nqubits, nqubits)*2pi
rotfreq = favg*ones(nqubits)

H_mat = H_drift_mat(nqubits, nlevels, freq01_all, rotfreq, self_kerr, zz, Jkl)
#Set up bcparams with initial pcof
dT = 0.01
steps = Int64((T - t0)/dT)
splines = 6

carrier_freq = [[freq01_all[iq] - favg] for iq in 1:nqubits] 
nparams = 2*splines*(sum(length, carrier_freq))
pcof = ones(nparams)*0.01
bc_params = bcparams(T, splines, carrier_freq, pcof)

if plot_pulse == true
    pulse = 1
    time_range = LinRange(0, T, steps)
    p_eval = zeros(length(time_range))
    q_eval = zeros(length(time_range))
    for j = 1:steps 
        p_eval[j] = bcarrier2(time_range[j], bc_params, 2*(pulse - 1))*(500/pi)
        q_eval[j] = bcarrier2(time_range[j], bc_params, 2*(pulse - 1) + 1)*(500/pi)
    end
    pulse_plot = plot(time_range, [p_eval, q_eval], labels = ["p(t)" "q(t)"], xlabel = "time(ns)", ylabel = "MHz")
end

#Evolve system just using exponential solver
ans_vec, sol_history, _, _ = td_exp_solver(H_mat, nlevels, bc_params, init_vec, nqubits, 0, 1, steps)
p_history = abs2.(sol_history)



p_obs = create_synthetic_data(p_history)

colors = palette(:auto)[1:2^nqubits]

p_plot = plot(LinRange(t0, T, steps + 1), p_history, labels = [L"T:|00\rangle" L"T:|01\rangle" L"T: |10\rangle" L"T: |11\rangle"], palette = colors)
plot!(LinRange(t0, T, steps + 1), p_obs, labels = [L"Obs: |00\rangle" L"Obs: |01\rangle" L"Obs: |10\rangle" L"Obs: |11\rangle"], legend=:outertop, legend_columns = 4, xlabel = "t", ylabel = "Population", dpi = 250, alpha = 0.4)
savefig(p_plot, "Dispersive_Qubits.png")



