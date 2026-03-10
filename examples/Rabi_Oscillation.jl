using Revise
using LinearAlgebra, Plots, LaTeXStrings
using QuantumDigitalTwins
using QuantumGateDesign


nqubits = 1
H_drift = zeros(2,2)
H_control_real = [0.0 1;
                  1   0]
H_control_imag = [0.0 1;
                  -1  0]
real_control_ops = [H_control_real]
imag_control_ops = [H_control_imag]

#psi0 = reshape([1.0, 0], :, 1)
psi0 = [1.0, 0]
t0 = 0.0
T = 50.0 # time in nanoseconds
nsteps = 100
sym_ops = [H_control_real]
asym_ops = [H_control_imag]
prob = SchrodingerProb(H_drift, real_control_ops, imag_control_ops, psi0, T, nsteps)

N_GRAPE_amplitudes = 1
control = GRAPEControl(N_GRAPE_amplitudes, T)
pcof = [pi/(2*T), 0]

history = eval_forward(prob, control, pcof)
pcof_init = zeros(2)
target = [0.0, 1]

pl = plot_controls(control, pcof)

p_history = abs2.(history)'


samples = 100000
epsilon_vec = 0.001*rand(2^nqubits)

p_hat, p_obs  = create_synthetic_data(p_history, samples, epsilon_vec)

pred_infidelity = round(1 - abs2(history[2,end]), digits = 10)
meas_infidelity = round(1 - p_hat[end,2], digits = 10)

colors = palette(:auto)[1:2^nqubits]
time_range = LinRange(t0, T, nsteps + 1)
pi_plot = plot(time_range, p_history, labels = [L"True:|0\rangle" L"True:|1\rangle" L"|10\rangle" L"|11\rangle"], palette = colors)
plot!(time_range, p_hat, labels = [L"Obs: |0\rangle" L"Obs: |1\rangle" L"|10\rangle" L"|11\rangle"], legend=:outertop, legend_columns = 4, xlabel = "t", ylabel = "Population", dpi = 250, alpha = 0.4)
annotate!(pi_plot, time_range[end] - 2.0, p_history[end,:2] - 0.1, text("I(Pred) $(pred_infidelity)", 5))
annotate!(pi_plot, time_range[end] - 2.0, p_history[end,:2] - 0.2, text("I(Meas) $(meas_infidelity)", 5))
# savefig(p_plot, "RabiOscillation.png")


gate = [0.0, 1.0]
#Infidelity calculations
pred_infidel = 1 - fidelity_general(p_history[end,:], gate)
obs_infidel = 1 - fidelity_general(p_obs[end,:], gate)
#Previous infidelity


# N_GRAPE_amplitudes = 1
# control = GRAPEControl(N_GRAPE_amplitudes, T)
# pcof = [pi/(4*T), 0]
# history = eval_forward(prob, control, pcof)
# pcof_init = zeros(2)
# target = [1/sqrt(2), 1/sqrt(2)]
# pl = plot_controls(control, pcof)
# p_history = abs2.(history)'
control_plot = plot(LinRange(t0, T, nsteps + 1), fill(pcof[1], nsteps + 1), label = "Real Pulse", xlabel = "t", dpi = 250)
plot!(LinRange(t0, T, nsteps + 1), fill(pcof[2], nsteps + 1), label = "Imaginary Pulse")

# samples = 5000
# epsilon_vec2 = 0.01*rand(2^nqubits)

# p_hat, p_obs  = create_synthetic_data(p_history, samples, epsilon_vec2)

# pi_half_plot = plot(LinRange(t0, T, nsteps + 1), p_history, labels = [L"True:|0\rangle" L"True:|1\rangle" L"|10\rangle" L"|11\rangle"], palette = colors)
# plot!(LinRange(t0, T, nsteps + 1), p_hat, labels = [L"Obs: |0\rangle" L"Obs: |1\rangle" L"|10\rangle" L"|11\rangle"], legend=:outertop, legend_columns = 4, xlabel = "t", ylabel = "Population", dpi = 250, alpha = 0.4)

