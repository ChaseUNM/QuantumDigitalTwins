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

p_obs = create_synthetic_data(p_history)


colors = palette(:auto)[1:2^nqubits]

p_plot = plot(LinRange(t0, T, nsteps + 1), p_history, labels = [L"T:|0\rangle" L"T:|1\rangle" L"|10\rangle" L"|11\rangle"], palette = colors)
plot!(LinRange(t0, T, nsteps + 1), p_obs, labels = [L"Obs: |0\rangle" L"Obs: |1\rangle" L"|10\rangle" L"|11\rangle"], legend=:outertop, legend_columns = 4, xlabel = "t", ylabel = "Population", dpi = 250, alpha = 0.4)
savefig(p_plot, "RabiOscillation.png")
