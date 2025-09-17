import numpy as np
import matplotlib.pyplot as plt

# This solves 1D time-independent Schrodinger's equation at a given energy
# Assume incoming wave from right to left
# Potential profile defined as V


# Parameters
ħ = 1.0
m = 1.0
E = 1                 # particle energy
x_min, x_max = -20, 20   # simulation domain
N = 20000
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Define smooth potential (example: Gaussian barrier)
V0 = 2.0
sigma = 0.5
V = V0 * np.exp(-x**2 / (2 * sigma**2))

# Wave numbers far left/right
kL = np.sqrt(2*m*(E - V[0]))/ħ
kR = np.sqrt(2*m*(E - V[-1]))/ħ

# Numerov algorithm to integrate Schr eqn from left to right
def numerov(E, V, x, dx):
    k2 = 2*m*(E - V)/ħ**2
    ψ = np.zeros_like(x, dtype=complex)
    # Initial condition: only transmitted wave at left
    ψ[0] = np.exp(-1j*kL*x[0])
    ψ[1] = np.exp(-1j*kL*x[1])
    # Numerov step
    for n in range(1, len(x)-1):
        ψ[n+1] = ( (2*(1 - 5*dx**2*k2[n]/12)*ψ[n])
                  - (1 + dx**2*k2[n-1]/12)*ψ[n-1] ) / (1 + dx**2*k2[n+1]/12) # See Fig. 1 of https://github.com/QijingZheng/numerov_schrod1d/blob/main/doc/Numerical_Solutions_to_the_Time-Independent_1-D_Schrodinger_Equation.pdf
    return ψ

ψ = numerov(E, V, x, dx)

# --- Extract scattering coefficients by asymptotic fitting ---
# ψ at left (x<<0) is set to be exp(-i kL x)
# ψ at right (x>>0) is fit to form: A exp(-i kL x) + B exp(i kL x); B is reflected, A is incoming

right_region = (x > 10)
M_right = np.vstack([np.exp(-1j*kL*x[right_region]),
                    np.exp(1j*kL*x[right_region])]).T
coeffs_right, _, _, _ = np.linalg.lstsq(M_right, ψ[right_region], rcond=None)
A, B = coeffs_right

plt.figure()
plt.plot(A*M_right[:,0] + B*M_right[:,1], label="Fitted ψ at right")
plt.plot(ψ[right_region], '--', label="Numerov ψ at right")
plt.legend(); plt.show()


# Reflection and transmission probabilities (flux ratio)
C = 1 # transmitted wave has amplitude of 1
R = abs(B/A)**2
T = (kR/kL) * abs(C/A)**2

print(f"Reflection R = {R:.4f}, Transmission T = {T:.4f}, R+T = {R+T:.4f}")

# Plot wavefunction and potential
plt.figure()
# plt.plot(x, np.real(ψ)/np.max(np.abs(ψ)), label="Re ψ")
plt.plot(x, np.real(ψ), label="Re ψ")
plt.plot(x,np.abs(ψ), '--', label = "abs (ψ)",alpha=0.4)
plt.plot(x, V/V0, 'k--', label="V(x)/V0")
plt.legend(); plt.show()
