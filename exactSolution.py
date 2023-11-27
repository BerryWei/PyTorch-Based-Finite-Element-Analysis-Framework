import numpy as np
import matplotlib.pyplot as plt

# Given constants for the copper beam
P_0 = 10  # Harmonic force amplitude in Newtons
L = 0.15  # Length of the beam in meters
E = 130 * 1e9  # Young's modulus in Pascals (converted from GPa)
rho = 8.96 * 1000  # Density in kg/m^3 (converted from g/cm^3)
height = 0.03  # Width and height of the cross-section in meters
width = 0.03
A = width * height  # Cross-sectional area in m^2
a = L / 2  # Force applied at the midpoint

# Compute the first natural frequency omega_1
I = (width * height ** 3) / 12  # Moment of inertia of the beam's cross-section
omega_1 = np.sqrt(E * I / (rho * A)) * (np.pi / L) ** 2

# Given omega for the harmonic force
omega = 0.1 * omega_1

print(f'omega= {omega}')
# Time and space discretization parameters
x_points = 100
t_points = 100
x = np.linspace(0, L, x_points)
t = np.linspace(0, 1.0e-4*100, t_points)  # Time range of 1 second for illustration

# Calculating the exact solution
def calculate_u(x, t, L, a, omega, P_0, rho, A):
    u = np.zeros((len(t), len(x)))
    for n in range(1, 100):  # Summing over n up to 99
        omega_n = np.sqrt(E * I / (rho * A)) * (n * np.pi / L) ** 2
        term = (2 * P_0) / (rho * A * L) * 1 / (omega_n ** 2 - omega ** 2) * np.sin(n * np.pi * a / L) * np.sin(n * np.pi * x / L)
        for i, ti in enumerate(t):
            u[i, :] += term * np.sin(omega * ti)
    return u

# Calculate the solution
u = calculate_u(x, t, L, a, omega, P_0, rho, A)

# Plotting the solution at different times
plt.figure(figsize=(12, 8))
for i in range(0, len(t), len(t)//10):
    plt.plot(x, u[i, :], label=f't = {t[i]:.5f}s')
plt.title('Displacement of the Copper Beam Over Time')
plt.xlabel('Position along the beam (m)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid(True)
plt.show()
