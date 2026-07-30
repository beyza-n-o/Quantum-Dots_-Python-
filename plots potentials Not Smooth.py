import numpy as np
import matplotlib.pyplot as plt
m0 = 9.11e-31  # kg
V_shell_eV = [-0.5, -0.3]  # Potential in eV for each shell
Eg_core_eV = 1.74  # eV for the core
Eg_shell_eV = [3.7, 2.5]  # eV for each shell

L_core = 2e-9  # meters, 2 nm core length
L_shell = [3e-9, 2e-9]  # meters, 3 nm and 2 nm shell lengths respectively
L = L_core + sum(L_shell)
x = np.linspace(-L, L, 1000)

m_eff_core = 0.11 * m0  # Effective mass for core
m_eff_shell = [0.25 * m0, 0.20 * m0]  # Effective masses for each shell
valence_m_eff_core = 0.44 * m0  # Valence band effective mass for core
valence_m_eff_shell = [0.6 * m0, 0.5 * m0]  # Valence band effective masses for each shell

V_core = 0
V_shell = [1.60218e-19 * V for V in V_shell_eV]  # Convert potential to Joules
Eg_core = 1.60218e-19 * Eg_core_eV  # Convert to Joules
Eg_shell = [1.60218e-19 * Eg for Eg in Eg_shell_eV]  # Convert to Joules

# Function definitions
def valence_band_potential(x, V_core, V_shell, L_core, L_shell):
    L = L_core + sum(L_shell)
    V_valence = np.zeros_like(x)
    
    boundaries = [0, L_core]
    
    for i in range(len(L_shell)):
        boundaries.append(L_core + sum(L_shell[:i + 1]))
    
    potentials = [0, V_core] + V_shell
    
    
    
    for i in range(1, len(potentials)):
        V_valence[(x > boundaries[i-1]) & (x <= boundaries[i])] = potentials[i]
        V_valence[(x < -boundaries[i-1]) & (x >= -boundaries[i])] = potentials[i]
    V_valence[(x == -L) | (x == L)] = -20/1.60218e-19 #in J
    return V_valence

def conduction_band_potential(x, V_valence, Eg_core, Eg_shell, L_core, L_shell):
    L = L_core + sum(L_shell)
    V_conduction = np.zeros_like(V_valence)
    boundaries = [0, L_core]
    
    for i in range(len(L_shell)):
        boundaries.append(L_core + sum(L_shell[:i + 1]))
    
    potentials = [0, Eg_core] + [Eg_shell[i] for i in range(len(Eg_shell))]
    
    
    for i in range(1, len(potentials)):
        idx_minus = (x < -boundaries[i-1]) & (x >= -boundaries[i])
        idx_plus = (x > boundaries[i-1]) & (x <= boundaries[i])
        
      
        V_conduction[idx_plus]  = V_valence[(x > boundaries[i-1]) & (x <= boundaries[i])] + potentials[i]
        V_conduction[idx_minus] = V_valence[(x < -boundaries[i-1]) & (x >= -boundaries[i])] + potentials[i]
    V_conduction[(x == -L) | (x == L)] = 20/1.60218e-19 #in J

    return V_conduction

# Calculate potentials
V_valence = valence_band_potential(x, V_core, V_shell, L_core, L_shell)
V_conduction = conduction_band_potential(x, V_valence, Eg_core, Eg_shell, L_core, L_shell)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x * 1e9, V_valence / 1.60218e-19, label='Valence Band Potential')
plt.plot(x * 1e9, V_conduction / 1.60218e-19, label='Conduction Band Potential')
plt.xlabel('Position (nm)')
plt.ylabel('Potential Energy (eV)')
plt.ylim(-20, 20)
plt.title('Band Potential Profiles')
plt.legend()
plt.grid(True)
plt.show()
