import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import scipy.integrate

hbar = 1.0545718e-34  # J.s
m0 = 9.11e-31  # kg
eV_to_J = 1.60218e-19  # 1 eV = 1.60218e-19 Joules
J_to_eV = 6.25e18
e = 1.60218e-19  # Elementary charge in Coulombs
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m

def smooth_transition(x, x1, x2, m1, m2, width=1e-10):
    center = (x1 + x2) / 2
    transition = 0.5 * (np.tanh((x - center) / width) + 1)
    return m1 * (1 - transition) + m2 * transition

def valence_band_potential(x, V_core, V_shell, L_core, L_shell):
    V_valence = np.zeros_like(x)
    
    V_valence[(x >= -L_core) & (x <= L_core)] = V_core
    V_valence[(x >= -L_core - L_shell) & (x <= -L_core)] = V_shell
    V_valence[(x >= L_core) & (x <= L_core + L_shell)] = V_shell
    
    V_valence[(x == -L_core - L_shell) | (x == L_core + L_shell)] = -50*eV_to_J
    return V_valence

def conduction_band_potential(V_valence, Eg_core, Eg_shell, L_core, L_shell):
    V_conduction = np.zeros_like(V_valence)
    V_conduction[(x >= -L_core) & (x <= L_core)] = V_valence[(x >= -L_core) & (x <= L_core)] + Eg_core
    V_conduction[(x >= -L_core - L_shell) & (x <= -L_core)] = V_valence[(x >= -L_core - L_shell) & (x <= -L_core)] + Eg_shell
    V_conduction[(x >= L_core) & (x <= L_core + L_shell)] = V_valence[(x >= L_core) & (x <= L_core + L_shell)] + Eg_shell
    
    V_conduction[(x == -L_core - L_shell) | (x == L_core + L_shell)] = 50*eV_to_J
    
    return V_conduction

def effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell, smooth_width=0.2):
    m_eff = np.ones_like(x) * m_eff_core
    L = L_core + L_shell
    smooth_transition_region1 = smooth_transition(x, (-L_core-smooth_width), (-L_core+smooth_width), m_eff_shell, m_eff_core, smooth_width)
    smooth_transition_region2 = smooth_transition(x, L_core-smooth_width, L_core + smooth_width, m_eff_core, m_eff_shell, smooth_width)
    

    m_eff[(x >= -L_core - L_shell) & (x <= -L_core)] = smooth_transition_region1[(x >= -L_core - L_shell) & (x <= -L_core)]
    
    m_eff[(x >= L_core) & (x <= L_core + L_shell)] = smooth_transition_region2[(x >= L_core) & (x <= L_core + L_shell)]
    
    m_eff[(x >= -L_core) & (x <= 0)] = smooth_transition_region1[(x >= -L_core) & (x <= 0)]
    
    m_eff[(x >= 0) & (x <= L_core)] = smooth_transition_region2[(x >= 0) & (x <= L_core)]
    
    return m_eff

def plot_wave_functions_and_energies(V_conduction, V_valence, m_eff_conduction, m_eff_valence, L, title):
    x = np.linspace(-L, L, 1000)
    dx = x[1] - x[0]

    m_eff_conduction = effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell)
    m_eff_valence = effective_mass_profile(x, valence_m_eff_core, valence_m_eff_shell, L_core, L_shell)

    # Conduction band 
    diagonal_conduction = hbar**2 / (m_eff_conduction * dx**2) + V_conduction
    off_diagonal_conduction = -hbar**2 / (2*m_eff_conduction[:-1] * dx**2) * np.ones_like(x[:-1])
    eigenvalues_conduction, eigenvectors_conduction = eigh_tridiagonal(diagonal_conduction, off_diagonal_conduction)

    eigenvalues_conduction, eigenvectors_conduction = eigenvalues_conduction* J_to_eV, eigenvectors_conduction* J_to_eV

    # For valence band 
    diagonal_valence = hbar**2 / (m_eff_valence * dx**2) + (-V_valence)  # Here it's minus because they are holes
    off_diagonal_valence = -hbar**2 / (2*m_eff_valence[:-1] * dx**2) * np.ones_like(x[:-1])
    eigenvalues_valence, eigenvectors_valence = eigh_tridiagonal(diagonal_valence, off_diagonal_valence)
    eigenvalues_valence, eigenvectors_valence = eigenvalues_valence*J_to_eV, eigenvectors_valence*J_to_eV


####################################
    V_coulomb = 0
    for i in range(len(x)):
        for j in range(len(x)):
            r_i = x[i]
            r_j = x[j]
            psi_e = (eigenvectors_conduction[:, 0][i]*eV_to_J)**2
            psi_h = (eigenvectors_valence[:, 0][j]*eV_to_J)**2
        if r_i != r_j:
            V_coulomb += psi_e * psi_h * e**2 / (4 * np.pi * epsilon_0 * np.abs(r_i - r_j))
    V_coulomb *= -J_to_eV
    print(f"Coulomb Interaction Potential: {V_coulomb:.20f} eV")


####################################

    plt.figure(figsize=(10, 8))
    plt.plot(x, V_conduction*J_to_eV, 'k-', linewidth=2, label='Conduction Band Potential')
    plt.plot(x, V_valence*J_to_eV, 'k-', label='Valence Band Potential')

    scaling_factor = 0.00001 * np.max(V_conduction)*J_to_eV  # Scaling faktörü korundu

#Conduction
    for n in range(2):  # for 2 n
        wave_function_not_normalizedC = eigenvectors_conduction[:, n] 
        normalization_factorC = np.sqrt(scipy.integrate.simpson(abs(wave_function_not_normalizedC) ** 2, x=x))
        wave_functionC = wave_function_not_normalizedC / normalization_factorC
        offsetC = eigenvalues_conduction[n] + V_coulomb
        plt.plot(x, (offsetC + wave_functionC * scaling_factor), label=f'Conduction Wave Function {n+1}')
        plt.axhline(y=eigenvalues_conduction[n], color='r', linestyle='--', label=f'Conduction Energy Level {n+1}')

#Valence
    for n in range(2):
        wave_function_not_normalizedV = eigenvectors_valence[:, n] 
        
        normalization_factorV = np.sqrt(scipy.integrate.simpson(abs(wave_function_not_normalizedV) ** 2, x=x))
        wave_functionV = wave_function_not_normalizedV / normalization_factorV
        offsetV = -eigenvalues_valence[n] - V_coulomb
        plt.plot(x, (offsetV - wave_functionV * scaling_factor), label=f'Valence Wave Function {n+1}')
        plt.axhline(y=-eigenvalues_valence[n], color='b', linestyle='--', label=f'Valence Energy Level {n+1}')

    for n in range(2):
        E_difference = ((eigenvalues_conduction[n] + V_coulomb) - (eigenvalues_valence[n]-V_coulomb))
        print(f"E{n+1} Conduction - E{n+1} Valence: {E_difference} eV")
        
    E_difference_conduction = (eigenvalues_conduction[1] - eigenvalues_conduction[0])
    E_difference_valance = (eigenvalues_valence[1] - eigenvalues_valence[0])
    print(f"E1 Conduction - E0 Conduction: {E_difference_conduction} eV")
    print(f"E1 Valance - E0 Valance: {E_difference_valance} eV")
    
    
    plt.xlabel('Position')
    plt.ylabel('Energy / Wave Function')
    plt.title(title)
    plt.ylim(-20, 20)  # ylim genişletildi
    plt.xlim(-1*L, 1*L)
    
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    #return eigenvalues_valence, eigenvectors_valence

V_shell_eV = -0.5  # Potential in eV
Eg_core_eV = 1.74  # eV
Eg_shell_eV = 3.7  # eV

L_core = 2e-9  # meters
L_shell = 3e-9  # meters
L = L_core + L_shell
x = np.linspace(-L, L, 1000)

m_eff_core = 0.11 * m0
m_eff_shell = 0.25 * m0
valence_m_eff_core = 0.44 * m0
valence_m_eff_shell = 0.6 * m0

V_core = 0
V_shell = 1.60218e-19 * V_shell_eV  # Potential in J
Eg_core = 1.60218e-19 * Eg_core_eV  # J
Eg_shell = 1.60218e-19 * Eg_shell_eV  # J

V_valence = valence_band_potential(x, V_core, V_shell, L_core, L_shell)
V_conduction = conduction_band_potential(V_valence, Eg_core, Eg_shell, L_core, L_shell)
m_eff_conduction = effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell)
m_eff_valence = effective_mass_profile(x, valence_m_eff_core, valence_m_eff_shell, L_core, L_shell)

output = plot_wave_functions_and_energies(V_conduction, V_valence, m_eff_conduction, m_eff_valence, L, "CdSe ZnS")
