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

def smooth_transition(x, x1, x2, m1, m2, width=0.8e-10):
    center = (x1 + x2) / 2
    transition = 0.5 * (np.tanh((x - center) / width) + 1)
    return m1 * (1 - transition) + m2 * transition

def valence_band_potential(x, V_core, V_shell, L_core, L_shell, s_width=0.8e-10):
    L = L_core + sum(L_shell)
    V_valence = np.zeros_like(x)
    
    boundaries = [0, L_core]
    
    for i in range(len(L_shell)):
        boundaries.append(L_core + sum(L_shell[:i + 1]))
    
    potentials = [0, V_core] + V_shell
    
    V_valence[(x < -L) | (x > L)] = np.inf
    
    for i in range(2,len(potentials)):
        
        lbfminus= - boundaries[i-1] - s_width
        rbfminus= - boundaries[i-1] + s_width
        
        lbfplus = boundaries[i-1] - s_width
        rbfplus = boundaries[i-1] + s_width
        
        smooth_transition_regionminus = smooth_transition(x, lbfminus, rbfminus, potentials[i],potentials[i-1], s_width)
        smooth_transition_regionplus = smooth_transition(x, lbfplus, rbfplus, potentials[i-1], potentials[i], s_width)
        
        idx_minus_right = (x >= -boundaries[i-1])&(x <= -(boundaries[i-2]+boundaries[i-1])/2)
        idx_plus_left = (x >= (boundaries[i-2]+boundaries[i-1])/2) & (x <= boundaries[i-1])
        
        if i != len(potentials) -1:
            
            idx_minus_left = (x >= -(boundaries[i] +boundaries[i-1])/2) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= (boundaries[i-1]+boundaries[i])/2)
        
        else:
            
            idx_minus_left = (x >= -boundaries[i]) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= boundaries[i])
            
            
        V_valence[idx_minus_right] = smooth_transition_regionminus[idx_minus_right] 
        V_valence[idx_minus_left] = smooth_transition_regionminus[idx_minus_left]

        V_valence[idx_plus_right] = smooth_transition_regionplus[idx_plus_right]
        V_valence[idx_plus_left] = smooth_transition_regionplus[idx_plus_left]
    
    return V_valence

def conduction_band_potential(x, V_valence, Eg_core, Eg_shell, L_core, L_shell, s_width=0.8e-10):
    L = L_core + sum(L_shell)
    V_conduction = np.zeros_like(x)*Eg_core
    
    boundaries= [0, L_core]
    
    for i in range(len(L_shell)):
        boundaries.append(L_core + sum(L_shell[:i + 1]))
    
    
     
    potentials = [0, Eg_core]
    
    for i in range(len(Eg_shell)):
    
        potentials.append(V_valence[i+1] + Eg_shell[i])
    
    V_conduction[(x < -L) | (x > L)] = np.inf
    
    V_conduction[(x >= -L_core)&(x <= L_core)] = Eg_core
    
    
    for i in range(2,len(potentials)):
        
        lbfminus= - boundaries[i-1] - s_width
        rbfminus= - boundaries[i-1] + s_width
        
        lbfplus = boundaries[i-1] - s_width
        rbfplus = boundaries[i-1] + s_width
        
        smooth_transition_regionminus = smooth_transition(x, lbfminus, rbfminus, potentials[i],potentials[i-1], s_width)
        smooth_transition_regionplus = smooth_transition(x, lbfplus, rbfplus, potentials[i-1], potentials[i], s_width)
        
        idx_minus_right = (x >= -boundaries[i-1])&(x <= -(boundaries[i-2]+boundaries[i-1])/2)
        idx_plus_left = (x >= (boundaries[i-2]+boundaries[i-1])/2) & (x <= boundaries[i-1])
        
        if i != len(potentials) -1:
            
            idx_minus_left = (x >= -(boundaries[i] +boundaries[i-1])/2) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= (boundaries[i-1]+boundaries[i])/2)
        

    
        else:
            
            idx_minus_left = (x >= -boundaries[i]) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= boundaries[i])
            
            
        V_conduction[idx_minus_right] = smooth_transition_regionminus[idx_minus_right] 
        V_conduction[idx_minus_left] = smooth_transition_regionminus[idx_minus_left]

        V_conduction[idx_plus_right] = smooth_transition_regionplus[idx_plus_right]
        V_conduction[idx_plus_left] = smooth_transition_regionplus[idx_plus_left]
    
    return V_conduction

def effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell, s_width=0.8e-10):
    m_eff = np.ones_like(x) * m_eff_core
    boundaries = [0, L_core]
    
    for i in range(len(L_shell)):
        boundaries.append(L_core + sum(L_shell[:i + 1]))
    
    masses = [0, m_eff_core] + m_eff_shell
    
    for i in range(2, len(masses)):
        lbfminus = -boundaries[i-1] - s_width
        rbfminus = -boundaries[i-1] + s_width
        
        lbfplus = boundaries[i-1] - s_width
        rbfplus = boundaries[i-1] + s_width
        
        smooth_transition_regionminus = smooth_transition(x, lbfminus, rbfminus, masses[i], masses[i-1], s_width)
        smooth_transition_regionplus = smooth_transition(x, lbfplus, rbfplus, masses[i-1], masses[i], s_width)
        
        idx_minus_right = (x >= -boundaries[i-1]) & (x <= -(boundaries[i-2] + boundaries[i-1])/2)
        idx_plus_left = (x >= (boundaries[i-2] + boundaries[i-1])/2) & (x <= boundaries[i-1])
        
        if i != len(masses) - 1:
            idx_minus_left = (x >= -(boundaries[i] + boundaries[i-1])/2) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= (boundaries[i-1] + boundaries[i])/2)
        else:
            idx_minus_left = (x >= -boundaries[i]) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= boundaries[i])
            
        m_eff[idx_minus_right] = smooth_transition_regionminus[idx_minus_right]
        m_eff[idx_minus_left] = smooth_transition_regionminus[idx_minus_left]

        m_eff[idx_plus_right] = smooth_transition_regionplus[idx_plus_right]
        m_eff[idx_plus_left] = smooth_transition_regionplus[idx_plus_left]
    
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

# Define parameters
V_shell_eV = [-0.5]  # Potential in eV for each shell

Eg_core_eV = 1.74  # eV for the core
Eg_shell_eV = [3.7]  # eV for each shell

L_core = 2e-9  # meters, 2 nm core length
L_shell = [3e-9]  # meters, 3 nm and 2 nm shell lengths respectively
L = L_core + sum(L_shell)
x = np.linspace(-L, L, 1000)

m_eff_core = 0.11 * m0  # Effective mass for core
m_eff_shell = [0.25 * m0]  # Effective masses for each shell
valence_m_eff_core = 0.44 * m0  # Valence band effective mass for core
valence_m_eff_shell = [0.6 * m0]  # Valence band effective masses for each shell

V_core = 0
V_shell = [eV_to_J * V for V in V_shell_eV]  # Convert potential to Joules
Eg_core = eV_to_J * Eg_core_eV  # Convert to Joules
Eg_shell = [eV_to_J * Eg for Eg in Eg_shell_eV]  # Convert to Joules

# Calculate potentials and effective masses
V_valence = valence_band_potential(x, V_core, V_shell, L_core, L_shell)
V_conduction = conduction_band_potential(x, V_valence, Eg_core, Eg_shell, L_core, L_shell)
m_eff_conduction = effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell)
m_eff_valence = effective_mass_profile(x, valence_m_eff_core, valence_m_eff_shell, L_core, L_shell)

# Plot wave functions and energies
plot_wave_functions_and_energies(V_conduction, V_valence, m_eff_conduction, m_eff_valence, L, "Quantum Dot with Multiple Shells")
