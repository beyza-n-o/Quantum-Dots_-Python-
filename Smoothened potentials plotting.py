# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:40:56 2024

@author: PC
"""
import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0545718e-34  # J.s
m0 = 9.11e-31  # kg
eV_to_J = 1.60218e-19  # 1 eV = 1.60218e-19 Joules
J_to_eV = 6.25e18
e = 1.60218e-19  # Elementary charge in Coulombs
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
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
V_shell = [eV_to_J * V for V in V_shell_eV]  # Convert potential to Joules
Eg_core = eV_to_J * Eg_core_eV  # Convert to Joules
Eg_shell = [eV_to_J * Eg for Eg in Eg_shell_eV]  # Convert to Joules



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
    print(potentials)
    
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



# m_eff profilini hesapla
V_valence = valence_band_potential(x, V_core, V_shell, L_core, L_shell, s_width=0.8e-10)
V_conduction = conduction_band_potential(x, V_valence, Eg_core, Eg_shell, L_core, L_shell, s_width=0.8e-10)

# m_eff profilini plotla
plt.figure(figsize=(10, 6))

plt.plot(x * 1e9, V_valence / 1.60218e-19, label='Valence Band Potential')
plt.plot(x * 1e9, V_conduction / 1.60218e-19, label='Conduction Band Potential')

plt.xlabel('Position (m)')
plt.ylabel('eV')
plt.ylim(-5, 5)  # y ekseni sınırlarını genişletiyoruz
plt.title('Valence Potential')
plt.legend()
plt.grid(True)
plt.show()