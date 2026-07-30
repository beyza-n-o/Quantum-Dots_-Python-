import numpy as np
import matplotlib.pyplot as plt
m0 = 9.11e-31  # kg

def smooth_transition(x, x1, x2, m1, m2, width=0.8e-10):
    center = (x1 + x2) / 2
    transition = 0.5 * (np.tanh((x - center) / width) + 1)
    return m1 * (1 - transition) + m2 * transition

def effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell, s_width=0.8e-10):
    #L = L_core + sum(L_shell)
    m_eff = np.ones_like(x) *m_eff_core
    boundaries = [0, L_core]
    #s_width=2e-10
    for i in range(len(L_shell)):
        
        boundaries.append(L_core + sum(L_shell[:i + 1]))
    
    print(boundaries)
    
    masses = [0, m_eff_core] + m_eff_shell
    
    
    for i in range(2,len(masses)):
        
        lbfminus= - boundaries[i-1] - s_width
        rbfminus= - boundaries[i-1] + s_width
        
        lbfplus = boundaries[i-1] - s_width
        rbfplus = boundaries[i-1] + s_width
        
        smooth_transition_regionminus = smooth_transition(x, lbfminus, rbfminus, masses[i], masses[i-1], s_width)
        smooth_transition_regionplus = smooth_transition(x, lbfplus, rbfplus, masses[i-1], masses[i], s_width)
        
        idx_minus_right = (x >= -boundaries[i-1])&(x <= -(boundaries[i-2]+boundaries[i-1])/2)
        idx_plus_left = (x >= (boundaries[i-2]+boundaries[i-1])/2) & (x <= boundaries[i-1])
        
        if i != len(masses) -1:
            
            idx_minus_left = (x >= -(boundaries[i] +boundaries[i-1])/2) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= (boundaries[i-1]+boundaries[i])/2)
        
        else:
            
            idx_minus_left = (x >= -boundaries[i]) & (x <= -boundaries[i-1])
            idx_plus_right = (x >= boundaries[i-1]) & (x <= boundaries[i])
            
            
            
            
        m_eff[idx_minus_right] = smooth_transition_regionminus[idx_minus_right] 
        m_eff[idx_minus_left] = smooth_transition_regionminus[idx_minus_left]

        m_eff[idx_plus_right] = smooth_transition_regionplus[idx_plus_right]
        m_eff[idx_plus_left] = smooth_transition_regionplus[idx_plus_left]
    
    return m_eff

# Verilen değerler
L_core = 2e-9  # meters, 2 nm core length
L_shell = [3e-9]  # meters, 3 nm and 2 nm shell lengths respectively
L = L_core + sum(L_shell)
x = np.linspace(-L, L, 1000)

m_eff_core = 0.11 * m0  # Effective mass for core
m_eff_shell = [0.25 * m0]  # Effective masses for each shell
valence_m_eff_core = 0.44 * m0  # Valence band effective mass for core
valence_m_eff_shell = [0.6 * m0]  # Valence band effective masses for each shell

# m_eff profilini hesapla
m_eff = effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell, s_width=0.8e-10)
valence_m_eff = effective_mass_profile(x, valence_m_eff_core, valence_m_eff_shell, L_core, L_shell, s_width=0.8e-10)

# m_eff profilini plotla
plt.figure(figsize=(10, 6))
plt.plot(x, m_eff / m0, label='Conduction Band Effective Mass')
plt.plot(x, valence_m_eff / m0, label='Valence Band Effective Mass')
plt.xlabel('Position (m)')
plt.ylabel('Effective Mass (m/m0)')
plt.ylim(0, 1.5)  # y ekseni sınırlarını genişletiyoruz
plt.title('Effective Mass Profile')
plt.legend()
plt.grid(True)
plt.show()
"""
def effective_mass_profile(x, m_eff_core, m_eff_shell, L_core, L_shell, smooth_width=1e-10):
    #L = L_core + sum(L_shell)
    m_eff = np.ones_like(x) * m_eff_core
    boundaries = [0, L_core]
    
    for i in range(len(L_shell)):
        boundaries.append(L_core + sum(L_shell[:i + 1]))
    
    masses = [0, m_eff_core] + m_eff_shell
    
    for i in range(1,len(masses)):
        m_eff[(x > boundaries[i-1]) & (x <= boundaries[i])] = masses[i]
        m_eff[(x < -boundaries[i-1]) & (x >= -boundaries[i])] = masses[i]
    return m_eff
"""