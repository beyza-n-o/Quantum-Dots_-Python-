Project Summary

This code calculates the electronic properties of multi-layer (core/shell) quantum dots. It numerically solves the Schrödinger equation to find energy levels and wavefunctions.

What It Does
- Constructs conduction and valence band potentials
- Uses smooth transitions between layers
- Calculates the Coulomb interaction between electrons and holes
- Visualizes results with graphs

Purpose
Used to test how different material and thickness combinations affect the optical and electronic properties of quantum dots.

Example Parameters
```python
V_shell_eV = [-0.5]          # 1 shell, potential -0.5 eV
Eg_shell_eV = [3.7]          # 1 shell, bandgap 3.7 eV
L_shell = [3e-9]             # 1 shell, thickness 3 nm
