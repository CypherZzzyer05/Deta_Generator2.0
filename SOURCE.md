# Source and independent package identity

Derived from CypherZzzyer05/Deta_Generator commit d35973a988933356b3848cd213a4bf4819505106.
The original repository, installed package, datasets and jobs are preserved.

v2 distribution: deta-generator2, Python module: deta_generator2, version: 2.0.0.
Repository: https://github.com/CypherZzzyer05/Deta_Generator2.0

The required physical change is bottom-to-top SiO2 -> Au -> As2Se3 tetramer.
The pillars contact the metal at z=0. Defaults: SiO2 -450..-100 nm, Au -100..0 nm.
Only the metal-region z mesh is overridden (8 nm for the parallel preset).
Finite substrate thickness and bottom air padding are retained for the controlled comparison.
v2 jobs require explicit metal_position=above_substrate and a separate output directory.
The original optical material data and fitting settings are inherited, not newly certified.
