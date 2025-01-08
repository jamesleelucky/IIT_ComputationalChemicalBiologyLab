import mdtraj as md
import jax.numpy as jnp
traj = md.load('decaalanine_1us_split3.dcd', top='ala_deca_peptide.prmtop')
cart_coords = jnp.array(traj.xyz)