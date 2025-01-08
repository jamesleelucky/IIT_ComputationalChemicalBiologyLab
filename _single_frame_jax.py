import jax
import jax.numpy as jnp

def _single_frame(self):
    # Calculate coordinates based on the root atoms
    # The rotation axis is a normalized vector pointing from atom 0 to 1
    # It is described in two degrees of freedom
    # by the polar angle and azimuth
    if self._root.dimensions is None:
        (p0, p1, p2) = self._root.positions
    else:
        (p0, p1, p2) = make_whole(self._root, inplace=False)
    v01 = p1 - p0
    v21 = p1 - p2
    # Internal coordinates
    r01 = jnp.sqrt(jnp.dot(v01, v01))  
    # Distance between first two root atoms
    r12 = jnp.sqrt(jnp.dot(v21, v21))  
    # Distance between second two root atoms
    # Angle between root atoms
    a012 = jnp.arccos(jnp.maximum(-1., jnp.minimum(1., jnp.dot(v01, v21) /
                               (jnp.sqrt(jnp.dot(v01, v01)) * jnp.sqrt(jnp.dot(v21, v21))))))

    # External coordinates
    e = v01 / r01
    phi = jnp.arctan2(e[1], e[0])  # Polar angle
    theta = jnp.arccos(e[2])  # Azimuthal angle
    # Rotation to the z axis
    cp = jnp.cos(phi)
    sp = jnp.sin(phi)
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    Rz = jnp.array([[cp * ct, ct * sp, -st], [-sp, cp, 0],
                    [cp * st, sp * st, ct]])
    pos2 = jnp.dot(Rz, p2 - p1)
    # Angle about the rotation axis
    omega = jnp.arctan2(pos2[1], pos2[0])
    root_based = jnp.concatenate((p0, [phi, theta, omega, r01, r12, a012]))

    # Calculate internal coordinates from the torsion list
    bonds = calc_bonds(self._ag1.positions,self._ag2.positions,box=self._ag1.dimensions)
    angles = calc_angles(self._ag1.positions,self._ag2.positions,self._ag3.positions,box=self._ag1.dimensions)
    torsions = calc_dihedrals(self._ag1.positions,self._ag2.positions,self._ag3.positions,self._ag4.positions,box=self._ag1.dimensions)
    
    # When appropriate, calculate improper torsions
    shift = torsions[self._primary_torsion_indices]
    shift = jax.ops.index_update(shift, self._unique_primary_torsion_indices, 0.)
    torsions -= shift
    
    # Wrap torsions to between -jnp.pi and jnp.pi
    torsions = ((torsions + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    self.results.bat = jax.ops.index_update(
        self.results.bat,
        (self._frame_index, slice(None)),
        jnp.concatenate((root_based, bonds, angles, torsions))
    )
