import jax
import jax.numpy as jnp

def Cartesian(self, bat_frame):
   
    origin = bat_frame[:3]
    (phi, theta, omega) = bat_frame[3:6]
    (r01, r12, a012) = bat_frame[6:9]
    n_torsions = (self._ag.n_atoms - 3)
    bonds = bat_frame[9:n_torsions + 9]
    angles = bat_frame[n_torsions + 9:2 * n_torsions + 9]
    torsions = jax.ops.index_update(bat_frame[2 * n_torsions + 9:], self._unique_primary_torsion_indices, 0.0)
    torsions += torsions[self._primary_torsion_indices]
    torsions = ((torsions + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    p0 = jnp.array([0., 0., 0.])
    p1 = jnp.array([0., 0., r01])
    p2 = jnp.array([r12 * jnp.sin(a012), 0., r01 - r12 * jnp.cos(a012)])

    co = jnp.cos(omega)
    so = jnp.sin(omega)
    Romega = jnp.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])
    p2 = jnp.dot(Romega, p2)

    cp = jnp.cos(phi)
    sp = jnp.sin(phi)
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    Re = jnp.array([[cp * ct, -sp, cp * st], [ct * sp, cp, sp * st], [-st, 0, ct]])
    p1 = jnp.dot(Re, p1)
    p2 = jnp.dot(Re, p2)

    p0 += origin
    p1 += origin
    p2 += origin

    XYZ = jnp.zeros((self._ag.n_atoms, 3))
    XYZ = jax.ops.index_update(XYZ, self._root_XYZ_inds[0], p0)
    XYZ = jax.ops.index_update(XYZ, self._root_XYZ_inds[1], p1)
    XYZ = jax.ops.index_update(XYZ, self._root_XYZ_inds[2], p2)

    for ((a0, a1, a2, a3), r01, angle, torsion) in zip(self._torsion_XYZ_inds, bonds, angles, torsions):
        p1 = XYZ[a1]
        p3 = XYZ[a3]
        p2 = XYZ[a2]

        sn_ang = jnp.sin(angle)
        cs_ang = jnp.cos(angle)
        sn_tor = jnp.sin(torsion)
        cs_tor = jnp.cos(torsion)

        v21 = p1 - p2
        v21 /= jnp.sqrt(jnp.einsum('i,i->', v21, v21))
        v32 = p2 - p3
        v32 /= jnp.sqrt(jnp.einsum('i,i->', v32, v32))

        vp = jnp.cross(v32, v21)
        cs = jnp.einsum('i,i->', v21, v32)

        sn = jnp.maximum(jnp.sqrt(1.0 - cs * cs), 0.0000000001)
        vp = vp / sn
        vu = jnp.cross(vp, v21)

        XYZ = jax.ops.index_update(XYZ, a0, p1 + r01 * (vu * sn_ang * cs_tor + vp * sn_ang * sn_tor - v21 * cs_ang))

    return XYZ
