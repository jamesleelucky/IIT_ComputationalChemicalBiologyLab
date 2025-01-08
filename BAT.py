import logging
import warnings

import numpy as np
import copy

import MDAnalysis as mda
from .base import AnalysisBase

from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
from MDAnalysis.lib.mdamath import make_whole

from ..due import due, Doi

logger = logging.getLogger(__name__)


def _sort_atoms_by_mass(atoms, reverse=False):
    r"""Sorts a list of atoms by name and then by index

    The atom index is used as a tiebreaker so that the ordering is reproducible.

    Parameters
    ----------
    ag_o : list of Atoms
        List to sort
    reverse : bool
        Atoms will be in descending order

    Returns
    -------
    ag_n : list of Atoms
        Sorted list
    """
    return sorted(atoms, key=lambda a: (a.mass, a.index), reverse=reverse)


def _find_torsions(root, atoms):
    """Constructs a list of torsion angles

    Parameters
    ----------
    root : AtomGroup
        First three atoms in the coordinate system
    atoms : AtomGroup
        Atoms that are allowed to be part of the torsion angle

    Returns
    -------
    torsions : list of AtomGroup
        list of AtomGroup objects that define torsion angles
    """
    torsions = []
    selected_atoms = list(root)
    while len(selected_atoms) < len(atoms):
        torsionAdded = False
        for a1 in selected_atoms:
            # Find a0, which is a new atom connected to the selected atom
            a0_list = _sort_atoms_by_mass(a for a in a1.bonded_atoms \
                if (a in atoms) and (a not in selected_atoms))
            for a0 in a0_list:
                # Find a2, which is connected to a1, is not a terminal atom,
                # and has been selected
                a2_list = _sort_atoms_by_mass(a for a in a1.bonded_atoms \
                    if (a!=a0) and len(a.bonded_atoms)>1 and \
                        (a in atoms) and (a in selected_atoms))
                for a2 in a2_list:
                    # Find a3, which is
                    # connected to a2, has been selected, and is not a1
                    a3_list = _sort_atoms_by_mass(a for a in a2.bonded_atoms \
                        if (a!=a1) and \
                            (a in atoms) and (a in selected_atoms))
                    for a3 in a3_list:
                        # Add the torsion to the list of torsions
                        torsions.append(mda.AtomGroup([a0, a1, a2, a3]))
                        # Add the new atom to selected_atoms
                        # which extends the loop
                        selected_atoms.append(a0)
                        torsionAdded = True
                        break  # out of the a3 loop
                    break  # out of the a2 loop
        if torsionAdded is False:
            print('Selected atoms:')
            print([a.index + 1 for a in selected_atoms])
            print('Torsions found:')
            print([list(t.indices + 1) for t in torsions])
            raise ValueError('Additional torsions not found.')
    return torsions


class BAT(AnalysisBase):
    """Calculate BAT coordinates for the specified AtomGroup.

    Bond-Angle-Torsions (BAT) internal coordinates will be computed for
    the group of atoms and all frame in the trajectory belonging to `ag`.

    """
    @due.dcite(Doi("10.1002/jcc.26036"),
               description="Bond-Angle-Torsions Coordinate Transformation",
               path="MDAnalysis.analysis.bat.BAT")
    def __init__(self, ag, initial_atom=None, filename=None, **kwargs):
        r"""Parameters
        ----------
        ag : AtomGroup or Universe
            Group of atoms for which the BAT coordinates are calculated.
            `ag` must have a bonds attribute.
            If unavailable, bonds may be guessed using
            :meth:`AtomGroup.guess_bonds <MDAnalysis.core.groups.AtomGroup.guess_bonds>`.
            `ag` must only include one molecule.
            If a trajectory is associated with the atoms, then the computation
            iterates over the trajectory.
        initial_atom : :class:`Atom <MDAnalysis.core.groups.Atom>`
            The atom whose Cartesian coordinates define the translation
            of the molecule. If not specified, the heaviest terminal atom
            will be selected.
        filename : str
            Name of a numpy binary file containing a saved bat array.
            If filename is not ``None``, the data will be loaded from this file
            instead of being recalculated using the run() method.

        Raises
        ------
        AttributeError
            If `ag` does not contain a bonds attribute
        ValueError
            If `ag` contains more than one molecule

        """
        super(BAT, self).__init__(ag.universe.trajectory, **kwargs)
        self._ag = ag

        # Check that the ag contains bonds
        if not hasattr(self._ag, 'bonds'):
            raise AttributeError('AtomGroup has no attribute bonds')
        if len(self._ag.fragments) > 1:
            raise ValueError('AtomGroup has more than one molecule')

        # Determine the root
        # The initial atom must be a terminal atom
        terminal_atoms = _sort_atoms_by_mass(\
            [a for a in self._ag.atoms if len(a.bonds)==1], reverse=True)
        if (initial_atom is None):
            # Select the heaviest root atoms from the heaviest terminal atom
            initial_atom = terminal_atoms[0]
        elif (not initial_atom in terminal_atoms):
            raise ValueError('Initial atom is not a terminal atom')
        # The next atom in the root is bonded to the initial atom
        # Since the initial atom is a terminal atom, there is only
        # one bonded atom
        second_atom = initial_atom.bonded_atoms[0]
        # The last atom in the root is the heaviest atom
        # bonded to the second atom
        # If there are more than three atoms,
        # then the last atom cannot be a terminal atom.
        if self._ag.n_atoms != 3:
            third_atom = _sort_atoms_by_mass(\
                [a for a in second_atom.bonded_atoms \
                if (a in self._ag) and (a!=initial_atom) \
                and (a not in terminal_atoms)], \
                reverse=True)[0]
        else:
            third_atom = _sort_atoms_by_mass(\
                [a for a in second_atom.bonded_atoms \
                if (a in self._ag) and (a!=initial_atom)], \
                reverse=True)[0]
        self._root = mda.AtomGroup([initial_atom, second_atom, third_atom])

        # Construct a list of torsion angles
        self._torsions = _find_torsions(self._root, self._ag)

        # Get indices of the root and torsion atoms
        # in a Cartesian positions array that matches the AtomGroup
        self._root_XYZ_inds = [(self._ag.indices==a.index).nonzero()[0][0] \
            for a in self._root]
        self._torsion_XYZ_inds = [[(self._ag.indices==a.index).nonzero()[0][0] \
            for a in t] for t in self._torsions]

        # The primary torsion is the first torsion on the list
        # with the same central atoms
        prior_atoms = [sorted([a1, a2]) for (a0, a1, a2, a3) in self._torsions]
        self._primary_torsion_indices = [prior_atoms.index(prior_atoms[n]) \
            for n in range(len(prior_atoms))]
        self._unique_primary_torsion_indices = \
            list(set(self._primary_torsion_indices))

        self._ag1 = mda.AtomGroup([ag[0] for ag in self._torsions])
        self._ag2 = mda.AtomGroup([ag[1] for ag in self._torsions])
        self._ag3 = mda.AtomGroup([ag[2] for ag in self._torsions])
        self._ag4 = mda.AtomGroup([ag[3] for ag in self._torsions])

        if filename is not None:
            self.load(filename)

    def _prepare(self):
        self.results.bat = np.zeros(
                (self.n_frames, 3*self._ag.n_atoms), dtype=np.float64)

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
        r01 = np.sqrt(np.einsum('i,i->',v01,v01))  
        # Distance between first two root atoms
        r12 = np.sqrt(np.einsum('i,i->',v21,v21))  
        # Distance between second two root atoms
        # Angle between root atoms
        a012 = np.arccos(max(-1.,min(1.,np.einsum('i,i->',v01,v21)/\
                              np.sqrt(np.einsum('i,i->',v01,v01)*np.einsum('i,i->',v21,v21)))))
        # External coordinates
        e = v01 / r01
        phi = np.arctan2(e[1], e[0])  # Polar angle
        theta = np.arccos(e[2])  # Azimuthal angle
        # Rotation to the z axis
        cp = np.cos(phi)
        sp = np.sin(phi)
        ct = np.cos(theta)
        st = np.sin(theta)
        Rz = np.array([[cp * ct, ct * sp, -st], [-sp, cp, 0],
                       [cp * st, sp * st, ct]])
        pos2 = Rz.dot(p2 - p1)
        # Angle about the rotation axis
        omega = np.arctan2(pos2[1], pos2[0])
        root_based = np.concatenate((p0, [phi, theta, omega, r01, r12, a012]))

        # Calculate internal coordinates from the torsion list
        bonds = calc_bonds(self._ag1.positions,
                           self._ag2.positions,
                           box=self._ag1.dimensions)
        angles = calc_angles(self._ag1.positions,
                             self._ag2.positions,
                             self._ag3.positions,
                             box=self._ag1.dimensions)
        torsions = calc_dihedrals(self._ag1.positions,
                                  self._ag2.positions,
                                  self._ag3.positions,
                                  self._ag4.positions,
                                  box=self._ag1.dimensions)
        # When appropriate, calculate improper torsions
        shift = torsions[self._primary_torsion_indices]
        shift[self._unique_primary_torsion_indices] = 0.
        torsions -= shift
        # Wrap torsions to between -np.pi and np.pi
        torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi

        self.results.bat[self._frame_index, :] = np.concatenate(
                (root_based, bonds, angles, torsions))

    def load(self, filename, start=None, stop=None, step=None):
        """Loads the bat trajectory from a file in numpy binary format

        Parameters
        ----------
        filename : str
            name of numpy binary file
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame

        See Also
        --------
        save: Saves the bat trajectory in a file in numpy binary format
        """
        logger.info("Choosing frames")
        self._setup_frames(self._trajectory, start, stop, step)

        logger.info("Loading file")
        self.results.bat = np.load(filename)

        # Check array dimensions
        if self.results.bat.shape != (self.n_frames, 3*self._ag.n_atoms):
            errmsg = ('Dimensions of array in loaded file, '
                      f'({self.results.bat.shape[0]},'
                      f'{self.results.bat.shape[1]}), differ from required '
                      f'dimensions of ({self.n_frames, 3*self._ag.n_atoms})')
            raise ValueError(errmsg)
        # Check position of initial atom
        if (self.results.bat[0, :3] != self._root[0].position).any():
            raise ValueError('Position of initial atom in file ' + \
                'inconsistent with current trajectory in starting frame.')
        return self

    def save(self, filename):
        """Saves the bat trajectory in a file in numpy binary format

        See Also
        --------
        load: Loads the bat trajectory from a file in numpy binary format
        """
        np.save(filename, self.results.bat)

    def Cartesian(self, bat_frame):
        """Conversion of a single frame from BAT to Cartesian coordinates

        One application of this function is to determine the new
        Cartesian coordinates after modifying a specific torsion angle.

        Parameters
        ----------
        bat_frame : numpy.ndarray
            an array with dimensions (3N,) with external then internal
            degrees of freedom based on the root atoms, followed by the bond,
            angle, and (proper and improper) torsion coordinates.

        Returns
        -------
        XYZ : numpy.ndarray
            an array with dimensions (N,3) with Cartesian coordinates. The first
            dimension has the same ordering as the AtomGroup used to initialize
            the class. The molecule will be whole opposed to wrapped around a
            periodic boundary.
        """
        # Split the bat vector into more convenient variables
        origin = bat_frame[:3]
        (phi, theta, omega) = bat_frame[3:6]
        (r01, r12, a012) = bat_frame[6:9]
        n_torsions = (self._ag.n_atoms - 3)
        bonds = bat_frame[9:n_torsions + 9]
        angles = bat_frame[n_torsions + 9:2 * n_torsions + 9]
        torsions = copy.deepcopy(bat_frame[2 * n_torsions + 9:])
        # When appropriate, convert improper to proper torsions
        shift = torsions[self._primary_torsion_indices]
        shift[self._unique_primary_torsion_indices] = 0.
        torsions += shift
        # Wrap torsions to between -np.pi and np.pi
        torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi

        # Set initial root atom positions based on internal coordinates
        p0 = np.array([0., 0., 0.])
        p1 = np.array([0., 0., r01])
        p2 = np.array([r12 * np.sin(a012), 0., r01 - r12 * np.cos(a012)])

        # Rotate the third atom by the appropriate value
        co = np.cos(omega)
        so = np.sin(omega)
        # $R_Z(\omega)$
        Romega = np.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])
        p2 = Romega.dot(p2)
        # Rotate the second two atoms to point in the right direction
        cp = np.cos(phi)
        sp = np.sin(phi)
        ct = np.cos(theta)
        st = np.sin(theta)
        # $R_Z(\phi) R_Y(\theta)$
        Re = np.array([[cp * ct, -sp, cp * st], [ct * sp, cp, sp * st],
                       [-st, 0, ct]])
        p1 = Re.dot(p1)
        p2 = Re.dot(p2)
        # Translate the first three atoms by the origin
        p0 += origin
        p1 += origin
        p2 += origin

        XYZ = np.zeros((self._ag.n_atoms, 3))
        XYZ[self._root_XYZ_inds[0]] = p0
        XYZ[self._root_XYZ_inds[1]] = p1
        XYZ[self._root_XYZ_inds[2]] = p2

        # Place the remaining atoms
        for ((a0,a1,a2,a3), r01, angle, torsion) \
            in zip(self._torsion_XYZ_inds, bonds, angles, torsions):

            p1 = XYZ[a1]
            p3 = XYZ[a3]
            p2 = XYZ[a2]

            sn_ang = np.sin(angle)
            cs_ang = np.cos(angle)
            sn_tor = np.sin(torsion)
            cs_tor = np.cos(torsion)

            v21 = p1 - p2
            v21 /= np.sqrt(np.einsum('i,i->',v21,v21))
            v32 = p2 - p3
            v32 /= np.sqrt(np.einsum('i,i->',v32,v32))

            vp = np.cross(v32, v21)
            cs = np.einsum('i,i->',v21,v32)

            sn = max(np.sqrt(1.0 - cs * cs), 0.0000000001)
            vp = vp / sn
            vu = np.cross(vp, v21)

            XYZ[a0] = p1 + \
              r01*(vu*sn_ang*cs_tor + vp*sn_ang*sn_tor - v21*cs_ang)
        return XYZ

    @property
    def atoms(self):
        """The atomgroup for which BAT are computed (read-only property)"""
        return self._ag