"""
Micro simulation
In this script we solve the Laplace equation with a grain depicted by a phase field on a square domain :math:`Ω`
with boundary :math:`Γ`, subject to periodic boundary conditions in both dimensions
"""
import math

from nutils import mesh, function, solver, export, cli
import treelog
import numpy as np
from copy import deepcopy


class NutilsMicroSimulation:

    def __init__(self, sim_id):
        """
        Constructor of MicroSimulation class.
        """
        self._sim_id = sim_id

        # Initial parameters
        self._nelems = 10  # Elements in one direction

        # self._ref_level = 3  # Number of levels of mesh refinement
        self._r_initial = 0.4  # Initial radius of the grain

        # Interpolation order for phi and u
        self._degree_phi = 2
        self._degree_u = 2

        # Set up mesh with periodicity in both X and Y directions
        self._topo, self._geom = mesh.rectilinear([np.linspace(-0.5, 0.5, self._nelems + 1)] * 2, periodic=(0, 1))
        self._topo_coarse = self._topo  # Save original coarse topology to use to re-refinement

        self._solu = None  # Solution of weights for which cell problem is solved for
        self._solphi = None  # Solution of phase field
        self._solphinm1 = None  # Solution of phase field at t_{n-1}
        self._solphi_checkpoint = None  # Save the current solution of the phase field as a checkpoint.
        self._topo_checkpoint = None  # Save the refined mesh as a checkpoint.
        self._ucons = None
        self._first_iter_done = False
        self._initial_condition_is_set = False
        self._k_nm1 = None  # Average effective conductivity of last time step

        self._output_n = 0  # Output counter

    def initialize(self):
        # Define initial namespace
        self._ns = function.Namespace()
        self._ns.x = self._geom

        self._ns.ubasis = self._topo.basis('std', degree=self._degree_u).vector(self._topo.ndims)
        self._ns.phibasis = self._topo.basis('std', degree=self._degree_phi)
        self._ns.phi = 'phibasis_n ?solphi_n'  # Initial phase field
        self._ns.lam = 0.4

        self._ns.gam = 0.05
        self._ns.kt = 1.0
        self._ns.eqconc = 0.5  # Equilibrium concentration^
        self._ns.kg = 0.0  # Conductivity of grain material
        self._ns.ks = 1.0  # Conductivity of sand material
        self._ns.reacrate = 'kt (?conc / eqconc)^2 - 1'  # Constructed reaction rate based on macro temperature
        self._ns.u = 'ubasis_ni ?solu_n'  # Weights for which cell problem is solved for
        self._ns.du_ij = 'u_i,j'  # Gradient of weights field
        self._ns.ddwpdphi = '16 phi (1 - phi) (1 - 2 phi)'  # gradient of double-well potential
        self._ns.dphidt = 'phibasis_n (?solphi_n - ?solphinm1_n) / ?dt'  # Implicit time evolution of phase field

        self._ucons = np.zeros(len(self._ns.ubasis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point


        # Initialize phase field
        solphi = self._get_analytical_phasefield(
            self._topo,
            self._ns,
            self._degree_phi,
            self._ns.lam,
            self._r_initial)

        # Refine the mesh
        # self._topo, self._solphi = self._refine_mesh(self._topo, solphi)
        # self._reinitialize_namespace(self._topo)

        self._initial_condition_is_set = True

        # Initialize phase field once more on refined topology
        # solphi = self._get_analytical_phasefield(self._topo, self._ns, self._degree_phi, self._ns.lam, self._r_initial)

        self._solphi = solphi  # Save solution of phi
        psi = self._get_avg_porosity(self._topo, solphi)
        self._psi_nm1 = psi  # Average porosity value of last time step

        # Solve the heat cell problem
        solu = self._solve_heat_cell_problem(self._topo, solphi)
        k = self._get_eff_conductivity(self._topo, solu, solphi)

        self._solu = solu  # Save solution for output

        output_data = dict()
        output_data["k_00"] = k[0][0]
        output_data["k_11"] = k[1][1]
        output_data["porosity"] = psi

        return output_data

    @staticmethod
    def _analytical_phasefield(x, y, r, lam):
        return 1. / (1. + np.exp(-4. / lam * (np.sqrt(x ** 2 + y ** 2) - r)))

    @staticmethod
    def _get_analytical_phasefield(topo, ns, degree_phi, lam, r):
        phi_ini = NutilsMicroSimulation._analytical_phasefield(ns.x[0], ns.x[1], r, lam)
        sqrphi = topo.integral((ns.phi - phi_ini) ** 2, degree=degree_phi * 2)
        solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

        return solphi

    def output(self):
        bezier = self._topo.sample('bezier', 2)
        x, u, phi = bezier.eval(['x_i', 'u_i', 'phi'] @ self._ns, solu=self._solu, solphi=self._solphi)
        with treelog.add(treelog.DataLog()):
            export.vtk("micro-heat-{}-{}".format(self._output_n, self._sim_id), bezier.tri, x, T=u, phi=phi)
        self._output_n += 1

    def get_state(self):
        return [self._solphi.copy(), deepcopy(self._topo)]

    def set_state(self, state):
        self._solphi = state[0]
        self._topo = state[1]
        # self._reinitialize_namespace(self._topo)  # The namespace also needs to reloaded to its earlier state

    def _solve_allen_cahn(self, topo, phi_coeffs_nm1, concentration, dt):
        """
        Solving the Allen-Cahn equation using a Newton solver.
        Returns porosity of the micro domain.
        """
        self._first_iter_done = True
        resphi = topo.integral('(lam^2 phibasis_n dphidt + gam phibasis_n ddwpdphi + gam lam^2 phibasis_n,i phi_,i + '
                               '4 lam reacrate phibasis_n phi (1 - phi)) d:x' @ self._ns, degree=self._degree_phi * 2)

        args = dict(solphinm1=phi_coeffs_nm1, dt=dt, conc=concentration)
        phi_coeffs = solver.newton('solphi', resphi, lhs0=phi_coeffs_nm1, arguments=args).solve(tol=1E-12)

        return phi_coeffs

    def _get_avg_porosity(self, topo, phi_coeffs):
        psi = topo.integral('phi d:x' @ self._ns, degree=self._degree_phi * 2).eval(solphi=phi_coeffs)

        return psi

    def _solve_heat_cell_problem(self, topo, phi_coeffs):
        """
        Solving the P1 homogenized heat equation
        Returns upscaled conductivity for the micro domain
        """
        res = topo.integral('((phi ks + (1 - phi) kg) u_i,j ubasis_ni,j - '
                            '(ks - kg) phi_,j $_ij ubasis_ni) d:x' @ self._ns, degree=self._degree_u * 2)

        args = dict(solphi=phi_coeffs)
        u_coeffs = solver.solve_linear('solu', res, constrain=self._ucons, arguments=args)

        return u_coeffs

    def _get_eff_conductivity(self, topo, u_coeffs, phi_coeffs):
        b = topo.integral(
            self._ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'),
            degree=self._degree_u *
            2).eval(
            solu=u_coeffs,
            solphi=phi_coeffs)

        return b.export("dense")

    def solve(self, macro_data, dt):
        if self._psi_nm1 < 0.95:
            # TRIAL: Do not refine the mesh
            #topo, solphi = self._refine_mesh(self._topo, self._solphi)
            #self._reinitialize_namespace(topo)

            # Simply set topo as self._topo and solphi as self._solphi
            topo = self._topo
            solphi = self._solphi

            assert ((solphi >= 0.0) & (solphi <= 1.0)).all()

            solphi = self._solve_allen_cahn(topo, solphi, macro_data["concentration"], dt)
            psi = self._get_avg_porosity(topo, solphi)

            solu = self._solve_heat_cell_problem(topo, solphi)
            k = self._get_eff_conductivity(topo, solu, solphi)

            # Save state variables
            self._topo = topo
            self._solphi = solphi
            self._solu = solu
            self._psi_nm1 = psi
            self._k_nm1 = k
        else:
            # Micro simulation has reached max porosity limit and hence is not solved
            k = self._k_nm1
            psi = self._psi_nm1

        output_data = dict()
        output_data["k_00"] = k[0][0]
        output_data["k_01"] = k[0][1]
        output_data["k_10"] = k[1][0]
        output_data["k_11"] = k[1][1]
        output_data["porosity"] = psi
        output_data["grain_size"] = math.sqrt((1 - psi) / math.pi)

        return output_data


def main():
    micro_problem = MicroSimulation(0)
    dt = 1e-3
    micro_problem.initialize()
    concentrations = [0.5, 0.4]
    t = 0.0
    n = 0
    concentration = dict()

    for conc in concentrations:
        concentration["concentration"] = conc

        micro_sim_output = micro_problem.solve(concentration, dt)

        micro_problem.output()
        t += dt
        n += 1
        print(micro_sim_output)


if __name__ == "__main__":
    cli.run(main)
