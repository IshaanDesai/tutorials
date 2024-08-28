import numpy as np

from pymor.algorithms.timestepping import TimeStepper
from pymor.models.interface import Model, OutputDMuResult
from pymor.operators.constructions import ConstantOperator, IdentityOperator, VectorOperator, ZeroOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

from micro import MicroSimulation

class NutilsModel(Model):

    @classmethod
    def create(cls, dt, id):
        nutils_model = MicroSimulation(id)
        nutils_model.initialize()
        u, phi = nutils_model._solu, nutils_model._solphi
        u = NumpyVectorSpace.from_numpy(u)
        phi = NumpyVectorSpace.from_numpy(phi)
        initial_data = BlockVectorSpace.make_array([phi, u])
        return cls(dt, initial_data, nutils_model)

    def __init__(self, dt, initial_data, nutils_model):

        if isinstance(initial_data, VectorArray):
            initial_data = VectorOperator(initial_data, name='initial_data')
        super().__init__()
        self.solution_space = initial_data.range

        self.parameters_own = {'T': 1}
        self.__auto_init(locals())

    def _compute(self, quantities, data, mu=None):
        if 'solution' in quantities or 'output' in quantities:
            mu = mu.with_(t=0.)
            # U0 = self.initial_data.as_range_array(mu)
            U0 = self.initial_data.as_vector()

            self.nutils_model._solphi = U0.blocks[0].to_numpy().ravel()
            self.nutils_model._solu = U0.blocks[1].to_numpy().ravel()

            output = self.nutils_model.solve({'concentration': mu['T'].item()}, self.dt)
            output = np.array([output['k_00'], output['k_01'], output['k_10'], output['k_11'], output['porosity']])

            phi, u = self.nutils_model._solphi, self.nutils_model._solu
            U = self.solution_space.from_numpy(np.hstack([phi, u]))

            data['solution'] = U
            data['output'] = output.reshape((1, -1))
            if 'solution' in quantities:
                quantities.remove('solution')
            if 'output' in quantities:
                quantities.remove('output')

        super()._compute(quantities, data, mu=mu)


def main():
    pymor_model = NutilsModel.create(dt=1e-3, id=0)

    U = pymor_model.initial_data
    for k in range(10):
        data = pymor_model.with_(initial_data=U).compute(solution=True, output=True, mu=0.4)
        output, U  = data['output'], data['solution']
        print(output)

if __name__ == "__main__":
    main()
