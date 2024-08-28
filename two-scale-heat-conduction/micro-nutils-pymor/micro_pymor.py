from pymor.algorithms.timestepping import TimeStepper
from pymor.models.interface import Model, OutputDMuResult
from pymor.operators.constructions import ConstantOperator, IdentityOperator, VectorOperator, ZeroOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace

from micro import MicroSimulation

class InstationaryModel(Model):
    """Generic class for models of instationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent vector-like |Operator|, and u_0 the
    initial data. The mass |Operator| M is assumed to be linear.

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `u_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for given |parameter values|.
    operator
        The |Operator| L.
    rhs
        The right-hand side F.
    mass
        The mass |Operator| `M`. If `None`, the identity is assumed.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
        to be used by :meth:`~pymor.models.interface.Model.solve`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, T, initial_data, operator, rhs, mass=None, time_stepper=None, num_values=None,
                 output_functional=None, products=None, error_estimator=None, visualizer=None, name=None):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')
        if isinstance(initial_data, VectorArray):
            assert initial_data in operator.source
            initial_data = VectorOperator(initial_data, name='initial_data')
        # mass = mass or IdentityOperator(operator.source)
        # rhs = rhs or ZeroOperator(operator.source, NumpyVectorSpace(1))
        # output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), operator.source)

        # assert isinstance(time_stepper, TimeStepper)
        # assert initial_data.source.is_scalar
        # assert operator.source == initial_data.range
        # assert rhs.linear
        # assert rhs.range == operator.range
        # assert rhs.source.is_scalar
        # assert mass.linear
        # assert mass.source == mass.range
        # assert mass.source == operator.source
        # assert output_functional.source == operator.source

        # try:
        #     dim_input = [op.parameters['input']
        #                  for op in [operator, rhs, output_functional] if 'input' in op.parameters].pop()
        # except IndexError:
        #     dim_input = 0

        # super().__init__(dim_input=dim_input, products=products, error_estimator=error_estimator,
        #                  visualizer=visualizer, name=name)

        self.parameters_internal = dict(self.parameters_internal, t=1)
        self.__auto_init(locals())
        # self.solution_space = operator.source
        # self.linear = operator.linear and (output_functional is None or output_functional.linear)
        # self.dim_output = output_functional.range.dim

        # Create Nutils micro simulation model object
        self._nutils_model = MicroSimulation(0)

        # Initialize the micro simulation model
        self._initial_data = self._nutils_model.initialize()

    def _compute(self, quantities, data, mu=None):
        if 'solution' in quantities:
            mu = mu.with_(t=0.)
            # U0 = self.initial_data.as_range_array(mu)
            U0 = self._initial_data

            # U = self.time_stepper.solve(operator=self.operator,
            #                             rhs=None if isinstance(self.rhs, ZeroOperator) else self.rhs,
            #                             initial_data=U0,
            #                             mass=None if isinstance(self.mass, IdentityOperator) else self.mass,
            #                             initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)

            U = self._nutils_model.solve(data, self.T)
            data['solution'] = U
            quantities.remove('solution')

        super()._compute(quantities, data, mu=mu)


def main():
    pymor_model = InstationaryModel(0, None, None, None)
    dt = 1e-3

    # input concentrations for the model
    concentrations = VectorArray([0.5, 0.4, 0.3, 0.2])

    pymor_model = InstationaryModel(dt, concentrations, None, None, name='nutils_micro')

    input_data = dict()
    input_data["concentration"] = 0.5
    pymor_model.solve(input_data, dt)

if __name__ == "__main__":
    main()
