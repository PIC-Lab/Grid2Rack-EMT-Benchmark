"""PSCAD inverter shared-neutral (star return) path as a native RL projection.

Moved verbatim from JAXEMT_Model/ercot_circuit_models.py.
"""
from dataclasses import dataclass, replace

import numpy as np

import simulation.bootstrap  # noqa: F401  (installs the frozen model source first)
from jax_emt.modeling import EngineKind


@dataclass(frozen=True)
class NativeCommonModeRL:
    """Kron-reduced inverter star path using the native RL companion.

    With three equal arms and a shared neutral resistor, the zero-sequence
    arm impedance is Rarm + 3*Rneutral + s*Larm. The native converter already
    owns the differential-mode arms. P0=11'/3 selects only the missing star
    return; no positive-sequence shunt or artificial ground is introduced.
    """
    integration_method: str = 'backward_euler'

    @property
    def accepted_node_domains(self):
        from jax_emt.runtime import NodeDomain
        return (frozenset((NodeDomain.AC_PHASE,)),) * 3

    @property
    def child(self):
        from jax_emt.components.dynamic.emt_passive import FunctionalThreePhaseSeriesRLBranch
        return FunctionalThreePhaseSeriesRLBranch(self.integration_method)

    @property
    def schema(self):
        return self.child.schema

    @property
    def parameter_names(self):
        return self.child.parameter_names

    integration_variant_layout_compatible = True

    def integration_method_variant(self, integration_method):
        return replace(self, integration_method=integration_method)

    def projection(self, ops):
        top = ops.ones((3, 3)) / 3.
        return ops.concatenate((top, ops.zeros_like(top)), axis=0)

    def initialize(self, ops, parameters, operating_point, port_voltage, dt):
        mapped = port_voltage @ self.projection(ops).T
        return self.child.initialize(ops, parameters, operating_point, mapped, dt)

    def predict(self, *args):
        return self.child.predict(*args)

    def retime_companion(self, ops, parameters, state, port_voltage, inputs, dt):
        return self.child.retime_companion(ops, parameters, state,
            port_voltage @ self.projection(ops).T, inputs, dt)

    def norton(self, ops, parameters, predicted_state, port_voltage, inputs, dt, prediction_data):
        from jax_emt.runtime.devices import NortonPortData
        a = self.projection(ops)
        stamp = self.child.norton(ops, parameters, predicted_state,
            port_voltage @ a.T, inputs, dt, prediction_data)
        return NortonPortData(ops.einsum('ij,bjk,kl->bil', a.T, stamp.voltage_gain, a),
                              stamp.source @ a, stamp.iteration_data)

    def commit(self, ops, parameters, predicted_state, solved_port_voltage, inputs,
               dt, prediction_data, iteration_data):
        return self.child.commit(ops, parameters, predicted_state,
            solved_port_voltage @ self.projection(ops).T, inputs, dt, prediction_data, iteration_data)


@dataclass(frozen=True)
class NativeCommonModeRLAdapter:
    integration_method: str
    engine: EngineKind = EngineKind.INSTANTANEOUS_EMT

    def lower(self, context):
        from jax_emt.components.datacenter.base import FunctionalComponentBuildSpec
        kernel = NativeCommonModeRL(self.integration_method)
        parameters = np.asarray([[context.parameters['resistance_pu'],
            context.parameters['reactance_pu'], 2*np.pi*context.nominal_frequency_hz]])
        voltage = np.asarray(context.backend_electrical_port_voltage('terminal')).reshape(1, 3)
        indices = tuple(context.backend_electrical_port_nodes('terminal'))
        operating = np.zeros((1, 3))
        return FunctionalComponentBuildSpec(kernel=kernel,
            indices=np.asarray([indices]),
            parameters=parameters, operating_point=operating, initial_port_voltage=voltage,
            initial_state=kernel.initialize(np, parameters, operating, voltage, context.time_step_s))


def common_mode_rl_initializer(context):
    from jax_emt.initialization import InitializationResult
    return InitializationResult(state=np.zeros((1, 12)),
        residuals={'balanced_inverter_neutral_current': 0.0})
