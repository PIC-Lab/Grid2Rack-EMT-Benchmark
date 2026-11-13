"""Initialization-only battery SOC assignment for the native composite facility.

Moved verbatim from JAXEMT_Model/ercot_circuit_models.py.
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class NativeInitialSOC:
    """Apply a JSON operating-point datum to a SOC-independent battery state.

    The registered preset has no initial_soc parameter. Delegate electrical
    initialization, then set its labeled SOC before native EMT lowering.
    IdealBattery and a constant-OCV ECM without polarization or thermal states
    permit changing SOC within limits without changing electrical equilibrium. No runtime model
    equation, state integration or plotted value is replaced.
    """
    original: object
    factory: object

    def __call__(self, context):
        result = self.original(context)
        if 'initial_soc' not in context.operating_values:
            return result
        from jax_emt.components.datacenter.energy.battery import IdealBattery, BatteryECM

        requested = float(context.operating_values['initial_soc'])
        from jax_emt.modeling.datacenter_catalog import _composite_binding
        binding = _composite_binding(context, self.factory(context.parameters))
        battery = binding.model.it_binding.model.ups.storage_interface.storage
        soc_independent = isinstance(battery, IdealBattery)
        if isinstance(battery, BatteryECM):
            p = battery.parameters
            soc_independent = (len(set(p.ocv_voltage_points_v)) == 1
                and not p.polarization_enabled and not p.thermal_enabled)
        if not soc_independent:
            raise TypeError('SOC-only initialization requires SOC-independent battery electrical behavior')
        if (not np.isfinite(requested) or not battery.parameters.minimum_soc <= requested
                <= battery.parameters.maximum_soc):
            raise ValueError(f'initial_soc must lie within native battery limits '
                             f'[{battery.parameters.minimum_soc}, {battery.parameters.maximum_soc}]')
        indices = [i for i, name in enumerate(binding.state_names)
                   if 'state_of_charge' in name.lower() or name.lower().endswith('_soc')]
        if len(indices) != 1:
            raise ValueError('Cannot identify exactly one native battery SOC state')
        state = np.array(result.state, copy=True)
        if state.shape != (1, binding.state_count + 1):
            raise ValueError('Unexpected native composite state layout during SOC initialization')
        state[0, indices[0]] = requested
        outputs = dict(result.outputs)
        outputs['storage.state_of_charge'] = requested
        return replace(result, state=state, outputs=outputs)
