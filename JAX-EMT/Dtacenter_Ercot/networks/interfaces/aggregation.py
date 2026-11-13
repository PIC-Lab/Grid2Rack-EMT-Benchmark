"""PSCAD NO_LL representative-circuit aggregation at the network side.

Moved verbatim from JAXEMT_Model/ercot_circuit_models.py.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PSCADPrimaryCurrentScaler:
    """Scale grid-side transformer current, retaining representative LV physics.

    PSCAD's NO_LL scaler adds (N-1)*Iprimary at the upstream bus.
    Multiplying only primary Norton rows implements that same algebraic
    boundary. The child owns all transformer equations and integration.
    """
    child: object
    scale: float

    def __getattr__(self, name):
        return getattr(self.child, name)

    @property
    def schema(self):
        return self.child.schema

    @property
    def accepted_node_domains(self):
        return self.child.accepted_node_domains

    def initialize(self, *args):
        return self.child.initialize(*args)

    def predict(self, *args):
        return self.child.predict(*args)

    def retime_companion(self, *args):
        return self.child.retime_companion(*args)

    def algebraic_state(self, *args):
        return self.child.algebraic_state(*args)

    def integration_method_variant(self, method):
        return replace(self, child=self.child.integration_method_variant(method))

    def _scaled(self, ops, stamp):
        rows = ops.asarray([self.scale]*3 + [1.0]*3)
        return stamp._replace(voltage_gain=stamp.voltage_gain*rows[None, :, None],
                              source=stamp.source*rows[None, :])

    def norton(self, ops, *args):
        return self._scaled(ops, self.child.norton(ops, *args))

    def iterate_norton(self, ops, *args):
        if hasattr(self.child, 'iterate_norton'):
            return self._scaled(ops, self.child.iterate_norton(ops, *args))
        return self.norton(ops, *args[:-1])

    def algebraic_norton(self, ops, *args):
        return self._scaled(ops, self.child.algebraic_norton(ops, *args))

    def commit(self, ops, *args):
        result = self.child.commit(ops, *args)
        factors = ops.asarray([self.scale if name.startswith('from_network_injection_') else 1.0
                               for name in self.schema.output_names])
        return result._replace(outputs=result.outputs*factors[None, :])
