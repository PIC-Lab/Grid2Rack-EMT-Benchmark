"""Build a normalized-case fragment for one data-center subsystem.

The fragment contains the facility, its LV filters, neutral path, bypass
breaker, internal nodes, internal signal links and external inputs. Its
terminal token is replaced by the network bus chosen by the case.
"""
from __future__ import annotations

import copy

from common.config import ConfigError
from datacenter.config import facility_values, terminal_token


def native_parameter_names(registry) -> frozenset[str]:
    registration = registry.registration_for('DATA_CENTER_COMPOSITE_FACILITY')
    return frozenset(d.name for d in registration.parameter_schema.definitions)


def build_fragment(config: dict, *, terminal_bus: str, terminal_seed, parameter_names) -> dict:
    token = terminal_token(config)

    def node(name):
        return terminal_bus if name == token else name

    facility = config['facility']
    values = facility_values(config)
    if 'preset_id' in values:
        raise ConfigError('preset_id belongs to facility.preset_id, not to a section')
    parameters = {'preset_id': facility['preset_id']}
    metadata = {}
    for key, value in values.items():
        (parameters if key in parameter_names else metadata)[key] = copy.deepcopy(value)
    metadata['description'] = config['description']
    records = [{'component_id': facility['component_id'], 'model_type': facility['model_type'],
                'parameters': parameters, 'metadata': metadata}]
    connections = [{'component_id': facility['component_id'], 'port': port,
                    'node_ids': [node(n) for n in nodes]} for port, nodes in facility['ports'].items()]
    for branch in config['branches']:
        record = {k: copy.deepcopy(v) for k, v in branch.items() if k != 'ports'}
        records.append(record)
        connections.extend({'component_id': branch['component_id'], 'port': port,
                            'node_ids': [node(n) for n in nodes]} for port, nodes in branch['ports'].items())

    init = config['initialization']
    rule = init.get('internal_node_voltage_seed', {}).get('rule', 'terminal')
    if rule != 'terminal':
        raise ConfigError(f'Unsupported internal node seed rule {rule!r}')
    seeds = {n['node_id']: list(terminal_seed) for n in config['internal_nodes']}
    component_values = copy.deepcopy(init.get('component_values', {}))
    if init.get('initial_soc') is not None:
        component_values.setdefault(facility['component_id'], {})['initial_soc'] = init['initial_soc']
    return {
        'nodes': copy.deepcopy(config['internal_nodes']),
        'components': records,
        'electrical_connections': connections,
        'signal_connections': copy.deepcopy(config['signal_connections']),
        'signal_bindings': copy.deepcopy(config['external_inputs']),
        'node_voltage_seeds_pu': seeds,
        'component_values': component_values,
        'limitations': list(config['limitations']),
    }
