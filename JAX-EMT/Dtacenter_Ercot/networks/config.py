"""Load and validate an external-network configuration (ercot-network-v1)."""
from __future__ import annotations

from pathlib import Path

from common.config import ConfigError, read_json, require_keys, resolve_references

SCHEMA = 'ercot-network-v1'
TOP_LEVEL = ('schema', 'name', 'description', 'system_base_mva', 'nominal_frequency_hz', 'nodes',
             'components', 'external_inputs', 'datacenter_interfaces', 'initialization', 'limitations')


def load_network(path: str | Path, datacenter: dict) -> dict:
    raw = read_json(path)
    config = resolve_references(raw, {'network': raw, 'datacenter': datacenter}, own='network')
    validate_network(config, source=str(path))
    return config


def validate_network(config: dict, *, source: str = 'network configuration') -> None:
    where = source
    require_keys(config, TOP_LEVEL, where=where)
    if config['schema'] != SCHEMA:
        raise ConfigError(f'{where}: schema must be {SCHEMA}')
    if config['system_base_mva'] <= 0 or config['nominal_frequency_hz'] <= 0:
        raise ConfigError(f'{where}: system base and frequency must be positive')
    nodes = [n['node_id'] for n in config['nodes']]
    if len(set(nodes)) != len(nodes):
        raise ConfigError(f'{where}: duplicate node identifiers')
    ids = [c['component_id'] for c in config['components']]
    if len(set(ids)) != len(ids):
        raise ConfigError(f'{where}: duplicate component identifiers')
    for component in config['components']:
        require_keys(component, ('component_id', 'model_type', 'parameters', 'ports'),
                     where=f'{where} component')
        for port, attached in component['ports'].items():
            missing = [n for n in attached if n not in nodes]
            if missing:
                raise ConfigError(f'{where}: {component["component_id"]}.{port} uses unknown node(s) {missing}')
    for binding in config['external_inputs']:
        if binding['target_component_id'] not in ids:
            raise ConfigError(f'{where}: external input targets unknown component '
                              f'{binding["target_component_id"]}')
    for name, interface in config['datacenter_interfaces'].items():
        require_keys(interface, ('bus', 'supply_transformer'), where=f'{where} interface {name}')
        if interface['bus'] not in nodes:
            raise ConfigError(f'{where}: interface {name} bus {interface["bus"]} is not a network node')
        transformer = next((c for c in config['components']
                            if c['component_id'] == interface['supply_transformer']), None)
        if transformer is None or transformer['model_type'] != 'THREE_PHASE_TRANSFORMER':
            raise ConfigError(f'{where}: interface {name} supply transformer must be a THREE_PHASE_TRANSFORMER')
    seeds = config['initialization'].get('node_voltage_seeds_pu', {})
    missing = [n for n in nodes if n not in seeds]
    if missing:
        raise ConfigError(f'{where}: missing node voltage seeds for {missing}')
    for name, bus in config.get('monitored_buses', {}).items():
        if bus not in nodes:
            raise ConfigError(f'{where}: monitored bus {name} -> {bus} is not a network node')


def interface(config: dict, name: str) -> dict:
    try:
        return config['datacenter_interfaces'][name]
    except KeyError as exc:
        raise ConfigError(f'Network {config["name"]} has no data-center interface {name!r}') from exc
