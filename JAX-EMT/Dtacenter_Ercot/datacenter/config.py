"""Load and validate a data-center subsystem configuration (ercot-datacenter-v1)."""
from __future__ import annotations

from math import isclose, isfinite, pi
from pathlib import Path

from common.config import (ConfigError, read_json, require_keys, resolve_references,
                           unresolved_references)

SCHEMA = 'ercot-datacenter-v1'
TOP_LEVEL = ('schema', 'name', 'description', 'terminal', 'aggregation', 'facility', 'internal_nodes',
             'branches', 'signal_connections', 'external_inputs', 'initialization', 'limitations')
FACILITY_SECTIONS = ('ratings', 'operating_mode', 'numerics', 'controls', 'protection', 'rectifier',
                     'inverter', 'dcdc_and_battery', 'electrical', 'aggregation_links')
FACILITY_PORTS = ('terminal', 'protected_load', 'inverter_output')
# Component identifiers used by the retained startup/filter initialization algebra.
FILTER_ROLE_IDS = {
    'inverter_standby_enabled': ('INV_FILTER_C', 'INV_DAMP_C', 'INV_DAMP_R', 'INV_DAMP_L'),
    'rectifier_terminal_power_measurement': ('RECT_FILTER_C', 'RECT_DAMP_C', 'RECT_DAMP_R', 'RECT_DAMP_L'),
}
TAMU_PRESET = 'tamu_ercot_2026_centralized_ups'


def terminal_token(config: dict) -> str:
    return '@' + config['terminal']['name']


def facility_values(config: dict) -> dict:
    """Flatten the facility sections back to the native parameter/metadata keys."""
    values = {}
    for section in FACILITY_SECTIONS:
        for key, value in config['facility'].get(section, {}).items():
            if key in values:
                raise ConfigError(f'Facility key {key} appears in more than one section')
            values[key] = value
    return values


def load_datacenter(path: str | Path, overrides: dict | None = None) -> dict:
    """Load and validate a data-center JSON; ``overrides`` (a case's ``datacenter_overrides``) is
    deep-merged first, so one case can change data-center settings without editing the shared file."""
    raw = read_json(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    config = resolve_references(raw, {'datacenter': raw}, own='datacenter')
    validate_datacenter(config, source=str(path) + (' (with case datacenter_overrides)' if overrides else ''))
    return config


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if key.startswith('_'):
            continue                       # "_description"-style comments in the override block
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_datacenter(config: dict, *, source: str = 'data-center configuration') -> None:
    where = f'{source}'
    require_keys(config, TOP_LEVEL, where=where)
    if config['schema'] != SCHEMA:
        raise ConfigError(f'{where}: schema must be {SCHEMA}')
    leftovers = unresolved_references(config)
    if leftovers:
        raise ConfigError(f'{where}: unresolved references {leftovers}')

    terminal = config['terminal']
    require_keys(terminal, ('name', 'line_line_voltage_v', 'nominal_frequency_hz', 'phases'),
                 where=where + ' terminal')
    if terminal['line_line_voltage_v'] <= 0 or terminal['nominal_frequency_hz'] <= 0:
        raise ConfigError(f'{where}: terminal voltage and frequency must be positive')

    aggregation = config['aggregation']
    require_keys(aggregation, ('network_scale_units', 'representative_circuit_units',
                               'representative_it_power_w'), where=where + ' aggregation')
    for key in ('network_scale_units', 'representative_circuit_units', 'representative_it_power_w'):
        value = aggregation[key]
        if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise ConfigError(f'{where}: aggregation.{key} must be finite and positive')

    facility = config['facility']
    require_keys(facility, ('component_id', 'model_type', 'preset_id', 'ports', *FACILITY_SECTIONS),
                 where=where + ' facility')
    if set(facility['ports']) != set(FACILITY_PORTS):
        raise ConfigError(f'{where}: facility ports must be {FACILITY_PORTS}')
    values = facility_values(config)
    require_keys(values, ('facility_mva', 'it_fraction', 'cooling_fraction', 'support_fraction',
                          'facility_terminal_line_line_voltage_v', 'network_aggregation_scale',
                          'phase_connections'), where=where + ' facility')
    fractions = [values[k] for k in ('it_fraction', 'cooling_fraction', 'support_fraction')]
    if any(f <= 0 for f in fractions) or not isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ConfigError(f'{where}: IT/cooling/support fractions must be positive and sum to 1')
    if values['facility_terminal_line_line_voltage_v'] != terminal['line_line_voltage_v']:
        raise ConfigError(f'{where}: facility terminal voltage differs from terminal.line_line_voltage_v')
    if values['network_aggregation_scale'] != aggregation['network_scale_units']:
        raise ConfigError(f'{where}: network_aggregation_scale must equal aggregation.network_scale_units')
    if facility['preset_id'] == TAMU_PRESET and 'preset_values' not in config:
        raise ConfigError(f'{where}: preset {TAMU_PRESET} requires preset_values')

    token = terminal_token(config)
    internal = [n['node_id'] for n in config['internal_nodes']]
    if len(set(internal)) != len(internal):
        raise ConfigError(f'{where}: duplicate internal node identifiers')
    allowed_nodes = set(internal) | {token}
    if facility['ports']['terminal'] != [token]:
        raise ConfigError(f'{where}: facility terminal port must be [{token}]')
    for port, nodes in facility['ports'].items():
        if any(node not in allowed_nodes for node in nodes):
            raise ConfigError(f'{where}: facility port {port} uses unknown node(s) {nodes}')
    ids = [facility['component_id']] + [b['component_id'] for b in config['branches']]
    if len(set(ids)) != len(ids):
        raise ConfigError(f'{where}: duplicate component identifiers')
    for branch in config['branches']:
        require_keys(branch, ('component_id', 'model_type', 'parameters', 'ports'),
                     where=f'{where} branch')
        for port, nodes in branch['ports'].items():
            if any(node not in allowed_nodes for node in nodes):
                raise ConfigError(f'{where}: {branch["component_id"]}.{port} uses unknown node(s) {nodes}')
    for flag, role_ids in FILTER_ROLE_IDS.items():
        if values.get(flag, False):
            missing = [cid for cid in role_ids if cid not in ids]
            if missing:
                raise ConfigError(f'{where}: {flag}=true requires filter components {missing}')
    wiring = values['phase_connections']
    if wiring.get('external_bypass_breaker', False):
        breaker = wiring.get('bypass_breaker_component')
        if breaker not in ids:
            raise ConfigError(f'{where}: bypass breaker {breaker} is not a data-center branch')
    for connection in config['signal_connections']:
        for key in ('target_component_id', 'source_component_id'):
            if connection[key] not in ids:
                raise ConfigError(f'{where}: signal connection references external component '
                                  f'{connection[key]}; data-center signals must stay internal')
    for binding in config['external_inputs']:
        if binding['target_component_id'] not in ids:
            raise ConfigError(f'{where}: external input targets unknown component '
                              f'{binding["target_component_id"]}')
    soc = config['initialization'].get('initial_soc')
    if soc is not None and not 0.0 <= soc <= 1.0:
        raise ConfigError(f'{where}: initialization.initial_soc must lie in [0, 1]')
    for cid in config['initialization'].get('component_values', {}):
        if cid not in ids:
            raise ConfigError(f'{where}: initial value for unknown component {cid}')


def check_branch_per_unit(config: dict, system_base_mva: float, frequency_hz: float) -> list[str]:
    """Cross-check per-unit branch values against their SI per-unit metadata.

    Returns human-readable mismatches. Values are not modified.
    """
    zbase = config['terminal']['line_line_voltage_v']**2 / (system_base_mva * 1e6)
    omega = 2 * pi * frequency_hz
    problems = []
    for branch in config['branches']:
        meta = branch.get('metadata', {})
        units = meta.get('parallel_units')
        p = branch['parameters']
        checks = []
        if 'capacitance_f_per_unit' in meta and units:
            checks.append(('susceptance_pu', omega * meta['capacitance_f_per_unit'] * units * zbase))
        if 'resistance_ohm_per_unit' in meta and units:
            checks.append(('resistance_pu', meta['resistance_ohm_per_unit'] / units / zbase))
        if 'inductance_h_per_unit' in meta and units:
            checks.append(('reactance_pu', omega * meta['inductance_h_per_unit'] / units / zbase))
        for key, expected in checks:
            if key in p and not isclose(p[key], expected, rel_tol=1e-9, abs_tol=0.0):
                problems.append(f'{branch["component_id"]}.{key}={p[key]!r} but SI metadata gives {expected!r}')
    return problems
