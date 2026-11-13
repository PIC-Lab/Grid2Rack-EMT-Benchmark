"""Load and validate a network configuration built from PSS/E RAW/DYR (psse-network-v1).

Used by network-only cases (simulation/psse_assembly.py); data-center networks use networks/config.py.

The network keeps the source files unchanged. Components that the PSS/E data do
not contain (for example an event-controlled fault breaker) are declared in
``added_components`` and appended to the imported native case at assembly.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from common.config import ConfigError, read_json, require_keys

NETWORK_SCHEMA = 'psse-network-v1'
SUPPORTED_PROVIDERS = ('psse-raw-dyr',)
SUPPORTED_POWER_FLOW = ('pandapower',)


def load_network(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = read_json(path)
    validate_network(config, source=str(path), base=path.parent)
    return config


def source_root(config: dict, base: Path) -> Path:
    """Folder the PSS/E file paths are relative to.

    ``source.root_env`` (an environment variable, e.g. JAX_EMT_PNNL_IEEE39_ROOT) wins when set;
    otherwise ``source.root`` relative to the network file; otherwise the network file folder.
    """
    data = config['source']
    variable = data.get('root_env')
    if variable and os.environ.get(variable):
        return Path(os.environ[variable]).resolve()
    if data.get('root'):
        root = Path(data['root'])
        return root if root.is_absolute() else (base / root).resolve()
    return base


def source_files(config: dict, base: Path) -> dict[str, Path]:
    """Absolute paths of the PSS/E files."""
    root = source_root(config, base)
    files = {}
    for role, value in config['source']['files'].items():
        item = Path(value)
        files[role] = item if item.is_absolute() else (root / item).resolve()
    return files


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_network(config: dict, *, source: str = 'network configuration', base: Path) -> None:
    require_keys(config, ('schema', 'name', 'description', 'source', 'power_flow', 'added_components',
                          'external_inputs', 'monitored_buses', 'limitations'), where=source)
    if config['schema'] != NETWORK_SCHEMA:
        raise ConfigError(f'{source}: schema must be {NETWORK_SCHEMA}')
    data = config['source']
    require_keys(data, ('provider', 'files'), where=source + ' source')
    if data['provider'] not in SUPPORTED_PROVIDERS:
        raise ConfigError(f'{source}: source.provider must be one of {SUPPORTED_PROVIDERS}')
    if set(data['files']) != {'raw', 'dyr'}:
        raise ConfigError(f'{source}: source.files must name exactly raw and dyr')
    for role, path in source_files(config, base).items():
        if not path.is_file():
            variable = data.get('root_env')
            hint = f'; set {variable} to the PNNL IEEE 39 repository folder' if variable else ''
            raise ConfigError(f'{source}: {role} file not found: {path}{hint}')
        expected = data.get('expected_sha256', {}).get(role)
        if expected is not None and sha256(path) != expected:
            raise ConfigError(f'{source}: {role} file {path.name} differs from expected_sha256 '
                              '(source data changed; update the hash only after review)')
    flow = config['power_flow']
    require_keys(flow, ('backend', 'initialization_until_s'), where=source + ' power_flow')
    if flow['backend'] not in SUPPORTED_POWER_FLOW:
        raise ConfigError(f'{source}: power_flow.backend must be one of {SUPPORTED_POWER_FLOW}')
    if not float(flow['initialization_until_s']) > 0:
        raise ConfigError(f'{source}: power_flow.initialization_until_s must be positive')
    added = set()
    for item in config['added_components']:
        require_keys(item, ('component_id', 'model_type', 'ports', 'parameters'),
                     where=f'{source} added component')
        if item['component_id'] in added:
            raise ConfigError(f'{source}: duplicate added component {item["component_id"]}')
        added.add(item['component_id'])
        if not item['ports'] or not all(isinstance(nodes, list) and nodes for nodes in item['ports'].values()):
            raise ConfigError(f'{source}: {item["component_id"]} ports must map to node lists')
    names = set()
    for binding in config['external_inputs']:
        require_keys(binding, ('source', 'target_component_id', 'target_port'), where=f'{source} external input')
        if binding['source'].get('kind') != 'external-input':
            raise ConfigError(f'{source}: external inputs must have source.kind external-input')
        name = binding['source']['input_id']
        if binding['source'].get('definition', {}).get('name') != name:
            raise ConfigError(f'{source}: external input {name} definition.name must equal input_id')
        if name in names:
            raise ConfigError(f'{source}: duplicate external input {name}')
        names.add(name)
    for key, value in config['monitored_buses'].items():
        if not isinstance(value, (str, list)):
            raise ConfigError(f'{source}: monitored_buses.{key} must be a bus id or a list of bus ids')
