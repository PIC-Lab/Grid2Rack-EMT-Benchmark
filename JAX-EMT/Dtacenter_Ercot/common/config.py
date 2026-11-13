"""JSON configuration loading shared by the data-center, network and case files.

A string value of exactly ``"${path.to.value}"`` is replaced by the value at
that dotted path of the same document. ``"${datacenter:path}"`` refers to the
data-center document attached at assembly time. References keep single-source
values (for example the aggregation unit count) without changing their type.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = re.compile(r'^\$\{(?:(?P<scope>[a-z_]+):)?(?P<path>[A-Za-z0-9_.]+)\}$')


class ConfigError(ValueError):
    """Raised for missing, inconsistent or invalid configuration."""


def read_json(path: str | Path) -> dict:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ConfigError(f'Configuration file not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f'Invalid JSON in {path}: {exc}') from exc


def lookup(document: Mapping, dotted: str, *, label: str = 'document') -> Any:
    value: Any = document
    for key in dotted.split('.'):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f'Reference ${{{dotted}}} not found in {label}')
        value = value[key]
    return value


def resolve_references(document: Any, scopes: Mapping[str, Mapping], *, own: str) -> Any:
    """Return a deep copy with every ``${...}`` reference replaced."""

    def visit(value: Any, depth: int = 0) -> Any:
        if depth > 64:
            raise ConfigError('Configuration nesting or reference chain is too deep')
        if isinstance(value, dict):
            return {key: visit(item, depth + 1) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, depth + 1) for item in value]
        if isinstance(value, str):
            match = _REFERENCE.match(value)
            if match:
                scope = match.group('scope') or own
                if scope not in scopes:
                    raise ConfigError(f'Reference scope "{scope}" is not available for {value}')
                return visit(copy.deepcopy(lookup(scopes[scope], match.group('path'), label=scope)),
                             depth + 1)
        return value

    return visit(document)


def unresolved_references(document: Any) -> list[str]:
    found = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and _REFERENCE.match(value):
            found.append(value)

    visit(document)
    return found


def require_keys(mapping: Mapping, keys, *, where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f'{where}: missing required key(s) {missing}')
