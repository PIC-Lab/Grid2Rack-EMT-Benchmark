"""JSON-driven TAMU/ERCOT centralized-UPS preset.

Rebuilds the same architecture as the frozen ``TAMUERCOT2026Preset.build()``
(model_source/.../architectures/centralized_ups.py) with every value taken
from ``preset_values`` of the data-center JSON. The construction steps and
equations are unchanged; tests/test_preset.py checks field-by-field equality
with the original builder.

``install_datacenter_preset`` replaces the name ``TAMUERCOT2026Preset`` used by
``functional_facility.CompositeFacilityConfiguration.build_binding``. The frozen
source file itself is not edited.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from math import pi
from types import MappingProxyType

import simulation.bootstrap  # noqa: F401
from jax_emt.components.datacenter import functional_facility
from jax_emt.components.datacenter.architectures import centralized_ups as native
from jax_emt.components.datacenter.protection.ercot_ups import (ERCOTUPSProtectionSettings,
                                                                ercot_ups_protection_domain)

_ORIGINAL = native.TAMUERCOT2026Preset
_VALUES: dict | None = None


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def _protection_settings(values: dict) -> ERCOTUPSProtectionSettings:
    fields = {f.name for f in dataclasses.fields(ERCOTUPSProtectionSettings)}
    unknown = set(values) - fields
    if unknown:
        raise ValueError(f'Unknown ERCOT UPS protection settings: {sorted(unknown)}')
    return ERCOTUPSProtectionSettings(**{k: _tuples(v) for k, v in values.items()})


@dataclass(frozen=True)
class JSONTAMUERCOTPreset(_ORIGINAL):
    """Same signature and validation as TAMUERCOT2026Preset; values from JSON."""

    def build(self):
        if _VALUES is None:
            raise RuntimeError('install_datacenter_preset() must be called before building the preset')
        v = _VALUES
        a = v['architecture']
        supervisor = v['supervisor_by_reconnection_strategy'][self.reconnection_strategy]
        parameters = native.CentralizedUPSArchitectureParameters(
            facility_mva=self.facility_mva,
            it_fraction=self.it_fraction,
            cooling_fraction=self.cooling_fraction,
            support_fraction=self.support_fraction,
            ups_reserve_ratio=a['ups_reserve_ratio'],
            nominal_line_line_voltage_v=a['nominal_line_line_voltage_v'],
            dc_link_voltage_v=a['dc_link_voltage_v'],
            storage_voltage_v=a['storage_voltage_v'],
            unit_ups_output_power_w=a['unit_ups_output_power_w'],
            unit_dc_capacitance_f=a['unit_dc_capacitance_f'],
            unit_cooling_input_power_w=a['unit_cooling_input_power_w'],
            unit_cooling_motor_power_w=a['unit_cooling_motor_power_w'],
            inverter_filter_reactance_pu=a['inverter_filter_reactance_pu'],
            rectifier_current_limit_pu=a['rectifier_current_limit_pu'],
            inverter_current_limit_pu=a['inverter_current_limit_pu'],
            storage_duration_s=self.storage_duration_s,
            bypass_resistance_pu=a['bypass_resistance_pu'],
            bypass_undervoltage_threshold_pu=a['bypass_undervoltage_threshold_pu'],
            bypass_undervoltage_hold_s=a['bypass_undervoltage_hold_s'],
            supervisor=native.UPSSupervisorParameters(
                reconnect_delay_s=supervisor['reconnect_delay_s'],
                reconnect_ramp_s=supervisor['reconnect_ramp_s'],
                battery_release_delay_s=supervisor['battery_release_delay_s'],
            ),
            rectifier_dc_proportional_gain_scale=a['rectifier_dc_proportional_gain_scale'],
        )
        record = v['provenance']
        source_parameters = {k: _tuples(value) for k, value in record['source_parameters'].items()}
        source_parameters.update({
            'selected_reconnection_strategy': self.reconnection_strategy,
            'selected_it_fraction': self.it_fraction,
            'selected_cooling_fraction': self.cooling_fraction,
            'selected_support_fraction': self.support_fraction,
            'selected_storage_duration_s': self.storage_duration_s,
        })
        provenance = native.CentralizedUPSProvenance(
            record['source'], record['edition'], record['artifact'], record['artifact_sha256'],
            source_parameters, tuple(record['engineering_assumptions']))
        architecture = native._build_architecture(parameters, provenance)
        inverter = v['inverter_bridge']
        unit_inverter_va = inverter['unit_rating_va']
        units = parameters.protected_it_power_w / parameters.unit_ups_output_power_w
        unit_zbase = parameters.nominal_line_line_voltage_v**2 / unit_inverter_va
        inverter_r = (inverter['series_r_pu_on_unit_base'] * unit_zbase
                      + inverter['internal_breaker_closed_resistance_ohm']) / units
        inverter_l = (inverter['series_x_pu_on_unit_base'] * unit_zbase
                      / (2.0 * pi * inverter['base_frequency_hz']) / units)
        stage = architecture.ups.output_stage
        stage = replace(stage,
            bridge=replace(stage.bridge, parameters=replace(stage.bridge.parameters,
                filter_resistance_ohm=inverter_r, filter_inductance_h=inverter_l)),
            controller=replace(stage.controller, parameters=replace(stage.controller.parameters,
                filter_resistance_ohm=inverter_r, filter_inductance_h=inverter_l,
                power_measurement_time_constant_s=inverter['power_measurement_time_constant_s'],
                power_measurement_base_va=unit_inverter_va*units)))
        rect = v['rectifier_bridge']
        rectifier = architecture.ups.rectifier
        rectifier_zbase = parameters.nominal_line_line_voltage_v**2 / rect['unit_rating_va']
        rectifier_r = (rect['series_r_pu_on_unit_base']*rectifier_zbase
                       + rect['internal_breaker_closed_resistance_ohm'])/units
        rectifier = replace(rectifier,
            bridge=replace(rectifier.bridge, parameters=replace(rectifier.bridge.parameters,
                filter_resistance_ohm=rectifier_r)),
            controller=replace(rectifier.controller, parameters=replace(rectifier.controller.parameters,
                filter_resistance_ohm=rectifier_r)))
        protection = ercot_ups_protection_domain(_protection_settings(v['ups_protection_settings']))
        architecture = replace(architecture, ups=replace(architecture.ups, output_stage=stage, rectifier=rectifier,
                                                         protection_domains=(protection,)))
        # Default cooling: PSCAD CoolingLoad_1_1 equivalent (same as the native preset); values from
        # preset_values.cooling_front_end. type 'native_afe' selects the earlier AFE/VFD/CIM5 cooling.
        settings = dict(v.get('cooling_front_end') or {})
        cooling_model = settings.get('type', self.cooling_model)
        return native.tamu_cooling(architecture, cooling_model,
                                   settings if cooling_model == 'pscad_cooling_drive' else None)


def install_datacenter_preset(datacenter: dict) -> None:
    """Route the composite facility's TAMU/ERCOT preset through JSON values."""
    global _VALUES
    if datacenter['facility']['preset_id'] != 'tamu_ercot_2026_centralized_ups':
        return
    _VALUES = MappingProxyType(dict(datacenter['preset_values']))
    functional_facility.TAMUERCOT2026Preset = JSONTAMUERCOTPreset


def original_preset():
    return _ORIGINAL
