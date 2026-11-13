"""Stage and run a PSCAD case: ERCOT/TAMU data-center model or PNNL Enhanced IEEE 39-bus system.

Quick use: edit the SETTINGS block below and press Run (jax-emt Python), or
override any setting on the command line:
    python RUN_PSCAD.py
    python RUN_PSCAD.py --system ieee39_base --event bus16-fault --event-time-s 5
    python RUN_PSCAD.py --system ieee39_5dc --duration-s 10 --event bus16-fault --event-time-s 2
    python RUN_PSCAD.py --system ieee39_5dc_forced_oscillation
    python RUN_PSCAD.py --system ercot_voltage_sag --sag-pu 0.7 --duration-s 6
    python RUN_PSCAD.py --system forced_oscillation
    python RUN_PSCAD.py --system forced_oscillation --duration-s 40 --time-step-us 25
    python RUN_PSCAD.py --dry-run             # check inputs; forced case also stages/audits

Models (folders next to this script):
    ErcotDataCenter/  TestCase_VoltageSag_Average / _Switching / ITLoadFluctuation_Average
                      + AIDataCenter_Model_Library.pslx, ComputingLoad.txt
    PNNL_IEEE39/      Enhanced IEEE 39-Bus System[_1IBR|_3IBRs|_Bus16DC|_5DC]/PSCAD + E-Tran library ETRAN46.pslx
                      (_Bus16DC and _5DC also load AIDataCenter_Model_Library.pslx)
The source folders are never modified: the model folder (without build folders and
backups) is copied to Results/<system>_<event>/stage (replaced on each run), and only
the staged copy is edited (time settings, event timers, sag, E-Tran release time).
After a successful run, Results/<system>_<event>/simulation.png is saved and a plot
window opens. Use --connect to run in an already-open licensed PSCAD session.
The forced_oscillation option delegates to RUN_PSCAD_ForcedOsccilation.py:
70 s by default (100 MW IT request for 20 s, then 50 s of the real load),
the original data-center parameters, and a staged load/timing setup.
Its plots are saved without opening windows; --connect is not supported.
The ieee39_5dc_forced_oscillation option uses the ordinary five-site staged
model for 70 s: 100 MW per site until 20 s, then the first 50 s of the
original IT profile. It reads the shared JAX profile settings and preserves source
controller/capacitor/machine settings and the native 0.5-1-s IT startup ramp;
--dry-run prepares and audits this copy without launching PSCAD.
"""

from __future__ import print_function

# ============================== SETTINGS (edit here) ==============================
SYSTEM = "ieee39_5dc"       # see SYSTEMS below:
                               #   ercot_voltage_sag, ercot_voltage_sag_switching, ercot_it_load_fluctuation,
                               #   ieee39_base, ieee39_1ibr, ieee39_3ibr,
                               #   ieee39_bus16dc (base IEEE 39 + ERCOT data center at bus 16),
                               #   ieee39_5dc (base IEEE 39 + data centers at buses 15, 16, 18, 21, 23),
                               #   ieee39_5dc_forced_oscillation (same five-site parameters; JAX-matched load replay),
                               #   forced_oscillation (same forced case as RUN_PSCAD_ForcedOsccilation.py)
# None uses case defaults: ordinary 5 s/50 us/100 us; single forced JAX JSON (70 s/50 us/1000 us);
# five-site forced uses 70 s and reads step/sampling from its JAX JSON (50 us/200 us).
# Set numeric values here to override the selected case, or use CLI arguments.
DURATION_S = 15              # None: both forced cases use 70 s (20 s constant + 50 s real profile)
TIME_STEP_US = None            # solution step (us); ERCOT switching case needs 1 us
SAMPLE_STEP_US = None          # output sampling step (us)
DC_TI_S = None                # forced only; optional source-value check, never a parameter override

# ERCOT voltage-sag cases: source voltage band (None = as saved in the project)
SAG_PU = 0.7                  # source voltage during the sag, e.g. 0.7
SAG_START_S = 2.0             # e.g. 2.0
SAG_END_S = 3.0               # e.g. 3.0

# IEEE 39 cases
EVENT = "bus16-fault"             # "as-saved", "bus16-fault", "gen-trip", "none"
EVENT_TIME_S = 6.0             # used by bus16-fault / gen-trip (2.0 = JAX-EMT ieee39_base_datacenter_bus16)
FAULT_DURATION_S = 0.1         # used by bus16-fault
RELEASE_TIME_S = None          # E-Tran hold/release time (s) of loads and machines; None = as saved (1.0)
# IEEE 39 data-center cases (diagnostics; edits the staged copy, never the source project)
# For ieee39_5dc the shared constant applies to every data center.
NO_LL = None                   # data-center NO_LL current-scaler count; None = as saved (100)
# ================================================================================

import argparse
import hashlib
import json
import os
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"
FIVE_SITE_FORCED_CASE = ROOT.parent / "JAX-EMT-main/examples/datacenter_ercot/configs/cases/ieee39_base_datacenter_5sites_forced_oscillation_case.json"
ERCOT_FOLDER = Path("ErcotDataCenter")
IEEE39_FOLDER = Path("PNNL_IEEE39")
# IEEE 39 timed breaker logic (master:tbreakn). PSCAD breaker signal: 1 = open, 0 = closed.
# fault: drives breaker BRK, 0.1 ohm (0.26525 mH) three-phase fault at bus 16, INIT open.
# trip:  drives breaker Trip, generator terminal breaker (generator 32 in the base system), INIT closed.
ERCOT = {"family": "ercot", "folder": ERCOT_FOLDER,
         "library_file": "AIDataCenter_Model_Library.pslx", "library": "AIDataCenter_Model_Library",
         "support": ("ComputingLoad.txt",)}
IEEE39 = {"family": "ieee39", "library_file": "ETRAN46.pslx", "library": "ETRAN", "support": ("ETRAN_GF46.lib",)}
SYSTEMS = {
    "forced_oscillation": {"family": "forced_oscillation"},
    "ercot_voltage_sag": dict(ERCOT, case="TestCase_VoltageSag_Average.pscx",
                              project="TestCase_VoltageSag_Average", sag=True),
    "ercot_voltage_sag_switching": dict(ERCOT, case="TestCase_VoltageSag_Switching.pscx",
                                        project="TestCase_VoltageSag_Switching", sag=True),
    "ercot_it_load_fluctuation": dict(ERCOT, case="TestCase_ITLoadFluctuation_Average.pscx",
                                      project="TestCase_ITLoadFluctuation_Ave", sag=False),
    "ieee39_base": dict(IEEE39, folder=IEEE39_FOLDER / "Enhanced IEEE 39-Bus System" / "PSCAD",
                        case="Orginal.pscx", project="Orginal",
                        timers={"fault": "1268011081", "trip": "923618620"}),
    "ieee39_1ibr": dict(IEEE39, folder=IEEE39_FOLDER / "Enhanced IEEE 39-Bus System_1IBR" / "PSCAD",
                        case="IBR_DM.pscx", project="IBR_DM",
                        timers={"fault": "1268011081", "trip": "1750255191"}),
    "ieee39_3ibr": dict(IEEE39, folder=IEEE39_FOLDER / "Enhanced IEEE 39-Bus System_3IBRs" / "PSCAD",
                        case="3IBR.pscx", project="a3IBR",
                        timers={"fault": "1268011081", "trip": "1750255191"}),
    # Base IEEE 39 with the ERCOT/TAMU data center at bus 16 (345/35 kV + 2 MVA 35/0.48 kV x NO_LL=100).
    # Builds with the E-Tran GF46 toolchain; needs the AIDataCenter library in addition to E-Tran.
    "ieee39_bus16dc": dict(IEEE39, folder=IEEE39_FOLDER / "Enhanced IEEE 39-Bus System_Bus16DC" / "PSCAD",
                           case="Orginal.pscx", project="Orginal",
                           timers={"fault": "1268011081", "trip": "923618620"},
                           extra_libraries=(("AIDataCenter_Model_Library.pslx", "AIDataCenter_Model_Library"),),
                           plot_datacenter=True, no_ll_const="1478788133"),
    "ieee39_5dc": dict(IEEE39, folder=IEEE39_FOLDER / "Enhanced IEEE 39-Bus System_5DC" / "PSCAD",
                       case="Orginal_5DC.pscx", project="Orginal_5DC",
                       timers={"fault": "1268011081", "trip": "923618620"},
                       extra_libraries=(("AIDataCenter_Model_Library.pslx", "AIDataCenter_Model_Library"),),
                       no_ll_const="132835211",
                       # INF suffixes follow the compiled P3 data-center call order.
                       datacenter_sites=((15, ""), (16, "_1"), (18, "_2"), (21, "_3"), (23, "_4"))),
}
SYSTEMS["ieee39_5dc_forced_oscillation"] = dict(
    SYSTEMS["ieee39_5dc"], it_profile_case=FIVE_SITE_FORCED_CASE,
    duration_s=70., output_label="ieee39_5dc_forced_oscillation")
STAGE_IGNORE = ("*.gf46", "*.gf81*", "*.gf95", "*.bakx", "*.bak")   # build output folders and backups
PARKED_S = 1000.0            # a timer time beyond any run = event disabled


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_inventory(stage, project_name):
    files = sorted(stage.rglob(project_name + "*.out"))
    infos = sorted(stage.rglob(project_name + ".inf"))
    return [
        {
            "file": str(path.relative_to(stage)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in infos + files
    ]


def main_page(root):
    main = [d for d in root.iter("Definition") if d.get("name") == "Main"]
    if len(main) != 1:
        raise ValueError("expected one Main page definition")
    return main[0]


def params(element):
    return {p.get("name"): p for p in element.iter("param")}


def set_sag(project, sag_pu, start_s, end_s):
    """ERCOT: edit the source-voltage band (master:range with OL = OH = 1.0) on the Main page."""
    if sag_pu is None and start_s is None and end_s is None:
        return None
    tree = ET.parse(project)
    bands = [e for e in main_page(tree.getroot()).iter() if e.get("defn") == "master:range"
             and params(e).get("OL") is not None and params(e)["OL"].get("value") == "1.0"
             and params(e)["OH"].get("value") == "1.0"]
    if len(bands) != 1:
        raise ValueError("expected one source-voltage band (master:range, OL = OH = 1.0) on Main")
    p = params(bands[0])
    for name, value in (("OM", sag_pu), ("XL", start_s), ("XU", end_s)):
        if value is not None:
            p[name].set("value", f"{value:g}")
    tree.write(project, encoding="utf-8", xml_declaration=True)
    return {"id": bands[0].get("id"), "sag_pu": p["OM"].get("value"),
            "start_s": p["XL"].get("value"), "end_s": p["XU"].get("value")}


def set_event(project, timers, event, event_time_s, fault_duration_s):
    """IEEE 39: edit only the two timed-breaker components; return the applied timer settings."""
    if event == "as-saved":
        return None
    tree = ET.parse(project)
    root = tree.getroot()
    elements = {}
    for key, identifier in timers.items():
        matches = [e for e in root.iter() if e.get("id") == identifier]
        if len(matches) != 1 or matches[0].get("defn") != "master:tbreakn":
            raise ValueError("expected one master:tbreakn with id " + identifier)
        elements[key] = matches[0]
    fault_on, fault_off, trip = PARKED_S, PARKED_S + 0.1, PARKED_S
    if event == "bus16-fault":
        fault_on, fault_off = event_time_s, event_time_s + fault_duration_s
    elif event == "gen-trip":
        trip = event_time_s
    values = {
        "fault": {"NUMS": "2", "INIT": "1", "TO1": f"{fault_on:g} [s]", "TO2": f"{fault_off:g} [s]"},
        "trip": {"NUMS": "1", "INIT": "0", "TO1": f"{trip:g} [s]"},
    }
    for key, settings in values.items():
        for parameter in elements[key].iter("param"):
            if parameter.get("name") in settings:
                parameter.set("value", settings[parameter.get("name")])
    tree.write(project, encoding="utf-8", xml_declaration=True)
    return values


def set_no_ll(project, const_id, value):
    """Data-center case: set the master:const that feeds the NO_LL current scaler; return the edit."""
    if value is None:
        return None
    tree = ET.parse(project)
    matches = [e for e in tree.getroot().iter() if e.get("id") == const_id]
    if len(matches) != 1 or matches[0].get("defn") != "master:const":
        raise ValueError("expected one master:const (NO_LL) with id " + const_id)
    p = params(matches[0])
    old = p["Value"].get("value")
    p["Value"].set("value", f"{value:g}")
    tree.write(project, encoding="utf-8", xml_declaration=True)
    return {"id": const_id, "saved": old, "applied": f"{value:g}"}


def set_release_time(project, release_time_s):
    """IEEE 39: set the E-Tran hold/release time on the Main page; return the edited values.

    Main page, top left:
      master:time-enab T                  -> L_HLD, load hold released at T
      ETRAN Common_Set/Get "Load Control" -> TStart, the same load release on sub-pages
      master:const "Time to Release Machines" -> TMStart; each generator page releases the
        machine at max(TMStart, 0.3 s), its exciter 0.1 s later and its governor 0.2 s later.
    """
    if release_time_s is None:
        return None
    tree = ET.parse(project)
    root = tree.getroot()
    main = main_page(root)
    edited = []
    timers = [e for e in main.iter() if e.get("defn") == "master:time-enab"]
    links = [e for e in main.iter() if e.get("defn") == "master:radiolink"
             and params(e).get("Name") is not None and params(e)["Name"].get("value") == "TMStart"]
    if len(timers) != 1 or len(links) != 1:
        raise ValueError("expected one time-enable (L_HLD) and one TMStart radio link on Main")
    params(timers[0])["T"].set("value", f"{release_time_s:g}")
    edited.append({"component": "master:time-enab", "id": timers[0].get("id"), "T": release_time_s})
    # The constant wired into the TMStart radio link sits on the same row, left of it.
    link_x, link_y = int(links[0].get("x")), int(links[0].get("y"))
    constants = [e for e in main.iter() if e.get("defn") == "master:const"
                 and int(e.get("y")) == link_y and 0 < link_x - int(e.get("x")) <= 108]
    if len(constants) != 1:
        raise ValueError("expected one constant feeding TMStart on Main")
    params(constants[0])["Value"].set("value", f"{release_time_s:g}")
    edited.append({"component": "master:const (TMStart)", "id": constants[0].get("id"), "Value": release_time_s})
    for element in root.iter():
        if element.get("defn") in ("ETRAN:Electranix_Common_Set", "ETRAN:Electranix_Common_Get"):
            p = params(element)
            if p.get("Name") is not None and p["Name"].get("value") == "Load Control" and "TStart" in p:
                p["TStart"].set("value", f"{release_time_s:g}")
                edited.append({"component": element.get("defn"), "id": element.get("id"), "TStart": release_time_s})
    tree.write(project, encoding="utf-8", xml_declaration=True)
    return edited


def saved_settings(project):
    """Time settings, event timers and source-voltage bands as saved in the source project."""
    root = ET.parse(project).getroot()
    settings = {p.get("name"): p.get("value") for p in root.find("paramlist")
                if p.get("name") in ("time_duration", "time_step", "sample_step")}
    timers = {e.get("id"): {p.get("name"): p.get("value") for p in e.iter("param")
                            if p.get("name") in ("NUMS", "INIT", "TO1", "TO2")}
              for e in root.iter() if e.get("defn") == "master:tbreakn"}
    bands = {e.get("id"): {p.get("name"): p.get("value") for p in e.iter("param")
                           if p.get("name") in ("XL", "XU", "OL", "OM", "OH")}
             for e in root.iter() if e.get("defn") == "master:range"}
    return {"settings": settings, "timers": timers, "bands": bands}


def build_event_consumer(events, manifest, manifest_path):
    """Retain bounded build diagnostics while PSCAD's synchronous run blocks."""

    while True:
        event = yield
        events.append({"repr": repr(event), "received_at_s": time.time()})
        del events[:-200]
        manifest["build_events"] = events
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


def run_timing(events, run_started, run_ended):
    """Build and EMTDC times from the timestamped PSCAD build events (wall clock, s)."""
    def first(name, status):
        for event in events:
            text = event["repr"]
            if f"'name': '{name}'" in text and f"'status': '{status}'" in text:
                return event["received_at_s"]
        return None

    build_end = first("Project Builder", "END")
    emtdc_begin = first("EMTDC RUN", "BEGIN")
    emtdc_end = first("EMTDC RUN", "END") or first("Mediator", "END") or first("Workspace", "END") or run_ended
    timing = {"pscad_build_and_run_s": run_ended - run_started}
    if build_end is not None:
        timing["build_compile_s"] = build_end - run_started
    if emtdc_begin is not None:
        timing["emtdc_simulation_s"] = emtdc_end - emtdc_begin
    return timing


def read_channels(stage, project_name):
    """Map PGB names to channel numbers from PSCAD's channel index."""
    infos = list(stage.rglob(project_name + '.inf'))
    if len(infos) != 1:
        raise ValueError('Expected exactly one PSCAD channel index')
    info = infos[0]
    channels = {}
    for line in info.read_text().splitlines():
        match = re.search(r'PGB\((\d+)\).*?Desc="([^"]*)"', line)
        if match:
            channels.setdefault(match[2], int(match[1]))
    return info, channels


SITE_PLOT_CHANNELS = (
    ('Grid voltage (pu)', 'VRMS_Grid'),
    ('Protected-load voltage (pu)', 'VRMS_Load'),
    ('IT power (MW/unit)', 'P_Load'),
    ('Cooling power (kW/unit)', 'P_VFD'),
    ('Facility active power (pu)', 'P_Grid'),
    ('Facility reactive power (pu)', 'Q_Grid'),
    ('DC link (kV)', 'V_DC_Link'),
    ('SOC (pu)', 'SOC'),
)


def plot_groups(family, datacenter=False, datacenter_sites=()):
    if family == "ieee39" and datacenter_sites:
        return plot_groups("ieee39") + tuple(
            (label, [name + suffix for _, suffix in datacenter_sites])
            for label, name in SITE_PLOT_CHANNELS
        )
    if family == "ieee39" and datacenter:
        # Grid machines plus the data-center channels (groups without recorded channels are dropped).
        return plot_groups("ieee39") + plot_groups("ercot")
    if family == "ercot":
        return (
            ('Voltage (pu)', ['VRMS_Grid', 'VRMS_Load']),
            ('Current (pu)', ['IRMS_Bypass', 'IRMS_Rect']),
            ('Command', ['En_Bypass', 'DBlk_Rect', 'Charging1_Discharging0']),
            ('Power (MW)', ['P_Bypass', 'P_Rect', 'P_Load']),
            ('Power (MW)', ['P_Inv', 'P_DCDC', 'P_battery']),
            ('Power (pu)', ['P_Grid', 'Q_Grid']),
            ('DC link (kV)', ['V_DC_Link']),
            ('SOC (pu)', ['SOC']),
        )
    buses = range(30, 40)
    return (
        ('Voltage (pu)', ['V%d' % b for b in buses]),
        ('Speed', ['SPD%d' % b for b in buses]),
        ('Active power', ['P%d' % b for b in buses]),
        ('Reactive power', ['Q%d' % b for b in buses]),
    )


def plot_results(stage, project_name, family, destination, show=True, datacenter=False,
                 datacenter_sites=()):
    """Plot the main channels from PSCAD's indexed, ten-channel OUT shards."""
    import numpy as np
    import matplotlib.pyplot as plt

    info, channels = read_channels(stage, project_name)
    groups = [(label, [n for n in names if n in channels])
              for label, names in plot_groups(family, datacenter, datacenter_sites)]
    groups = [group for group in groups if group[1]]
    if not groups:
        raise ValueError('None of the plot channels were recorded')
    cache = {}
    site_labels = {name + suffix: 'Bus %d' % bus
                   for bus, suffix in datacenter_sites for _, name in SITE_PLOT_CHANNELS}
    figure, axes = plt.subplots(len(groups), 1, figsize=(13, 2.2 * len(groups) + 1), sharex=True,
                                squeeze=False)
    axes = axes[:, 0]
    try:
        for axis, (label, names) in zip(axes, groups):
            for name in names:
                number = channels[name]
                shard = (number - 1) // 10 + 1
                if shard not in cache:
                    cache[shard] = np.loadtxt(
                        info.parent / (project_name + f'_{shard:02d}.out'), ndmin=2)
                data = cache[shard]
                axis.plot(data[:, 0], data[:, (number - 1) % 10 + 1],
                          label=site_labels.get(name, name), linewidth=1,
                          drawstyle='steps-post' if label == 'Command' else 'default')
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.4)
            axis.legend(loc='lower left', ncol=5, fontsize=7)
            axis.ticklabel_format(axis='y', style='plain', useOffset=False)
        axes[-1].set_xlabel('Time (s)')
        figure.suptitle('PSCAD: ' + project_name)
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(destination, dpi=180)
        print('Plot saved: ' + str(destination), flush=True)
        if show:
            plt.show(block=True)
    finally:
        plt.close(figure)


def forced_oscillation_command(arguments):
    """Reuse the forced runner's defaults, original-model audit and result policy."""
    python = Path(sys.executable)
    try:
        import mhi.pscad
        import numpy
        import matplotlib
    except ImportError:
        environment_python = Path.home() / "Software/Miniconda/envs/jax-emt/python.exe"
        if environment_python.is_file():
            python = environment_python
    command = [str(python), "-B", str(ROOT / "RUN_PSCAD_ForcedOsccilation.py"),
               "--pscad-version", arguments.pscad_version]
    # An omitted option inherits the dedicated runner's current setting. In
    # particular, the ordinary 5-s default must not shorten the 70-s forced case.
    for flag, value in (("--duration-s", arguments.duration_s),
                        ("--time-step-us", arguments.time_step_us),
                        ("--sample-step-us", arguments.sample_step_us),
                        ("--dc-ti-s", arguments.dc_ti_s),
                        ("--output", arguments.output)):
        if value is not None:
            command.extend((flag, str(value)))
    for flag, enabled in (("--dry-run", arguments.dry_run),
                          ("--run-prepared", arguments.run_prepared),
                          ("--analyze-only", arguments.analyze_only)):
        if enabled:
            command.append(flag)
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run a PSCAD case (ERCOT data center or PNNL IEEE 39-bus) and save its outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--system", choices=tuple(SYSTEMS), default=SYSTEM)
    parser.add_argument("--time-step-us", type=float, default=TIME_STEP_US,
                        help="solution step in microseconds; None uses the selected case default")
    parser.add_argument("--sample-step-us", type=float, default=SAMPLE_STEP_US,
                        help="output sampling step in microseconds; defaults 100 ordinary, 1000 single forced, JAX JSON for five-site forced")
    parser.add_argument("--duration-s", type=float, default=DURATION_S,
                        help="duration in seconds; defaults 5 ordinary, JAX JSON for single forced (70), 70 five-site forced")
    parser.add_argument("--dc-ti-s", type=float, default=DC_TI_S,
                        help="forced oscillation: verify the original rectifier integral time; a different value is rejected")
    parser.add_argument("--sag-pu", type=float, default=SAG_PU, help="ERCOT: source voltage during the sag")
    parser.add_argument("--sag-start-s", type=float, default=SAG_START_S, help="ERCOT: sag start time")
    parser.add_argument("--sag-end-s", type=float, default=SAG_END_S, help="ERCOT: sag end time")
    parser.add_argument(
        "--event",
        choices=("as-saved", "bus16-fault", "gen-trip", "none"),
        default=EVENT,
        help="IEEE 39: as-saved keeps the project's timers; bus16-fault: 3-phase fault at bus 16; "
             "gen-trip: open the generator Trip breaker; none: flat run",
    )
    parser.add_argument("--event-time-s", type=float, default=EVENT_TIME_S)
    parser.add_argument("--fault-duration-s", type=float, default=FAULT_DURATION_S)
    parser.add_argument("--release-time-s", type=float, default=RELEASE_TIME_S,
                        help="IEEE 39: E-Tran load/machine release time in s (default: as saved, 1.0)")
    parser.add_argument("--no-ll", type=float, default=NO_LL,
                        help="IEEE 39 data centers: NO_LL current-scaler count per site (default: as saved, 100)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output directory; forced case keeps its existing runner output, others use Results/<system>_<event>; replaced on each run")
    parser.add_argument("--pscad-version", default="5.0.2")
    parser.add_argument("--project-suffix", default="",
                        help="isolate the staged case project name when using an existing session")
    parser.add_argument("--compiler-bin", type=Path, default=None,
                        help="Fortran runtime DLL folder to put first on PATH "
                             "(ERCOT default: %%GFORTRAN81%%\\x86\\bin when set)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="check inputs without PSCAD; forced case also replaces/prepares/audits its staged run")
    mode.add_argument("--run-prepared", action="store_true",
                      help="forced oscillation: execute an existing unexecuted prepared case")
    mode.add_argument("--analyze-only", action="store_true",
                      help="forced oscillation: regenerate completed-run reports without simulation")
    parser.add_argument("--no-show", action="store_true",
                        help="save the plot without opening its window")
    parser.add_argument(
        "--connect",
        action="store_true",
        help="use the already-running licensed PSCAD session",
    )
    arguments = parser.parse_args()
    system = SYSTEMS[arguments.system]
    family = system["family"]
    provided_flags = {value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")}
    if family == "forced_oscillation":
        unsupported = provided_flags.intersection({
            "--event", "--event-time-s", "--fault-duration-s", "--sag-pu",
            "--sag-start-s", "--sag-end-s", "--release-time-s", "--no-ll",
            "--connect", "--project-suffix", "--compiler-bin"})
        for flag, value in (("--connect", arguments.connect),
                            ("--project-suffix", arguments.project_suffix),
                            ("--compiler-bin", arguments.compiler_bin),
                            ("--release-time-s", arguments.release_time_s),
                            ("--no-ll", arguments.no_ll)):
            if value is not None and value is not False and value != "":
                unsupported.add(flag)
        if unsupported:
            parser.error("forced_oscillation uses its dedicated load/event/startup policy; unsupported options: "
                         + ", ".join(sorted(unsupported)))
        command = forced_oscillation_command(arguments)
        print("Running forced-oscillation workflow: " + str(ROOT / "RUN_PSCAD_ForcedOsccilation.py"), flush=True)
        return subprocess.call(command)
    if arguments.run_prepared or arguments.analyze_only or "--dc-ti-s" in provided_flags:
        parser.error("--run-prepared, --analyze-only and --dc-ti-s require --system forced_oscillation")
    profile_settings = None
    if system.get("it_profile_case"):
        from ForcedOscillation.Stage_5DC_Profile import load_settings
        profile_settings = load_settings(system["it_profile_case"])
        unsupported = provided_flags.intersection({
            "--event-time-s", "--fault-duration-s", "--release-time-s", "--no-ll",
            "--sag-pu", "--sag-start-s", "--sag-end-s"})
        if "--event" in provided_flags and arguments.event != "none":
            unsupported.add("--event")
        if arguments.release_time_s is not None or arguments.no_ll is not None:
            unsupported.add("--release-time-s/--no-ll")
        if arguments.dc_ti_s is not None:
            unsupported.add("DC_TI_S")
        if unsupported:
            parser.error("The five-site forced case preserves its model parameters and has no fault; "
                         "unsupported overrides: " + ", ".join(sorted(unsupported)))
        arguments.event = "none"
    defaults = (("duration_s", 5.), ("time_step_us", 50.), ("sample_step_us", 100.))
    for name, default in defaults:
        if getattr(arguments, name) is None:
            case_default = profile_settings[name] if profile_settings is not None else default
            setattr(arguments, name, system.get(name, case_default))
    if profile_settings is not None:
        profile_settings = dict(profile_settings, duration_s=arguments.duration_s,
                                time_step_us=arguments.time_step_us,
                                sample_step_us=arguments.sample_step_us)
    project_name = system["project"]
    if arguments.project_suffix and not all(c.isalnum() or c == '_' for c in arguments.project_suffix):
        parser.error("project suffix must contain only letters, digits, or underscores")
    if len(project_name) + len(arguments.project_suffix) > 30:
        parser.error("PSCAD project names including suffix must be at most 30 characters")
    # Automation-launched EMTDC may not execute gf81vars.bat; the ERCOT build needs the x86
    # GFortran 8.1 runtime DLLs on PATH, ahead of Conda DLL folders.
    compiler_bin = arguments.compiler_bin
    if compiler_bin is None and family == "ercot" and os.environ.get("GFORTRAN81"):
        compiler_bin = Path(os.environ["GFORTRAN81"]) / "x86" / "bin"
    if compiler_bin is not None:
        compiler_bin = compiler_bin.resolve()
        if not compiler_bin.is_dir():
            parser.error("compiler runtime directory not found: " + str(compiler_bin))
        os.environ["PATH"] = str(compiler_bin) + os.pathsep + os.environ.get("PATH", "")
    values = [arguments.time_step_us, arguments.sample_step_us, arguments.duration_s,
              arguments.event_time_s, arguments.fault_duration_s]
    values += [v for v in (arguments.sag_pu, arguments.sag_start_s, arguments.sag_end_s,
                           arguments.release_time_s, arguments.no_ll) if v is not None]
    if not all(isfinite(value) for value in values):
        parser.error("time and event settings must be finite")
    if arguments.time_step_us <= 0.0 or arguments.sample_step_us < arguments.time_step_us:
        parser.error("time steps must be positive and sample step must be >= solution step")
    if arguments.duration_s <= 0.0:
        parser.error("duration must be positive")

    sag_requested = any(v is not None for v in (arguments.sag_pu, arguments.sag_start_s, arguments.sag_end_s))
    if family == "ercot":
        if sag_requested and not system["sag"]:
            parser.error(f"{arguments.system} has no source-voltage sag to edit")
        if arguments.event != "as-saved" or arguments.release_time_s is not None:
            print("Note: EVENT and RELEASE_TIME_S apply to IEEE 39 systems only; ignored.", flush=True)
        run_label = arguments.system
    else:
        if sag_requested:
            print("Note: SAG_* settings apply to ERCOT voltage-sag systems only; ignored.", flush=True)
        if arguments.event in ("bus16-fault", "gen-trip") and not (
                0.0 < arguments.event_time_s < arguments.duration_s):
            parser.error("event time must lie inside the simulation")
        if arguments.fault_duration_s <= 0.0:
            parser.error("fault duration must be positive")
        run_label = system.get("output_label", arguments.system + "_" + arguments.event.replace("-", "_"))

    source_root = (ROOT / system["folder"]).resolve()
    extra_libraries = tuple(system.get("extra_libraries", ()))   # (file, project name) pairs
    required = ((system["case"], system["library_file"]) + tuple(f for f, _ in extra_libraries)
                + tuple(system["support"]))
    for name in required:
        if not (source_root / name).is_file():
            parser.error("required model file not found: " + str(source_root / name))
    saved = saved_settings(source_root / system["case"])
    output_root = (arguments.output or (RESULTS / run_label)).resolve()
    if output_root == RESULTS.resolve() or output_root.is_relative_to(source_root):
        parser.error("Choose a run subfolder outside the source model and below the Results collection")
    settings = {
        "system": arguments.system, "case": system["case"], "project": project_name,
        "library": system["library"], "extra_libraries": [name for _, name in extra_libraries],
        "source_root": str(source_root),
        "output": str(output_root), "pscad_version": arguments.pscad_version,
        "time_step_us": arguments.time_step_us,
        "sample_step_us": arguments.sample_step_us,
        "duration_s": arguments.duration_s,
        "saved_in_project": saved,
        "compiler_runtime_bin": str(compiler_bin) if compiler_bin else None,
    }
    if system.get("datacenter_sites"):
        settings["datacenter_sites"] = [
            {"bus": bus, "channel_suffix": suffix} for bus, suffix in system["datacenter_sites"]
        ]
    if profile_settings is not None:
        settings["it_load_profile"] = profile_settings
    if family == "ercot":
        settings.update(sag_pu=arguments.sag_pu, sag_start_s=arguments.sag_start_s, sag_end_s=arguments.sag_end_s)
    else:
        settings.update(event=arguments.event, event_time_s=arguments.event_time_s,
                        fault_duration_s=arguments.fault_duration_s, release_time_s=arguments.release_time_s,
                        no_ll=arguments.no_ll)
    # Saved-project details stay in execution_manifest.json; print only the run settings.
    print(json.dumps({k: v for k, v in settings.items()
                      if k not in ("saved_in_project", "compiler_runtime_bin", "it_load_profile")}, indent=2), flush=True)
    if profile_settings is not None:
        schedule = profile_settings["profile_settings"]
        print(f"Five-site IT replay: {schedule['before_profile_power_mw']:g} MW per site until "
              f"{schedule['profile_start_s']:g} s, then synchronous direct-MW TAMU replay; "
              "the original startup ramp and equipment parameters are retained.", flush=True)
        replay_seconds = max(0., arguments.duration_s - schedule["profile_start_s"])
        source_seconds = profile_settings["source_time_range_s"][1]
        print(f"Run length: {arguments.duration_s:g} s; original profile used for "
              f"{min(replay_seconds, source_seconds):g} s after the hold.", flush=True)
    if arguments.dry_run and profile_settings is None:
        print("Input files verified. PSCAD was not launched.")
        return 0
    # Check the selected interpreter before copying model files or starting PSCAD.
    try:
        if not arguments.dry_run:
            import mhi.pscad
            import numpy
            import matplotlib
    except ImportError as exc:
        environment_python = (
            Path.home() / 'Software' / 'Miniconda' / 'envs' / 'jax-emt' / 'python.exe'
        )
        if (environment_python.is_file()
                and environment_python.resolve() != Path(sys.executable).resolve()):
            print(
                'Selected Python lacks simulation dependencies; restarting with: '
                + str(environment_python), flush=True,
            )
            return subprocess.call([
                str(environment_python), '-B', str(Path(__file__).resolve()),
                *sys.argv[1:],
            ])
        parser.error(
            f"Missing simulation dependency: {exc}.\n"
            f"Current Python: {sys.executable}\n"
            "In VS Code, choose Python: Select Interpreter, then select:\n"
            r"C:\Users\Picstudent\Software\Miniconda\envs\jax-emt\python.exe"
            "\nThen click Run Python File again."
        )
    # Each run replaces the previous one in the same folder, so no stale outputs remain.
    # Only folders under Results/ are cleared automatically.
    if output_root.exists() and any(output_root.iterdir()):
        if not output_root.is_relative_to(RESULTS.resolve()):
            parser.error("output directory outside Results/ must be empty; choose a new --output path")
        print("Replacing previous results in: " + str(output_root), flush=True)
        shutil.rmtree(str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    stage = output_root / "stage"
    # Copy the whole model folder so every relative resource (.lib, .txt, runtime folders) resolves.
    shutil.copytree(str(source_root), str(stage), ignore=shutil.ignore_patterns(*STAGE_IGNORE))
    source_manifest = [
        {
            "file": name,
            "source_sha256": sha256(source_root / name),
            "staged_sha256": sha256(stage / name),
            "bytes": (source_root / name).stat().st_size,
        }
        for name in required
    ]
    manifest = {
        "schema": "pscad-execution-v1",
        "status": "started",
        "case": system["case"],          # read by the PSCAD/JAX-EMT comparison scripts
        "settings": settings,
        "source_files": source_manifest,
        "staged_files": sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()),
        "outputs": [],
        "build_events": [],
    }
    manifest_path = output_root / "execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    case_name = project_name + arguments.project_suffix
    staged_case = stage / system["case"]
    if arguments.project_suffix:
        # Namespace only the staged case copy; the library keeps its name.
        content = staged_case.read_text(encoding="utf-8-sig")
        content = content.replace('name="%s"' % project_name, 'name="%s"' % case_name)
        content = content.replace('defn="%s:' % project_name, 'defn="%s:' % case_name)
        renamed = stage / (case_name + ".pscx")
        renamed.write_text(content, encoding="utf-8")
        staged_case.unlink()
        staged_case = renamed
    manifest["staged_project_names"] = {"library": system["library"], "case": case_name}
    if family == "ercot":
        manifest["sag_edit"] = set_sag(staged_case, arguments.sag_pu, arguments.sag_start_s, arguments.sag_end_s)
    else:
        manifest["event_timers"] = set_event(staged_case, system["timers"], arguments.event,
                                             arguments.event_time_s, arguments.fault_duration_s)
        manifest["release_time_edits"] = set_release_time(staged_case, arguments.release_time_s)
        if arguments.no_ll is not None:
            if "no_ll_const" not in system:
                parser.error(arguments.system + " has no NO_LL current scaler to edit")
            manifest["no_ll_edit"] = set_no_ll(staged_case, system["no_ll_const"], arguments.no_ll)
    if profile_settings is not None:
        from ForcedOscillation.Stage_5DC_Profile import stage_profile
        tree = ET.parse(staged_case)
        project_parameters = {item.get("name"): item for item in tree.getroot().find("paramlist")}
        for name, value in (("time_duration", arguments.duration_s),
                            ("time_step", arguments.time_step_us),
                            ("sample_step", arguments.sample_step_us)):
            project_parameters[name].set("value", f"{value:g}")
        tree.write(staged_case, encoding="utf-8", xml_declaration=True)
        manifest["it_load_profile"] = stage_profile(staged_case, stage, profile_settings)
        manifest["staged_files"] = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
    manifest["staged_case_sha256"] = sha256(staged_case)
    if arguments.dry_run:
        manifest["status"] = "prepared"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print("Five-site IT replay staged and audited. PSCAD was not launched.")
        print("Execution manifest: " + str(manifest_path))
        return 0
    app = None
    case = None
    library = None
    loaded_extra = []          # extra libraries this script loaded into a connected session
    owns_app = not arguments.connect
    try:
        import mhi.pscad

        libraries = [(system["library_file"], system["library"])] + list(extra_libraries)
        if arguments.connect:
            app = mhi.pscad.connect(timeout=30)
            existing = {project["name"] for project in app.projects()}
            if case_name in existing:
                raise RuntimeError(
                    "connected PSCAD session already contains project " + case_name
                    + "; use --project-suffix"
                )
            already = [name for _, name in libraries if name in existing]
            if already:
                manifest["library_source"] = "already loaded in the connected session: " + ", ".join(already)
            staged_projects = [str(stage / f) for f, name in libraries if name not in existing]
            staged_projects.append(str(staged_case))
            loaded_extra = [name for _, name in extra_libraries if name not in existing]
            app.load(*staged_projects)
        else:
            staged_projects = [str(stage / f) for f, _ in libraries] + [str(staged_case)]
            app = mhi.pscad.launch(
                version=arguments.pscad_version,
                x64=True,
                minimize=True,
                silence=True,
                timeout=30,
                load_user_profile=True,
                load=staged_projects,
            )
        library = app.project(system["library"]) if owns_app else None
        case = app.project(case_name)
        case.parameters(
            time_step=str(arguments.time_step_us),
            sample_step=str(arguments.sample_step_us),
            time_duration=str(arguments.duration_s),
            PlotType="OUT",
            StartType=0,
        )
        case.save()
        events = []
        consumer = build_event_consumer(events, manifest, manifest_path)
        next(consumer)
        run_started = time.time()
        case.run(consumer)
        manifest["timing_s"] = run_timing(events, run_started, time.time())
        outputs = output_inventory(stage, case.name)
        messages = [
            {
                "text": message.text,
                "label": message.label,
                "status": message.status,
                "scope": message.scope,
            }
            for message in case.messages()
        ]
        manifest["messages"] = messages
        if not all(any(item["file"].endswith(ext) and item["bytes"] > 0
                       for item in outputs) for ext in (".inf", ".out")):
            raise RuntimeError("PSCAD run returned without nonempty .inf and .out files")
        manifest["status"] = "completed"
        manifest["pscad_version"] = str(app.version)
        manifest["outputs"] = outputs
    except BaseException as exc:
        manifest["status"] = "blocked"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
    finally:
        if app is not None and owns_app:
            try:
                app.quit()
            except Exception:
                pass
        elif app is not None:
            projects = [case, library]
            for name in loaded_extra:
                try:
                    projects.append(app.project(name))
                except Exception:
                    pass
            for project in projects:
                if project is None:
                    continue
                try:
                    project.unload()
                except Exception:
                    pass
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Status: {manifest['status']}" + (f" ({manifest.get('error')})" if manifest["status"] != "completed" else ""))
    timing = manifest.get("timing_s", {})
    if timing:
        print(f"CPU time (wall clock): EMTDC simulation {timing.get('emtdc_simulation_s', float('nan')):.2f} s, "
              f"build/compile {timing.get('build_compile_s', float('nan')):.2f} s, "
              f"total PSCAD {timing['pscad_build_and_run_s']:.2f} s "
              f"for {arguments.duration_s:g} s simulated at {arguments.time_step_us:g} us")
    print("Execution manifest: " + str(manifest_path))
    if manifest["status"] == "completed":
        print("PSCAD .out/.inf results saved under: " + str(stage))
        try:
            plot_path = output_root / 'simulation.png'
            plot_results(stage, case_name, family, plot_path, show=not arguments.no_show,
                         datacenter=system.get("plot_datacenter", False),
                         datacenter_sites=system.get("datacenter_sites", ()))
            manifest['plot_file'] = str(plot_path)
        except Exception as exc:
            manifest['plot_error'] = str(exc)
            print('Simulation completed, but plotting failed: ' + str(exc), file=sys.stderr)
            return 2
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
