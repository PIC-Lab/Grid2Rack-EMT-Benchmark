"""Run JAX-EMT studies: ERCOT/TAMU data-center model and the PNNL IEEE 39-bus system.

Quick use: edit the SETTINGS block below and press Run (jax-emt Python), or use the
command line (options override the settings):
    python -B Run_JAXEMT.py
    python -B Run_JAXEMT.py --scenario ieee39_bus16_fault --no-show
    python -B Run_JAXEMT.py --scenario ieee39_base_datacenter_5sites --no-show
    python -B Run_JAXEMT.py --scenario ieee39_base_datacenter_5sites_forced_oscillation --no-show
    python -B Run_JAXEMT.py --scenario ieee39_bus16_forced_oscillation --no-show
    python -B Run_JAXEMT.py --scenario ercot_voltage_sag ieee39_flat_start --action validate
    python -B Run_JAXEMT.py --case configs/cases/my_case.json --output results/my_run

Each scenario is a case JSON in configs/cases/. The case "schema" selects the runner:
    ercot-case-v1  data center + external network  -> simulation/runner.py
    psse-case-v1   PSS/E network only (IEEE 39, base or 3-IBR) -> simulation/psse_runner.py
    ercot-psse-case-v1  data center + feeder at a PSS/E bus (IEEE 39 bus 16) -> simulation/runner.py
    ercot-psse-multi-case-v1  independent data centers at multiple PSS/E buses -> simulation/runner.py
Results go to results/<case output>/. PSCAD comparisons are separate scripts in validation/.
JAX-EMT is always imported from this repository (../../jax_emt), not from an installed copy.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ============================== SETTINGS (edit here) ==============================
SCENARIO = "ieee39_base_datacenter_5sites"  # one name or a list of names from SCENARIOS below
ACTION = "simulate"                # "simulate", or "validate" (import/assemble/check only, no EMT)

# Optional overrides of the scenario's case JSON (None = keep the case JSON value).
DURATION_S = 15                  # simulation length (s); None uses each case JSON (70 s for both forced cases)
TIME_STEP_S = 5e-5                 # EMT step (s); the cases are qualified at 5e-05
EVENTS = None                      # replace the event list, e.g. ERCOT sag to 0.8 pu:
# EVENTS = [{"time_s": 2.0, "name": "source_sag", "values": {"grid.voltage_scale": 0.8}},
#           {"time_s": 3.0, "name": "source_recovery", "values": {"grid.voltage_scale": 1.0}}]
# IEEE 39 bus-16 fault at 5.0-5.1 s:
# EVENTS = [{"time_s": 5.0, "name": "bus16_fault", "values": {"fault.bus16.closed": [True, True, True]}},
#           {"time_s": 5.1, "name": "bus16_fault_clear", "values": {"fault.bus16.closed": [False, False, False]}}]
# IEEE 39 3-IBR bus-16 fault at the PSS/E reference timing 20.0-20.1 s (also set DURATION_S, e.g. 30.0):
# EVENTS = [{"time_s": 20.0, "name": "bus16_fault", "values": {"fault.bus16.closed": [True, True, True]}},
#           {"time_s": 20.1, "name": "bus16_fault_clear", "values": {"fault.bus16.closed": [False, False, False]}}]

SHOW_PLOTS = True                  # open the plot window at the end (False: only save PNG)
# ================================================================================

SCENARIOS = {
    "ieee39_base_datacenter_5sites_forced_oscillation": {
        "case": "configs/cases/ieee39_base_datacenter_5sites_forced_oscillation_case.json",
        "about": "70-s five-site EMT: 20-s 100-MW request per site, then 50-s synchronous TAMU replay; existing DC parameters; no fault or SSA"},
    "ieee39_base_datacenter_5sites": {
        "case": "configs/cases/ieee39_base_datacenter_5sites_case.json",
        "about": "Base IEEE39 with five independent 100-MW IT data centers at buses 15/16/18/21/23; each replaces 100 MW of native active load"},
    "ieee39_bus16_forced_oscillation": {
        "case": "configs/cases/ieee39_bus16_forced_oscillation_case.json",
        "about": "70-s EMT: 20-s 100-MW hold then 50-s TAMU IT-load replay at IEEE39 bus 16; five-site-matched DC parameters; no fault or SSA prerequisite; 35-kV P/Q meter"},
    "ercot_voltage_sag": {
        "case": "configs/cases/ercot_average_case.json",
        "about": "ERCOT data center on the 138/35/0.48 kV grid; source voltage sag 2-3 s (PSCAD reproduction)"},
    "ercot_demo_69kv": {
        "case": "configs/cases/ercot_demo_69kv_case.json",
        "about": "Same data center on a 69 kV stiff demo grid; 0.5 pu sag 2.0-2.5 s"},
    "ieee39_flat_start": {
        "case": "configs/cases/ieee39_flat_start_case.json",
        "about": "PNNL IEEE 39-bus, no event: operating point must be preserved"},
    "ieee39_bus16_fault": {
        "case": "configs/cases/ieee39_bus16_fault_case.json",
        "about": "PNNL IEEE 39-bus, three-phase fault at bus 16 (0.1 ohm, 0.1 s)"},
    "ieee39_3ibr_flat_start": {
        "case": "configs/cases/ieee39_3ibr_flat_start_case.json",
        "about": "PNNL IEEE 39-bus with 3 WECC IBRs (buses 3001/3003/3005), no event"},
    "ieee39_3ibr_bus16_fault": {
        "case": "configs/cases/ieee39_3ibr_bus16_fault_case.json",
        "about": "PNNL IEEE 39-bus with 3 WECC IBRs, three-phase fault at bus 16 (0.1 ohm, 5.0-5.1 s)"},
    "ieee39_3ibr_datacenter_bus16": {
        "case": "configs/cases/ieee39_3ibr_datacenter_bus16_case.json",
        "about": "ERCOT data center (~100 MW) at bus 16 of the 3-IBR IEEE 39; IT ramp 0.5-1 s; bus-16 fault timing from case events"},
    "ieee39_base_datacenter_bus16": {
        "case": "configs/cases/ieee39_base_datacenter_bus16_case.json",
        "about": "ERCOT data center (~100 MW) at bus 16 of the base IEEE 39; IT ramp 0.5-1 s; bus-16 fault timing from case events"},
}
RUNNERS = {"ercot-case-v1": "simulation.runner", "ercot-psse-case-v1": "simulation.runner",
           "ercot-psse-multi-case-v1": "simulation.runner",
           "psse-case-v1": "simulation.psse_runner"}
ACTIONS = ("simulate", "validate")

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import simulation.bootstrap as bootstrap  # noqa: E402  (JAX-EMT from this repository, not the installed one)


def case_with_overrides(name, case_path, duration, time_step, events):
    """Return the case JSON to run; with overrides, write a derived copy next to the original."""
    if duration is None and time_step is None and events is None:
        return case_path
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if duration is not None:
        case["simulation"]["duration_s"] = float(duration)
    if time_step is not None:
        case["simulation"]["time_step_s"] = float(time_step)
    if events is not None:
        case["events"] = events
        check = case.get("verification", {})
        if check.get("kind") == "fault":
            # Keep the IEEE 39 fault check on the new fault window (first closing, next opening).
            closing = [e for e in events if any(v is True or v == [True] * 3 for v in e["values"].values())]
            opening = [e for e in events if closing and e["time_s"] > closing[0]["time_s"]]
            if closing and opening:
                check["fault_start_s"], check["fault_clear_s"] = closing[0]["time_s"], opening[0]["time_s"]
    case["description"] = case.get("description", "") + " [Run_JAXEMT overrides applied]"
    derived = case_path.with_name(f"_Run_JAXEMT_{name}.json")   # same folder: relative paths stay valid
    derived.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    print(f"Overrides written to {derived.relative_to(ROOT)}")
    return derived


def run_case(case_path, action, show, output=None):
    """Run one case in this process with the runner selected by its schema."""
    case = json.loads(case_path.read_text(encoding="utf-8"))
    schema = case.get("schema")
    if schema not in RUNNERS:
        raise SystemExit(f"{case_path}: unknown case schema {schema!r}; expected one of {sorted(RUNNERS)}")
    argv = ["--case", str(case_path)]
    if action == "validate":
        argv.append("--validate-only")
    elif not show:
        argv.append("--no-show")
    if output is not None:
        argv += ["--output", str(output)]
    import jax_emt
    print("JAX-EMT source: " + str(Path(jax_emt.__file__).resolve().parent), flush=True)
    runner = "ForcedOscillation.Run_Case" if "forced_oscillation" in case else RUNNERS[schema]
    __import__(runner, fromlist=["main"]).main(argv)


def scenario_overrides(case_path, duration, time_step):
    """CLI overrides IDE settings; None in both preserves the case JSON value."""
    return (duration if duration is not None else DURATION_S,
            time_step if time_step is not None else TIME_STEP_S)


def main():
    parser = argparse.ArgumentParser(description="Run JAX-EMT scenarios (see SETTINGS in this file).")
    parser.add_argument("--scenario", nargs="+", choices=tuple(SCENARIOS))
    parser.add_argument("--case", type=Path, help="run this case JSON instead of a named scenario")
    parser.add_argument("--action", choices=ACTIONS)
    parser.add_argument("--duration", type=float, help="override simulation duration (s)")
    parser.add_argument("--time-step", type=float, help="override EMT time step (s)")
    parser.add_argument("--output", type=Path, help="override the case output folder")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args = parser.parse_args()
    action = args.action or ACTION
    duration, time_step = args.duration, args.time_step
    show = SHOW_PLOTS and not args.no_show
    if action not in ACTIONS:
        parser.error(f"ACTION must be one of {ACTIONS}")

    if args.case is not None:
        jobs = [(args.case.stem, args.case.resolve(), "case file " + str(args.case))]
    else:
        names = args.scenario or ([SCENARIO] if isinstance(SCENARIO, str) else list(SCENARIO))
        unknown = [name for name in names if name not in SCENARIOS]
        if unknown:
            parser.error(f"unknown scenario {unknown}; choose from {', '.join(SCENARIOS)}")
        jobs = [(name, ROOT / SCENARIOS[name]["case"], SCENARIOS[name]["about"]) for name in names]

    if len(jobs) > 1:
        # Each scenario gets its own Python process (the data-center runner patches the
        # ERCOT preset class and JAX settings for the whole interpreter).
        for name, _, _ in jobs:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--scenario", name,
                       "--action", action]
            command += ["--duration", str(duration)] if duration is not None else []
            command += ["--time-step", str(time_step)] if time_step is not None else []
            command += [] if show else ["--no-show"]
            print("\n>>> " + " ".join(command), flush=True)
            code = subprocess.call(command, cwd=str(ROOT))
            if code != 0:
                raise SystemExit(f"Scenario {name} failed with exit code {code}")
        print("\nAll scenarios done.")
        return

    name, case_path, about = jobs[0]
    duration, time_step = scenario_overrides(case_path, duration, time_step)
    print(f"=== {name}: {about}\n    case {case_path.relative_to(ROOT) if case_path.is_relative_to(ROOT) else case_path}, "
          f"action {action}", flush=True)
    effective_case = case_with_overrides(name, case_path, duration, time_step, EVENTS)
    scheduled = json.loads(effective_case.read_text(encoding="utf-8")).get("events", [])
    for event in scheduled:
        print(f"    event {event['name']}: {event['time_s']:g} s", flush=True)
    run_case(effective_case, action, show, args.output)


if __name__ == "__main__":
    main()
