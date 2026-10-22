#
# Copyright (c) 2023 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""AERIS failure tree, evaluated by Monte Carlo.

Wraps the compiled ``aerisboom`` explosion/BLEVE/battery limit-state functions.

Each tagged component carries a set of parameters (volume, burst pressure, ...)
on ``component.uq_parameters`` as plain floats.  Their **baseline** values come
from the config's ``[Parameters.<marker>]`` section; their **uncertainty** is
declared exactly like the rest of TITAN's UQ -- in the campaign yaml, as
``<field>__<marker>`` parameters bound to distributions -- and the ``UQMapper``
overwrites the baseline once per Monte Carlo sample before the trajectory runs
(see ``Uncertainty/mappings.py`` and ``UQ_parameter_handbook.md``).

This engine therefore does **not** sample: it reads whatever concrete values are
on the component when the run starts, holds them fixed, and evaluates the limit
state ``g`` every iteration at the component's live temperature.  The first
iteration where ``g <= 0`` is that component's failure: it is latched, flagged on
the TITAN component, and ``Fragmentation`` separates it from its assembly on the
same iteration.  The fragment's different mass/inertia/ballistic coefficient then
pulls its trajectory away from the intact case, and the failure probability falls
out of the campaign as the fraction of samples that failed (``failure_summary``).

Imprecise (p-box) parameters are handled entirely by the UQ layer, not here: a
``pbox_normal`` distribution in the yaml makes ``MC_wrapper`` run one sub-campaign
per focal endpoint, and this engine just sees the concrete value it was handed.

Known-inert branches (verified against the compiled extension, not TITAN bugs):
``battery`` never reaches ``g <= 0`` because ``CheckExplosionTriggersBatteryLSF``
ignores the temperature it is handed, and ``bleve`` needs a wall stress that
TITAN's structural solve does not currently produce.  ``burst`` is the only mode
that responds to live reentry temperature.

The heavy ``aerisboom`` extension is imported lazily (:func:`_load_api`) so that
TITAN remains importable when the failure tree is disabled.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

# aerisboom API handle, populated on first use by _load_api().
_api = None
# same as above, needed for correct handling of CreateTank()
_DramaPrimitive = None

def _load_api():
    """Import and return the compiled ``aerisboom`` API, with a helpful error.

    The extension ships in ``aeris-boom/python`` at the repository root and is
    built for a specific CPython version, so this can fail if TITAN is run under
    a different interpreter.
    """
    global _api, _DramaPrimitive

    if _api is not None:
        return _api, _DramaPrimitive

    _here = os.path.dirname(os.path.abspath(__file__))
    boom_root = os.path.join(_here, "..", "aeris-boom")
    boom_python = os.path.join(boom_root, "pyaeris_boum")
    if not os.path.isdir(boom_python):
        boom_python = os.path.join(boom_root, "aeris-boom", "python")
    if boom_python not in sys.path:
        sys.path.insert(0, boom_python)

    try:
        from aerisboom import aeboomAPI as api
        from aerisboom import DramaPrimitive
    except ImportError as exc:
        raise ImportError(
            "The AERIS failure tree is enabled but the 'aerisboom' extension "
            "could not be imported from {}. Ensure aeris-boom is built for the "
            "Python interpreter running TITAN. Original error: {}".format(
                os.path.normpath(boom_python), exc
            )
        ) from exc

    _api = api
    _DramaPrimitive = DramaPrimitive
    return _api, _DramaPrimitive


# ---------------------------------------------------------------------------
# Component specifications
# ---------------------------------------------------------------------------
#
# Each spec holds the component's concrete (already-sampled) parameter values as
# floats and a live reference back to the TITAN Component for its temperature.


@dataclass
class WaterTankSpec:
    """One water-tank component: its parameters and live TITAN link."""

    name: str
    component: object  # TITAN Component, read for live temperature
    volume: float
    initialTemperature: float
    nominalTemperature: float
    nominalBurstPressure: float
    nominalUltimateTensileStrength: float
    liqVolume: float
    liqMass: float
    gasMass: float

    def lsf_params(self) -> "list[float]":
        """Values in the argument order expected by the burst LSF."""
        return [
            self.volume,
            self.initialTemperature,
            self.nominalBurstPressure,
            self.nominalTemperature,
            self.nominalUltimateTensileStrength,
            self.liqVolume,
            self.liqMass,
            self.gasMass,
        ]


@dataclass
class FuelTankSpec:
    """One fuel-tank component: its parameters and live TITAN link."""

    name: str
    component: object
    volume: float
    initialTemperature: float
    nominalTemperature: float
    nominalBurstPressure: float
    nominalUltimateTensileStrength: float
    liqVolume: float
    liqMass: float
    propMass: float
    gasMass: float

    def lsf_params(self) -> "list[float]":
        """Values in the argument order expected by the burst/BLEVE LSFs.

        propMass is not consumed by the LSFs but is kept on the spec for TITAN.
        """
        return [
            self.volume,
            self.initialTemperature,
            self.nominalBurstPressure,
            self.nominalTemperature,
            self.nominalUltimateTensileStrength,
            self.liqVolume,
            self.liqMass,
            self.gasMass,
        ]


@dataclass
class BatterySpec:
    """One battery component: its parameters and live TITAN link."""

    name: str
    component: object
    volume: float
    initialTemperature: float
    initialACR: float
    ventDistanceFromCeiling: float
    nominalCapacity: float

    def lsf_params(self) -> "list[float]":
        """Values in the argument order expected by the battery LSF."""
        return [
            self.volume,
            self.initialTemperature,
            self.initialACR,
            self.ventDistanceFromCeiling,
            self.nominalCapacity,
        ]


#: Spec class and failure modes per component type.
_SPEC_BY_TYPE = {
    "watertank": (WaterTankSpec, ("burst",)),
    "fueltank": (FuelTankSpec, ("burst", "bleve")),
    "battery": (BatterySpec, ("failure",)),
}


def _build_spec(component):
    """Build the appropriate ``*Spec`` from a tagged TITAN component.

    ``component.uq_parameters`` has been populated with baseline floats by the
    config reader and possibly overwritten per-sample by the UQMapper; it
    contains exactly the fields the spec dataclass needs.
    """
    spec_cls, _modes = _SPEC_BY_TYPE[component.component_type]
    name = component.marker if component.marker is not None else component.name
    values = {k: float(v) for k, v in component.uq_parameters.items()}
    return spec_cls(name=name, component=component, **values)


# ---------------------------------------------------------------------------
# Limit-state functions (thin wrappers over aerisboom)
# ---------------------------------------------------------------------------
#
# Each wrapper builds a throwaway aerisboom tank/battery at the component's
# values and asks for the limit-state value at the live wall temperature.
# g > 0 is safe, g <= 0 is failed.


def _eval_water_tank_burst_lsf(x, temperature, mass):
    """x = [volume, initialTemp, nominalBurstPressure, nominalTemp, nominalUTS,
    liqVolume, liqMass, gasMass]"""
    api, DramaPrimitive = _load_api()
    (volume, initial_temp, nominal_burst_pressure, nominal_temp, nominal_uts,
     liq_volume, liq_mass, gas_mass) = x

    dummy_shape=DramaPrimitive.SPHERE
    dummy_params= { "radius": "0.5", "thickness": "0.1"}
    dummy_liner_mat = ""
    dummy_cog=[0.0,0.0,0.0]
    dummy_yaw=0.0
    dummy_pitch=0.0
    dummy_roll=0.0
  
    tank_id = api.CreateTank(
        dummy_shape, dummy_params, mass, volume, temperature, nominal_burst_pressure, nominal_temp,
        "drama-A316", "water", liq_mass, "oxygen", gas_mass, dummy_liner_mat, dummy_cog, dummy_yaw, dummy_pitch, dummy_roll,
    )

    trigger = api.CheckExplosionTriggersTankBurstLSF(tank_id, temperature)
    api.RemoveComponent(tank_id)
    return trigger


def _eval_fuel_tank_burst_lsf(x, temperature, mass):
    """x = [volume, initialTemp, nominalBurstPressure, nominalTemp, nominalUTS,
    liqVolume, liqMass, gasMass]"""
    api, DramaPrimitive = _load_api()
    (volume, initial_temp, nominal_burst_pressure, nominal_temp, nominal_uts,
     liq_volume, liq_mass, gas_mass) = x

    dummy_shape=DramaPrimitive.SPHERE
    dummy_params= { "radius": "0.5", "thickness": "0.1"}
    dummy_liner_mat = ""
    dummy_cog=[0.0,0.0,0.0]
    dummy_yaw=0.0
    dummy_pitch=0.0
    dummy_roll=0.0

    tank_id = api.CreateTank(
        dummy_shape, dummy_params, mass, volume, temperature, nominal_burst_pressure, nominal_temp,
        "drama-A316", "hydrazine", liq_mass, "helium", gas_mass, dummy_liner_mat, dummy_cog, dummy_yaw, dummy_pitch, dummy_roll,
    )

    #tank_id = api.CreateTank(
    #    volume, temperature, nominal_burst_pressure, nominal_temp, nominal_uts,
    #    "drama-A316", "hydrazine", liq_volume, liq_mass, "helium", gas_mass,
    #)
    trigger = api.CheckExplosionTriggersTankBurstLSF(tank_id, temperature)
    api.RemoveComponent(tank_id)
    return trigger


def _eval_fuel_tank_bleve_lsf(x, temperature, stress, mass):
    """x = [volume, initialTemp, nominalBurstPressure, nominalTemp, nominalUTS,
    liqVolume, liqMass, gasMass]"""
    api, DramaPrimitive = _load_api()
    (volume, initial_temp, nominal_burst_pressure, nominal_temp, nominal_uts,
     liq_volume, liq_mass, gas_mass) = x

    dummy_shape=DramaPrimitive.SPHERE
    dummy_params= { "radius": "0.5", "thickness": "0.1"}
    dummy_liner_mat = ""
    dummy_cog=[0.0,0.0,0.0]
    dummy_yaw=0.0
    dummy_pitch=0.0
    dummy_roll=0.0

    tank_id = api.CreateTank(
        dummy_shape, dummy_params, mass, volume, temperature, nominal_burst_pressure, nominal_temp,
        "drama-A316", "hydrazine", liq_mass, "helium", gas_mass, dummy_liner_mat, dummy_cog, dummy_yaw, dummy_pitch, dummy_roll,
    )

    trigger = api.CheckExplosionTriggersTankBleveLSF(tank_id, temperature, stress)
    api.RemoveComponent(tank_id)
    return trigger


def _eval_battery_lsf(x, temperature, mass, time, soc):
    """x = [volume, initialTemp, initialACR, ventDistanceFromCeiling,
    nominalCapacity]"""
    api, DramaPrimitive = _load_api()
    (volume, initial_temp, initialACR, ventDistanceFromCeiling, nominalCapacity) = x

    geometry = 0
    chemistry = 0
    nrCells = 20

    dummy_shape=DramaPrimitive.SPHERE
    dummy_params= { "radius": "0.5", "thickness": "0.1"}
    dummy_cog=[0.0,0.0,0.0]
    dummy_yaw=0.0
    dummy_pitch=0.0
    dummy_roll=0.0

    # devnote: wall material based on educated guess, argument does not appear in RTech docs.

    battery_id = api.CreateBatteryEnvironment(
        dummy_shape, dummy_params, "drama-CFRP", mass, volume, initialACR, ventDistanceFromCeiling, geometry, chemistry,
        nominalCapacity, initial_temp, time, nrCells, dummy_cog, dummy_yaw, dummy_pitch, dummy_roll,
    )
    api.UpdateInternalStateBattery(battery_id, temperature, time, soc)
    trigger = api.CheckExplosionTriggersBatteryLSF(battery_id)
    api.RemoveComponent(battery_id)
    return trigger


def _evaluate(spec, mode, x, temperature, stress, mass, time):
    """Limit-state value for one component/mode at its fixed parameter values."""
    if mode == "burst":
        if isinstance(spec, WaterTankSpec):
            return float(_eval_water_tank_burst_lsf(x, temperature, mass))
        return float(_eval_fuel_tank_burst_lsf(x, temperature, mass))
    if mode == "bleve":
        return float(_eval_fuel_tank_bleve_lsf(x, temperature, stress, mass))
    if mode == "failure":
        return float(_eval_battery_lsf(x, temperature, mass, time, soc=0.0))
    raise ValueError("Unknown failure mode '{}'.".format(mode))


# ---------------------------------------------------------------------------
# Live TITAN state accessors
# ---------------------------------------------------------------------------


def _component_temperature(component):
    """Live component temperature [K] (mean over facets for array states)."""
    return float(np.mean(component.temperature))


def _component_stress(component):
    """Live max Von Mises stress [MPa] (obj.max_stress is stored in Pa).

    A value of 0 (structural solve not run) makes the BLEVE LSF fall back to
    its no-stress branch.
    """
    return float(component.max_stress) / 1.0e6

def _component_mass(component):
    """Live mass of component. Live == ablation may affect tank mass. """
    return float(component.mass)


# ---------------------------------------------------------------------------
# Failure engine
# ---------------------------------------------------------------------------


class _Tracked:
    """One tagged component, its fixed parameter values and its failure latch."""

    def __init__(self, spec, modes):
        self.spec = spec
        self.modes = modes
        #: the concrete parameter values for this run, held fixed
        self.x = spec.lsf_params()
        self.failed = False
        self.failure_mode = None
        self.failure_time = None


class FailureEngine:
    """Drives the AERIS failure tree from within the TITAN propagation loop.

    Built once per run (which for a Monte Carlo campaign is once per sample,
    after the UQMapper has set this sample's parameter values).  :meth:`step`
    evaluates every tagged component's limit state at the current temperature and
    flags newly failed components for fragmentation; :meth:`finalize` writes the
    time series and the failure events to the output folder.
    """

    def __init__(self, options, titan):

        self.options = options
        self.tracked = []

        for assembly in titan.assembly:
            for component in assembly.objects:
                ctype = getattr(component, "component_type", None)
                if ctype is None:
                    continue
                spec = _build_spec(component)
                self.tracked.append(_Tracked(spec, _SPEC_BY_TYPE[ctype][1]))

        self.has_components = bool(self.tracked)

        #: per-iteration limit-state margins
        self.results = []
        #: one row per component that failed
        self.events = []

        if not self.has_components:
            print(
                "Failure tree enabled but no components are tagged with a "
                "COMPONENT_TYPE; the failure tree will be idle."
            )
            return

        # Initialise aerisboom once, with the configured material database. Get MC run seed to pass to API
        prime_seed = options.uncertainty.prime_seed
        material_db = getattr(options, "failure_material_db", "")
        if not material_db:
            raise Exception(
                "The AERIS failure tree is enabled with tagged components but no "
                "material database was provided. Set 'Material_db' in the "
                "[Failure Tree] section of the config file."
            )
        api, DramaPrimitive = _load_api()
        api.Init(material_db, prime_seed)

        for tracked in self.tracked:
            print(
                "Failure tree - {} parameters: {}".format(
                    tracked.spec.name,
                    ", ".join("{:.6g}".format(v) for v in tracked.x),
                )
            )

    def step(self, titan, options):
        """Evaluate limit states and flag newly failed components.

        Called before ``Fragmentation.fragmentation`` in the loop, so a component
        that fails here is separated from its assembly on the same iteration.
        """
        if not self.has_components:
            return

        # Components can be removed by demise/fragmentation; a spec holding a
        # reference to one that is no longer in play must not be evaluated.
        live = {id(obj) for assembly in titan.assembly for obj in assembly.objects}

        for tracked in self.tracked:
            if tracked.failed:
                continue
            component = tracked.spec.component
            if id(component) not in live:
                continue

            temperature = _component_temperature(component)
            stress = _component_stress(component)
            mass = _component_mass(component)
                                       
            for mode in tracked.modes:
                g = _evaluate(
                    tracked.spec, mode, tracked.x, temperature, stress, mass, titan.time
                )
                self.results.append(
                    {
                        "Time": titan.time,
                        "Iter": titan.iter,
                        "Component": tracked.spec.name,
                        "Mode": mode,
                        "Temperature": temperature,
                        "g": g,
                    }
                )
                if g <= 0.0:
                    self._trigger(titan, tracked, mode, temperature, g)
                    break

    def _trigger(self, titan, tracked, mode, temperature, g):
        """Latch a component's failure and flag it for fragmentation."""
        tracked.failed = True
        tracked.failure_mode = mode
        tracked.failure_time = titan.time

        component = tracked.spec.component
        component.failed = True
        component.failure_mode = mode

        self.events.append(
            {
                "Time": titan.time,
                "Iter": titan.iter,
                "Component": tracked.spec.name,
                "Mode": mode,
                "Temperature": temperature,
                "g": g,
            }
        )
        print(
            "AERIS failure tree - {} FAILED ({}) at t={:.6g}s, T={:.4g}K, "
            "g={:.4g}; flagging for fragmentation".format(
                tracked.spec.name, mode, titan.time, temperature, g
            )
        )

    def finalize(self):
        """Write the limit-state time series and the failure events to CSV."""
        if not self.has_components:
            return

        from pathlib import Path

        import pandas as pd

        folder = self.options.output_folder + "/Failure/"
        Path(folder).mkdir(parents=True, exist_ok=True)

        if self.results:
            path = folder + "failure_tree.csv"
            pd.DataFrame(self.results).to_csv(path, index=False)
            print(
                "AERIS failure tree - wrote {} records to {}".format(
                    len(self.results), path
                )
            )

        # Always written, even when empty: a sample in which nothing failed is a
        # meaningful Monte Carlo outcome, and the campaign collation needs a row
        # per component to compute failure fractions.
        path = folder + "failure_events.csv"
        rows = list(self.events)
        for tracked in self.tracked:
            if not tracked.failed:
                rows.append(
                    {
                        "Time": np.nan,
                        "Iter": np.nan,
                        "Component": tracked.spec.name,
                        "Mode": "none",
                        "Temperature": np.nan,
                        "g": np.nan,
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)
        print(
            "AERIS failure tree - {} component(s) failed; wrote events to {}".format(
                len(self.events), path
            )
        )
