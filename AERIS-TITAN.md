Integration of the AERIS failure tree into TITAN's Monte Carlo UQ wrapper.
A tagged component (fuel tank, water tank, battery) is evaluated against its
limit state every propagation step using its live reentry temperature; the first
step where the limit state is crossed fragments that component off its
assembly, and the resulting mass/inertia change alters the trajectory from that
point on.

## What has changed

- `Failure/failure.py` (new) — the failure engine. No sampling
  lives here: it reads whatever determinsitic float values are on each tagged
  component (`component.uq_parameters`), holds them fixed for the run, and
  evaluates the relevant `aerisboom` limit-state function(s) every iteration at
  the component's live temperature. `g <= 0` latches `component.failed = True`
  and `component.failure_mode`. Writes `failure_tree.csv` (the margin time
  series) and `failure_events.csv` (one row per tracked component, `Mode='none'`
  if it never failed) under `<output_folder>/Failure/`.

- `Fragmentation/fragmentation.py` — a new branch in the Primitive path
  detaches any component with `obj.failed and not obj.failure_separated` from
  its assembly (`separate_primitive`, factored out of the existing `altitude`
  trigger path), on the same iteration it fails.

- `Uncertainty/mappings.py` — exposes 12 AERIS component fields
  (`_COMPONENT_UQ_FIELDS`) to the UQ address/assignment tables as
  `<field>__<marker>`, resolving onto `component.uq_parameters[field]`. Adds
  `pbox_normal`, a custom p-box distribution (mean and std/var each given as an
  interval) for imprecise-probability parameters, following the same pattern as
  the existing `quat_rotation` custom distribution. `get_component_index_from_name`
  now matches a component's `[Objects]` marker case-insensitively, because
  `configparser` lowercases option keys but the YAML and the failure tree use
  the original casing (e.g. `Tank_Yellow_0`).

- `Uncertainty/MC_wrapper.py` — detects any sampler with `n_corners > 1`
  (a p-box or a Dempster-Shafer structure) and runs one sub-campaign per corner
  combination, writing each to `Campaign_<seed>/Corner_<i>/` instead of flat
  `Campaign_<seed>/`. Every precise parameter uses the same seed stream across
  corners (common random numbers), so only the imprecise parameters differ
  between sub-campaigns. After the last corner, `collate_failure_bounds` writes
  `Campaign_<seed>/failure_bounds.csv` bracketing each component's failure
  probability across them.
  NOTE: This running of corners is an efficency hack that only works on the
  failure-tree related parameters as the failure tree limit state functions
  are monotonic. Should any LSF no longer be monotonic, this assumption is 
  violated and a more thorough P-box propagation will need to be added.
  The same caveat applies to the Dempster-Shafer structures, whose belief and
  plausibility bounds are likewise read off the focal-interval endpoints.

- `Uncertainty/utils.py` — `UQLoader`, a YAML loader with a widened float
  resolver, because stock PyYAML (YAML 1.1) only recognizes scientific notation
  with a signed exponent (`1.57e+6`) and silently leaves `1.57e6` as a string.
  Burst pressures are naturally written without the `+`, so the UQ YAML is now
  parsed with `UQLoader` instead of the default loader. Also adds
  `collate_failures`, which aggregates every sample's `failure_events.csv` in a
  sub-campaign into `failures.csv` and a per-component `failure_summary.csv`
  with `P_fail = (samples where that component failed) / (samples in the group)`.

- `Configuration/configuration.py` — `[Parameters.<marker>]` sections now
  hold only plain baseline floats (`EXPECTED_COMPONENT_PARAMETERS` lists the
  required fields per component type); `read_component_parameters` parses them
  with `float()` and raises if a value isn't a plain number, pointing at the
  YAML as the place to declare uncertainty instead.

- `Geometry/component.py` — new attributes: `component_type` (validated
  against `Component.COMPONENT_TYPES = ("watertank", "fueltank", "battery")`,
  set via `COMPONENT_TYPE = ...` in the `[Objects]` line), `marker`,
  `uq_parameters` (dict, populated by `read_component_parameters` and
  overwritten per-sample by `UQMapper`), `failed`, `failure_mode`,
  `failure_separated`.

- `TITAN.py` — `loop()` builds a `FailureEngine` once per run when
  `options.enable_failure_tree` is set, calls `failure_engine.step(titan, options)`
  immediately before `fragmentation.fragmentation(...)` every iteration, and
  calls `failure_engine.finalize()` on every exit path (normal completion,
  early break, exception).

## Config reference

### `[Objects]` — tag a component

Add `COMPONENT_TYPE = <FuelTank|WaterTank|Battery>` to the object's definition
line (case-insensitive):

```ini
Tank_Yellow_0 = [NAME = EEB_Tank_Ylw_0.stl, TYPE = Primitive, MATERIAL = drama-TiAl6v4_Semi-Empirical, COMPONENT_TYPE = FuelTank]
```

### `[Parameters.<marker>]` — baseline values

One section per tagged component, `<marker>` matching the object's name in
`[Objects]` (lowercase, e.g. `Tank_Yellow_0` → `[Parameters.Tank_Yellow_0]`).
Every field is a plain float; the required fields depend on the component type:

| Type | Required fields |
|---|---|
| `watertank` | `Volume`, `InitialTemperature`, `NominalTemperature`, `NominalBurstPressure`, `NominalUltimateTensileStrength`, `LiqVolume`, `LiqMass`, `GasMass` |
| `fueltank` | same as `watertank`, plus `PropMass` |
| `battery` | `Volume`, `InitialTemperature`, `InitialACR`, `VentDistanceFromCeiling`, `NominalCapacity` |

```ini
[Parameters.Tank_Yellow_0]
Volume                          = 0.2
InitialTemperature              = 300
NominalTemperature              = 300
NominalBurstPressure            = 1.575e6
NominalUltimateTensileStrength  = 630
LiqVolume                       = 0.1
LiqMass                         = 100
PropMass                        = 100
GasMass                         = 0.25
```

### `[Failure Tree]` — enable the engine

```ini
[Failure Tree]
Enable_Tree = True
Material_db = /path/to/material_db.json
```

`Material_db` points at the `aerisboom` material-property JSON.

### UQ YAML

Format is TITAN's standard UQ YAML: `parameters` (name → `[assembly_index,
distribution_index, index_in_distribution]`), `distributions` (index →
`{scipy_dist_name: {kwargs}}`), `outputs`. To put uncertainty on a component
field, name the parameter `<field>__<marker>`:

```yaml
parameters:
  nominalBurstPressure__Tank_Yellow_0: [0, 0, 0]
  nominalUltimateTensileStrength__Tank_Yellow_0: [0, 1, 0]

distributions:
  0: {norm: {loc: 1.575e+6, scale: 2.0e+4}}
  1: {norm: {loc: 630, scale: 20}}

outputs:
  Altitude: all
  Mass: all
```

Available fields (any of the required `[Parameters.<marker>]` fields above,
camelCase): `volume`, `initialTemperature`, `nominalTemperature`,
`nominalBurstPressure`, `nominalUltimateTensileStrength`, `liqVolume`,
`liqMass`, `propMass`, `gasMass`, `initialACR`, `ventDistanceFromCeiling`,
`nominalCapacity`.

### Imprecise parameters

Two imprecise (epistemic) models are available where a single distribution
cannot express what is actually known about a parameter. Both are propagated
the same way — see *Corner propagation* below — and both may be mixed freely
with ordinary scipy distributions and with each other.

#### P-box — `pbox_normal`

For a parameter that is normally distributed but whose mean and/or standard
deviation are themselves only known to lie in an interval:

```yaml
distributions:
  0:
    pbox_normal:
      mu_lo: 1.570e+6
      mu_hi: 1.580e+6
      std_lo: 1.2e+4
      std_hi: 2.0e+4
```

`var_lo`/`var_hi` may be given instead of `std_lo`/`std_hi`. Only the two ends
of the focal interval are evaluated, not all four vertices of the (mean, std)
box, because the limit states are monotonic in the sampled value.

#### Dempster-Shafer structure — `dempster_shafer` (alias `ds`)

For a parameter known through a finite body of evidence, where each piece of
evidence pins the value to an *interval* rather than a point and carries a mass
(a basic probability assignment). Masses must sum to 1:

```yaml
distributions:
  0:
    dempster_shafer:
      intervals:
        - [1.550e+6, 1.570e+6]    # vendor spec sheet
        - [1.560e+6, 1.600e+6]    # qualification test campaign
        - [1.500e+6, 1.620e+6]    # heritage/engineering judgement
      masses: [0.5, 0.3, 0.2]
```

Each sample draws one focal interval according to its mass; corner 0 then takes
that interval's lower endpoint and corner 1 its upper. Under common random
numbers both corners draw the *same* focal interval for a given sample, so the
sub-campaigns differ only in which end of it was evaluated. Given monotonic
limit states, the resulting failure probabilities are the belief and
plausibility of failure — the evidence-theory lower and upper bounds.

#### Corner propagation

`MC_wrapper` runs one full sub-campaign per joint corner assignment, so output
goes to `Campaign_<seed>/Corner_<i>/` rather than a flat `Campaign_<seed>/`.
Cost is `2 ** (number of imprecise parameters)` sub-campaigns; with one p-box
and one DS structure that is 4. Every precise parameter draws identically in
every corner, so a corner-to-corner comparison isolates the imprecision.

After the last corner, `failure_bounds.csv` is written at the campaign root
with each component's `Belief` and `Plausibility` — the smallest and largest
`P_fail` across the corners — and which corner produced each. Which corner
yields which bound depends on the direction the limit state moves in, so the
bracket is read off the results rather than assumed.

## Running

Output layout for a campaign:

```
<Output_folder>/Campaign_<seed>/                      # no imprecise params
<Output_folder>/Campaign_<seed>/failure_bounds.csv     # imprecise: Bel/Pl per component
<Output_folder>/Campaign_<seed>/Corner_0/              # imprecise: one folder per corner
<Output_folder>/Campaign_<seed>/Corner_1/
    MC_<i>/Failure/failure_tree.csv                   # per-sample margin time series
    MC_<i>/Failure/failure_events.csv                  # per-sample failure/no-failure per component
    failures.csv                                       # collate_failures: all samples' events
    failure_summary.csv                                 # collate_failures: P_fail per component
```
## Known limitations

- `aeris-boom` is not modified and its `ExplodeComponent`/EVOLVE breakup
  model is not used. The `CauseOfExplosion` enum it needs is not registered
  in the Python bindings, and calling it produced fragment clouds with no
  seedable RNG and (separately) an argument-order bug that mis-assigns
  `volume` in the explosion-energy calculation. Trajectory divergence after a
  failure comes entirely from TITAN's own fragmentation (mass, inertia,
  ballistic coefficient change, plus the existing separation velocity from
  angular rate), not an explosive delta-v.

- `CheckExplosionTriggersBatteryLSF` does not use
  the temperature argument it is given, so a battery component can never
  reach `g <= 0` through reentry heating in the current `aerisboom` build.

- The BLEVE limit state needs a wall stress; `component.max_stress` is 0 unless
  TITAN's structural solve has run, which falls back to the LSF's no-stress
  branch. `burst` is the only mode that responds to live reentry temperature
  in practice.

- `propMass` is accepted but never read by any limit-state function (fuel
  mass does not appear in the burst/BLEVE LSF argument lists) — it exists only
  because it was part of the original parameter set and TITAN still requires
  it as a baseline. Sampling `propMass__<marker>` in the YAML has no effect.

- `initialTemperature` is only used by the battery LSF. The tank LSFs
  (`_eval_water_tank_burst_lsf`, `_eval_fuel_tank_burst_lsf`,
  `_eval_fuel_tank_bleve_lsf`) accept it but pass the component's *live*
  temperature to `aerisboom` instead, by design — the whole point of this
  engine is evaluating at the current reentry temperature rather than a fixed
  initial one. Sampling `initialTemperature__<marker>` for a tank has no effect.
