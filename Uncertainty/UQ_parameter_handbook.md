## TITAN Uncertainty Propagation: Parameter Handbook
*24.11.2025* 
  
This handbook details all exposed variables that can be controlled using a *.yaml* file (see associated example in this folder).  Parametres are sorted by options section and given in the following format (note names are case sensitive)...  
*{**NAME** [data type] : Brief description}*

## Trajectory
 - **altitude** [metres]        : Initial height above body
 - **gamma** [radians]          : Initial flight path angle value.
 - **chi** [radians]            : Initial heading angle value.
 - **velocity** [metres/second] : Initial velocity value.
 - **latitude** [radians]       : Initial latitude value. 
 - **longitude** [radians]      : Initial longitude value.
 **OR**
 - **ECEF_x** [metres/second] : Initial x position in ECEF frame
 - **ECEF_y** [metres/second] : Initial y position in ECEF frame
 - **ECEF_z** [metres/second] : Initial z position in ECEF frame
 - **ECEF_u** [metres/second] : Initial u velocity in ECEF frame
 - **ECEF_v** [metres/second] : Initial v velocity in ECEF frame
 - **ECEF_w** [metres/second] : Initial w velocity in ECEF frame
 **OR**
 - **ECI_x** [metres/second]    : Initial x position in ECI frame
 - **ECI_y** [metres/second]    : Initial y position in ECI frame
 - **ECI_z** [metres/second]    : Initial z position in ECI frame
 - **ECI_u** [metres/second]    : Initial u velocity in ECI frame
 - **ECI_v** [metres/second]    : Initial v velocity in ECI frame
 - **ECI_w** [metres/second]    : Initial w velocity in ECI frame
 - **ECI_epoch_UNIX** [seconds] : ECI initial time (seconds since Jan 1 1970 00:00:00)

## Attitude
 - **omega_roll** [radians/second]  : Initial rotation rate about body roll (x) axis.
 - **omega_pitch** [radians/second] : Initial rotation rate about body pitch (y) axis.
 - **omega_yaw** [radians/second]   : Initial rotation rate about body yaw (z) axis.

 - **quat_x** [0-1] : Initial body-frame quaterion x component
 - **quat_y** [0-1] : Initial body-frame quaterion y component
 - **quat_z** [0-1] : Initial body-frame quaterion z component
 - **quat_w** [0-1] : Initial body-frame quaterion w component
  **OR**
 - **roll** [radians] : Initial rotation about wind roll (x) axis.
 - **aoa** [radians]  : Initial rotation about wind pitch (y) axis.
 - **slip** [radians] : Initial rotation about wind yaw (z) axis.

## Assembly
**Note if assembly fragments inertia tensor will be recalculated (these parameters will be discarded)**
- **Ixx** [kilogram/metre^2] : Moment of inertia about roll (x) axis
- **Iyy** [kilogram/metre^2] : Moment of inertia about pitch (y) axis
- **Izz** [kilogram/metre^2] : Moment of inertia about yaw (z) axis
- **Ixy** [kilogram/metre^2] : x-y product of inertia
- **Iyz** [kilogram/metre^2] : y-z product of inertia
- **Ixz** [kilogram/metre^2] : x-z product of inertia

## Aerothermo
- **catalycity** [0-1] : Surface catalytic efficiency/recombination, note "material" catalycity must be disabled

## Objects
Uncertain parameters assigned to objects should have the object name after the double underscore (e.g. 'trigger__Cube_A.stl')
- **trigger__** [float]      : The fragmentation trigger of the object, units determined by how the trigger is set
- **temperature__** [Kelvin] : The initial temperature of the object
**(Variables exposed by the in-development explosion model)**
- **energy__** [J] : Available energy of explosion

## AERIS failure tree
Components tagged with a `COMPONENT_TYPE` in the config's `[Objects]` block (watertank / fueltank / battery) have their uncertainty declared **here, the same way as everything else** — as `<field>__<marker>` parameters bound to distributions. The `<marker>` is the `[Objects]` name (e.g. `Tank_Yellow_0`, matched case-insensitively) or the STL filename. The **baseline** value of every field lives in the config's `[Parameters.<marker>]` section as a plain number; a field named here overrides its baseline once per sample, and a plain (non `-MC`) run uses the baselines unchanged.

Fields (per component type), matching the aerisboom limit-state arguments:
- fuel tank: `volume__`, `initialTemperature__`, `nominalTemperature__`, `nominalBurstPressure__`, `nominalUltimateTensileStrength__`, `liqVolume__`, `liqMass__`, `propMass__`, `gasMass__`
- water tank: as fuel tank without `propMass__`
- battery: `volume__`, `initialTemperature__`, `initialACR__`, `ventDistanceFromCeiling__`, `nominalCapacity__`

For independent scatter across several tanks (e.g. per-tank strength), use one `multivariate_normal` with a diagonal covariance and index each tank into it — a single univariate distribution shared by several parameters would make them fully correlated.

A component whose limit state is crossed during the run is separated from its assembly by `Fragmentation`, so a failure changes the trajectory of what remains rather than only being reported. The campaign writes `failure_summary.csv` per sub-campaign, in which `P_fail` is the fraction of samples whose component actually failed.

### Imprecise (epistemic) parameters

Two custom distributions model uncertainty that cannot be reduced to a single distribution without turning epistemic ignorance into aleatory spread. Both may be mixed freely with ordinary distributions and with each other.

**P-boxes.** `pbox_normal` — normally distributed, but with the mean in `[mu_lo, mu_hi]` and the std in `[std_lo, std_hi]` (or `var_lo`/`var_hi`). Only the two ends of the focal interval are evaluated, not the four vertices of the (mean, std) box.

**Dempster-Shafer structures.** `dempster_shafer` (alias `ds`) — a finite body of evidence, where each piece pins the value to an *interval* rather than a point and carries a mass (basic probability assignment). Focal intervals need not be nested or ordered and may overlap, so this expresses set-valued evidence a p-box cannot:

```yaml
distributions:
  0:
    dempster_shafer:
      intervals: [[1.550e+6, 1.570e+6], [1.560e+6, 1.600e+6], [1.500e+6, 1.620e+6]]
      masses: [0.5, 0.3, 0.2]      # must sum to 1; omit for equal support
```

A single focal interval of mass 1 is pure interval uncertainty (total ignorance within it); a focal interval with `lower == upper` is a precise value carrying that mass. Each sample draws one focal interval by its mass, then evaluates its lower endpoint in one corner and its upper in the other — so the two sub-campaign failure probabilities are the **belief** and **plausibility** of failure.

**Propagation (both models).** Imprecise parameters are the outer loop of a double-loop Monte Carlo: `-MC n` runs one full sub-campaign per joint corner assignment, written to `Campaign_<seed>/Corner_<i>/`, and the spread between them is the imprecision in the result. Precise parameters draw identically in every corner (common random numbers), so a corner-to-corner comparison isolates the imprecision. `Campaign_<seed>/failure_bounds.csv` brackets each component's `P_fail` across the corners as `Belief`/`Plausibility`.

Cost grows as `2 ** (number of imprecise parameters)` — keep the count small. This relies on the AERIS limit-state functions being **monotonic** in the sampled value, so that the box/focal interior is bracketed by its endpoints. Should a limit state stop being monotonic, the assumption is violated and a fuller propagation is needed.

Note scientific notation: the UQ yaml is read with a widened float resolver, so `1.57e6` is fine (unlike stock PyYAML, which needs a signed exponent `1.57e+6`).
