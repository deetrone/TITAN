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

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import *

## Lot of messy mapping information here to enable uq parameters to be easily exposed


## Addresses are of form 'access;pointer,access;pointer,etc'
def assembly_address(index):
    return "TITAN,attr;assembly,index;" + str(index)

# NEW_NOTE: check if correct 
def aerisboom_address(index):
    return "TITAN,attr;aerisboom_parameters"


def trajectory_address(index):
    return assembly_address(index) + ",attr;trajectory"


def component_address(assembly_index, component_index):
    return (
        assembly_address(assembly_index) + ",attr;objects,index;" + str(component_index)
    )


def option_aero_address(assembly_index):
    return "OPTIONS,attr;aerothermo"


def option_traj_address(assembly_index):
    return "OPTIONS,attr;dynamics"


def get_component_index_from_name(name, titan, assembly_index=0):
    for i_component, component in enumerate(titan.assembly[assembly_index].objects):
        # Match either the STL filename (e.g. EEB_Tank_Ylw_0.stl, as the entry-state
        # trigger__/temperature__ parameters use) or the [Objects] marker (e.g.
        # Tank_Yellow_0, which is how the AERIS failure tree names its components).
        # configparser lowercases the marker, so compare it case-insensitively.
        marker = getattr(component, "marker", None)
        if name in component.name or (
            marker is not None and name.lower() == marker.lower()
        ):
            return i_component
    raise Exception(
        "Could not find component {} in assembly {}".format(name, assembly_index)
    )


#: AERIS failure-tree component parameters exposed to the UQ yaml. Each is
#: <fieldName>__<component> and is assigned onto component.uq_parameters[fieldName]
#: (a float), overriding the baseline read from the config's [Parameters.<marker>]
#: section. The field names match the aerisboom limit-state function arguments.

# devnote: deleted original _COMPONENT_UQ_FIELDS and separated by TANK / BATTERY, address/assignment "duplicated" thus
_TANK_COMPONENT_UQ_FIELDS = [
    "volume",
    "initialTemperature",
    "nominalBurstPressure",
    "nominalTemperature",
    "propMass",
    "gasMass",
]

_BATTERY_COMPONENT_UQ_FIELDS = [
    "roomVolume",
    "initialACR",
    "ventDistanceFromCeiling",
    #"batteryGeometry", # not UQable
    #"batteryCathodeChemistry", # not UQable
    "nominalCapacity",
    "initialTemperature",
    "nrCells",
]

# NEW_NOTE: revise comments and library address assignments
#: AERIS-BOOM MODEL parameters exposed to the UQ yaml.
#: (a float), overriding the baseline read from the dataclass implementation, i.e. not exposed to cfg
#: section. The field names match the aerisboom API function arguments.
_AEBOOM_TANK_MODEL_UQ_FIELDS = [
    "tankSLTCoefficient"
]

_AEBOOM_BATTERY_MODEL_UQ_FIELDS = [
    "batteryHeatRateThresholdNMC",
    "batteryHeatRateThresholdLIP",
    "batteryGasEjectionDurationCylinder",
    "batteryGasEjectionDurationPouch",
    "batteryCapacityToGasVolumeFactor",
    "batteryPressureCriticalVolume",
    "batteryEjectedGasDensity",
    "batteryCoefficientRequiredACH",
]

_AEBOOM_BREAKUP_MODEL_UQ_FIELDS = [
    "explosionNRFragScaleFactor",
    "explosionPowerLawExponentCoefficient",
    "explosionKineticEnergyFraction",
    "collisionNrFragScaleFactor",
    "collisionPowerLawExponentCoefficient",
    "collisionLowVelocitySwitch",
    "AMswitchLcBigCoefficient",
    "AMswitchLcSmallCoefficient",
    "crossSectionAreaFactorCoefficient",
]


# Here parameters are mapped to relevant objects
library_addresses = {
    assembly_address: [
        "ECEF_x",
        "ECEF_y",
        "ECEF_z",
        "ECEF_u",
        "ECEF_v",
        "ECEF_w",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "omega_roll",
        "omega_pitch",
        "omega_yaw",
        "roll",
        "aoa",
        "slip",
        "Ixx",
        "Iyy",
        "Izz",
        "Ixy",
        "Iyz",
        "Ixz",
    ],
    trajectory_address: [
        "altitude",
        "gamma",
        "chi",
        "velocity",
        "latitude",
        "longitude",
    ],
    component_address: ["trigger__", "temperature__"]
    + [f + "__" for f in _TANK_COMPONENT_UQ_FIELDS]
    + [f + "__" for f in _BATTERY_COMPONENT_UQ_FIELDS],
    option_aero_address: ["catalycity"],
    option_traj_address: [
        "ECI_x",
        "ECI_y",
        "ECI_z",
        "ECI_u",
        "ECI_v",
        "ECI_w",
        "ECI_epoch_UNIX",
    ],
    aerisboom_address: 
        [f for f in _AEBOOM_TANK_MODEL_UQ_FIELDS]
        + [f for f in _AEBOOM_BATTERY_MODEL_UQ_FIELDS]
        + [f for f in _AEBOOM_BREAKUP_MODEL_UQ_FIELDS],
}


# Here parameters are mapped to relevant attributes (of mapped objects)
library_assignments = {
    "ECEF_x": ["state_vector", 0],
    "ECEF_y": ["state_vector", 1],
    "ECEF_z": ["state_vector", 2],
    "ECEF_u": ["state_vector", 3],
    "ECEF_v": ["state_vector", 4],
    "ECEF_w": ["state_vector", 5],
    "quat_x": ["state_vector", 6],
    "quat_y": ["state_vector", 7],
    "quat_z": ["state_vector", 8],
    "quat_w": ["state_vector", 9],
    "omega_roll": ["state_vector", 10],
    "omega_pitch": ["state_vector", 11],
    "omega_yaw": ["state_vector", 12],
    "altitude": ["altitude"],
    "gamma": ["gamma"],
    "chi": ["chi"],
    "velocity": ["velocity"],
    "latitude": ["latitude"],
    "longitude": ["longitude"],
    "ECI_x": ["trajectory_state", 0],
    "ECI_y": ["trajectory_state", 1],
    "ECI_z": ["trajectory_state", 2],
    "ECI_u": ["trajectory_state", 3],
    "ECI_v": ["trajectory_state", 4],
    "ECI_w": ["trajectory_state", 5],
    "roll": ["roll"],
    "aoa": ["aoa"],
    "slip": ["slip"],
    "trigger__": ["trigger_value"],
    "temperature__": ["temperature"],
    # AERIS failure-tree parameters -> component.uq_parameters[field]
    **{f + "__": ["uq_parameters", f] for f in _TANK_COMPONENT_UQ_FIELDS},
    **{f + "__": ["uq_parameters", f] for f in _BATTERY_COMPONENT_UQ_FIELDS},
    "Ixx": ["inertia", [0, 0]],
    "Iyy": ["inertia", [1, 1]],
    "Izz": ["inertia", [2, 2]],
    "Ixy": ["inertia", [0, 1]],
    "Iyz": ["inertia", [1, 2]],
    "Ixz": ["inertia", [0, 2]],
    "ECI_epoch_UNIX": ["trajectory_epoch"],
    "catalycity": ["cat_rate"],
    # AERIS-BOOM model parameters, NEWNOTE: check if this is correct, I think it should be.
    **{f : [f] for f in _AEBOOM_TANK_MODEL_UQ_FIELDS},
    **{f : [f] for f in _AEBOOM_BATTERY_MODEL_UQ_FIELDS},
    **{f : [f] for f in _AEBOOM_BREAKUP_MODEL_UQ_FIELDS},
}


def library_check(name):
    try:
        assert name in list(library_assignments.keys())
    except:
        raise Exception(
            "Could not find parameter {} in list, please check the UQ handbook".format(
                name
            )
        )


def state_vector_callback(titan, options, assem_ids):
    from Dynamics.propagation import construct_state_vector

    for i_assem in assem_ids:
        construct_state_vector(titan.assembly[i_assem])


def dynamic_attributes_callback(titan, options, assem_ids):
    from Dynamics.propagation import update_dynamic_attributes

    for i_assem in assem_ids:
        update_dynamic_attributes(
            titan.assembly[i_assem],
            titan.assembly[i_assem].state_vector,
            options,
            force=True,
        )


def ECI_position_callback(titan, options, assem_ids):
    import datetime as dt

    import pymap3d

    epoch = options.dynamics.trajectory_epoch
    if isinstance(epoch, str):
        epoch = dt.datetime.strptime(epoch, "%Y/%m/%d %H:%M:%S")
    elif isinstance(epoch, float) or isinstance(epoch, int):
        epoch = dt.datetime.fromtimestamp(epoch)
    for i_assem in assem_ids:
        state = options.dynamics.trajectory_state
        state[:3] = pymap3d.eci2ecef(state[0], state[1], state[2], time=epoch)
        rotMatrix = np.transpose(
            [
                pymap3d.eci2ecef(1, 0, 0, epoch),
                pymap3d.eci2ecef(0, 1, 0, epoch),
                pymap3d.eci2ecef(0, 0, 1, epoch),
            ]
        )
        state[3:] = rotMatrix @ state[3:] - np.cross(
            [0, 0, options.planet.omega()], state[:3]
        )
        titan.assembly[i_assem].state_vector[:6] = state
    dynamic_attributes_callback(titan, options, assem_ids)


def orientation_callback(titan, options, assem_ids):
    from Dynamics.dynamics import compute_quaternion

    for i_assem in assem_ids:
        compute_quaternion(titan.assembly[i_assem])
        titan.assembly[i_assem].state_vector[6:10] = titan.assembly[i_assem].quaternion
    state_vector_callback(titan, options, assem_ids)


def symmetrize_inertia_callback(titan, options, assem_ids):
    for i_assem in assem_ids:
        titan.assembly[i_assem].inertia[1, 0] = titan.assembly[i_assem].inertia[0, 1]
        titan.assembly[i_assem].inertia[2, 0] = titan.assembly[i_assem].inertia[0, 2]
        titan.assembly[i_assem].inertia[2, 1] = titan.assembly[i_assem].inertia[1, 2]


callbacks = {
    state_vector_callback: [
        "altitude",
        "latitude",
        "longitude",
        "gamma",
        "chi",
        "velocity",
    ],
    ECI_position_callback: [
        "ECI_x",
        "ECI_y",
        "ECI_z",
        "ECI_u",
        "ECI_v",
        "ECI_w",
        "ECI_epoch_UNIX",
    ],
    dynamic_attributes_callback: [
        "ECEF_x",
        "ECEF_y",
        "ECEF_z",
        "ECEF_u",
        "ECEF_v",
        "ECEF_w",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "omega_roll",
        "omega_pitch",
        "omega_yaw",
    ],
    orientation_callback: [
        "chi",
        "gamma",
        "roll",
        "aoa",
        "slip",
        "latitude",
        "longitude",
    ],
    symmetrize_inertia_callback: ["Ixy", "Iyz", "Ixz"],
}


## Valid classes for the distribution have to...
#  ...accept (even a dummy) input
#  ...have the attribute random_state
#  ...have the method rvs()
class quat_rotation:
    def __init__(self, d=3):
        self.d = 3
        self.random_state = None

    def rvs(self, n=1):
        quat = Rotation.random(num=n, random_state=self.random_state).as_quat()
        return quat[0]


class pbox_normal:
    """Normal p-box distribution: mean in [mu_lo, mu_hi], std in [std_lo, std_hi].

    A p-box carries epistemic uncertainty (the true distribution is unknown
    within the box) that a single scipy distribution cannot express, so it is
    propagated as the outer loop of a double-loop Monte Carlo rather than sampled
    once.  ``MC_wrapper`` detects samplers with ``n_corners > 1``, enumerates the
    corners, and sets ``corner`` on each before running that sub-campaign.

    Because the AERIS limit-state functions are monotonic in the sampled value,
    only the two ends of the focal interval matter -- the smallest and largest
    value the box can produce at a shared quantile -- so ``n_corners`` is 2, not
    the four vertices of the (mu, std) box.  ``rvs`` draws a single standard
    normal per call, so under common random numbers the two endpoints of a given
    sample land on the same quantile across sub-campaigns.
    """

    n_corners = 2

    def __init__(
        self, mu_lo, mu_hi, std_lo=None, std_hi=None, var_lo=None, var_hi=None
    ):
        if var_lo is None:
            var_lo = std_lo**2
        if var_hi is None:
            var_hi = std_hi**2
        if mu_lo > mu_hi:
            raise ValueError("pbox_normal: mu_lo must be <= mu_hi")
        if var_lo > var_hi or var_lo <= 0:
            raise ValueError("pbox_normal: var_lo must be > 0 and <= var_hi")
        self.mu_lo = mu_lo
        self.mu_hi = mu_hi
        self.var_lo = var_lo
        self.var_hi = var_hi
        self.random_state = None
        self.corner = 0

    def corner_label(self, corner):
        return ("lower", "upper")[corner]

    def rvs(self, n=1):
        z = float(self.random_state.standard_normal())
        if self.corner == 0:
            # Smallest value the box can reach at this quantile.
            var = self.var_lo if z >= 0 else self.var_hi
            return self.mu_lo + z * np.sqrt(var)
        # Largest value the box can reach at this quantile.
        var = self.var_hi if z >= 0 else self.var_lo
        return self.mu_hi + z * np.sqrt(var)


class dempster_shafer:
    """Dempster-Shafer structure: focal intervals, each carrying a mass.

    An evidence-theory model for a parameter known only through a finite body of
    evidence, where each piece of evidence pins the value to an *interval*
    rather than a point and carries a basic probability assignment (its mass).
    Masses must sum to 1.  Unlike a p-box the focal intervals need not be nested
    or ordered and may overlap freely, so this can express set-valued evidence
    -- two disagreeing test campaigns, a spec range plus a measured range --
    that no single distribution, however imprecise its parameters, represents.

    A structure with one focal interval of mass 1 is pure interval uncertainty:
    total ignorance within the interval, with no distribution assumed over it.

    Propagation reuses the same double-loop as :class:`pbox_normal`, one
    sub-campaign per corner.  Each sample draws a focal interval according to
    its mass; corner 0 then takes that interval's lower endpoint and corner 1
    its upper.  Under common random numbers both corners draw the *same* focal
    interval for a given sample, so the sub-campaigns differ only in which end
    of it was evaluated.  Given monotonic limit states, the resulting failure
    probabilities bracket the evidence-theory bounds::

        Bel(fail) = min(P_fail over corners)
        Pl(fail)  = max(P_fail over corners)

    Which corner yields which bound depends on the direction the limit state
    moves in, so the bracket is read off the results rather than assumed.
    """

    n_corners = 2

    def __init__(self, intervals, masses=None):
        intervals = np.asarray(intervals, dtype=float)
        if intervals.ndim != 2 or intervals.shape[0] == 0 or intervals.shape[1] != 2:
            raise ValueError(
                "dempster_shafer: intervals must be a non-empty list of "
                "[lower, upper] pairs, got shape {}".format(intervals.shape)
            )
        if np.any(intervals[:, 0] > intervals[:, 1]):
            raise ValueError(
                "dempster_shafer: every focal interval needs lower <= upper"
            )

        if masses is None:
            # No evidence weighting given: treat the focal intervals as equally
            # supported rather than guessing at a weighting.
            masses = np.full(len(intervals), 1.0 / len(intervals))
        else:
            masses = np.asarray(masses, dtype=float).ravel()
        if len(masses) != len(intervals):
            raise ValueError(
                "dempster_shafer: got {} focal intervals but {} masses".format(
                    len(intervals), len(masses)
                )
            )
        if np.any(masses <= 0):
            raise ValueError("dempster_shafer: every mass must be > 0")
        total = masses.sum()
        # Masses that miss 1 by more than rounding are a modelling error, not
        # something to silently renormalise away. (np.abs, not abs: the
        # `from scipy.stats import *` above shadows the builtin.)
        if np.abs(total - 1.0) > 1e-6:
            raise ValueError(
                "dempster_shafer: masses must sum to 1, got {:.8g}".format(total)
            )

        self.intervals = intervals
        self.masses = masses / total
        self._cumulative = np.cumsum(self.masses)
        self._cumulative[-1] = 1.0  # guard against cumsum landing just short
        self.random_state = None
        self.corner = 0

    def corner_label(self, corner):
        # Names the end of the focal interval, not the probability bound it
        # produces -- see the class docstring.
        return ("lower", "upper")[corner]

    def rvs(self, n=1):
        # Exactly one variate per call in either corner, so every sampler after
        # this one stays in sync across sub-campaigns under common random
        # numbers, and both corners select the same focal interval.
        u = float(self.random_state.random_sample())
        i = min(
            int(np.searchsorted(self._cumulative, u, side="right")),
            len(self.intervals) - 1,
        )
        return self.intervals[i][self.corner]


# Sometimes extensibility ain't pretty
available_distris = {
    "alpha": alpha,
    "anglit": anglit,
    "arcsine": arcsine,
    "argus": argus,
    "beta": beta,
    "betaprime": betaprime,
    "bradford": bradford,
    "burr": burr,
    "burr12": burr12,
    "cauchy": cauchy,
    "chi": chi,
    "chi2": chi2,
    "cosine": cosine,
    "crystalball": crystalball,
    "dgamma": dgamma,
    "dweibull": dweibull,
    "erlang": erlang,
    "expon": expon,
    "exponnorm": exponnorm,
    "exponweib": exponweib,
    "exponpow": exponpow,
    "f": f,
    "fatiguelife": fatiguelife,
    "fisk": fisk,
    "foldcauchy": foldcauchy,
    "foldnorm": foldnorm,
    "genlogistic": genlogistic,
    "gennorm": gennorm,
    "genpareto": genpareto,
    "genexpon": genexpon,
    "genextreme": genextreme,
    "gausshyper": gausshyper,
    "gamma": gamma,
    "gengamma": gengamma,
    "genhalflogistic": genhalflogistic,
    "genhyperbolic": genhyperbolic,
    "geninvgauss": geninvgauss,
    "gibrat": gibrat,
    "gompertz": gompertz,
    "gumbel_r": gumbel_r,
    "gumbel_l": gumbel_l,
    "halfcauchy": halfcauchy,
    "halflogistic": halflogistic,
    "halfnorm": halfnorm,
    "halfgennorm": halfgennorm,
    "hypsecant": hypsecant,
    "invgamma": invgamma,
    "invgauss": invgauss,
    "invweibull": invweibull,
    "johnsonsb": johnsonsb,
    "johnsonsu": johnsonsu,
    "kappa4": kappa4,
    "kappa3": kappa3,
    "ksone": ksone,
    "kstwo": kstwo,
    "kstwobign": kstwobign,
    "laplace": laplace,
    "laplace_asymmetric": laplace_asymmetric,
    "levy": levy,
    "levy_l": levy_l,
    "levy_stable": levy_stable,
    "logistic": logistic,
    "loggamma": loggamma,
    "loglaplace": loglaplace,
    "lognorm": lognorm,
    "loguniform": loguniform,
    "lomax": lomax,
    "maxwell": maxwell,
    "mielke": mielke,
    "moyal": moyal,
    "nakagami": nakagami,
    "ncx2": ncx2,
    "ncf": ncf,
    "nct": nct,
    "norm": norm,
    "normal": norm,
    "norminvgauss": norminvgauss,
    "pareto": pareto,
    "pearson3": pearson3,
    "powerlaw": powerlaw,
    "powerlognorm": powerlognorm,
    "powernorm": powernorm,
    "rdist": rdist,
    "rayleigh": rayleigh,
    "rice": rice,
    "recipinvgauss": recipinvgauss,
    "semicircular": semicircular,
    "skewcauchy": skewcauchy,
    "skewnorm": skewnorm,
    "studentized_range": studentized_range,
    "t": t,
    "trapezoid": trapezoid,
    "triang": triang,
    "truncexpon": truncexpon,
    "truncnorm": truncnorm,
    "truncpareto": truncpareto,
    "truncweibull_min": truncweibull_min,
    "tukeylambda": tukeylambda,
    "uniform": uniform,
    "vonmises": vonmises,
    "vonmises_line": vonmises_line,
    "wald": wald,
    "weibull_min": weibull_min,
    "weibull_max": weibull_max,
    "wrapcauchy": wrapcauchy,
    "multivariate_normal": multivariate_normal,
    "matrix_normal": matrix_normal,
    "dirichlet": dirichlet,
    "wishart": wishart,
    "invwishart": invwishart,
    "multinomial": multinomial,
    "special_ortho_group": special_ortho_group,
    "ortho_group": ortho_group,
    "unitary_group": unitary_group,
    "random_correlation": random_correlation,
    "multivariate_t": multivariate_t,
    "multivariate_hypergeom": multivariate_hypergeom,
    "random_table": random_table,
    "uniform_direction": uniform_direction,
    "bernoulli": bernoulli,
    "betabinom": betabinom,
    "binom": binom,
    "boltzmann": boltzmann,
    "dlaplace": dlaplace,
    "geom": geom,
    "hypergeom": hypergeom,
    "logser": logser,
    "nbinom": nbinom,
    "nchypergeom_fisher": nchypergeom_fisher,
    "nchypergeom_wallenius": nchypergeom_wallenius,
    "nhypergeom": nhypergeom,
    "planck": planck,
    "poisson": poisson,
    "randint": randint,
    "skellam": skellam,
    "yulesimon": yulesimon,
    "zipf": zipf,
    "zipfian": zipfian,
    "quat_rotation": quat_rotation,
    "pbox_normal": pbox_normal,
    "dempster_shafer": dempster_shafer,
    "ds": dempster_shafer,
}
