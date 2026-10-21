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


import configparser
import concurrent.futures
import itertools
import numpy as np
import pathlib
import copy
import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader

from TITAN import loop
from Configuration.configuration import read_config_file
from Uncertainty.utils import UQMapper, report_outputs_from_csv, collate_QoI, collate_failures, collate_failure_bounds
from Dynamics.propagation import construct_state_vector

def campaign_folder(prime_seed):
    """Root output folder for a campaign, relative to the base output folder."""
    return '/Campaign_' + str(prime_seed)

def campaign_subfolder(prime_seed, corner, n_corners):
    """Output folder for one sub-campaign, relative to the base output folder.

    A configuration with no imprecise parameters runs a single campaign and keeps
    the original layout; imprecise ones get one Corner_<corner> campaign each.
    """
    folder = campaign_folder(prime_seed)
    if n_corners > 1: folder += '/Corner_' + str(corner)
    return folder

def _epistemic_samplers(uq_mapper):
    """Distribution samplers that carry epistemic uncertainty (n_corners > 1).

    Covers every imprecise model propagated as corners of an outer loop -- both
    p-boxes and Dempster-Shafer structures. Returns ``[(sampler_index, sampler),
    ...]``; empty for an ordinary campaign.
    """
    return [(j, s) for j, s in enumerate(uq_mapper.samplers)
            if getattr(s, 'n_corners', 1) > 1]

def _sampler_param_names(uq_mapper, sampler_index):
    """Names of the yaml parameters bound to a given distribution sampler."""
    return [p.name for p in uq_mapper.parameters if p.code[1] == sampler_index]

def run(filename,n_samples):
    n_samples = int(n_samples)
    configParser = configparser.RawConfigParser()
    configFilePath = filename.lstrip()
    configParser.read(configFilePath)


    base_options, base_titan = read_config_file(configParser,'')
    base_folder = base_options.output_folder
    prime_seed = base_options.uncertainty.prime_seed

    seedlist = np.random.RandomState(prime_seed).random_integers(0,2**32-1,n_samples)

    base_titan.uq_mapper = UQMapper(base_titan, base_options)

    if base_options.collision.flag: # Can't serialise collision data
        for _assembly in base_titan.assembly: _assembly.collision = None

    # An imprecise distribution (n_corners > 1 -- a p-box or a Dempster-Shafer
    # structure) carries epistemic uncertainty that a single distribution can't
    # represent, so it is propagated as the outer loop of a double-loop Monte
    # Carlo: one full sub-campaign per joint corner assignment, and the spread
    # across them is the imprecision in the result.
    epistemic = _epistemic_samplers(base_titan.uq_mapper)
    combos = list(itertools.product(*[range(s.n_corners) for _j, s in epistemic])) or [()]
    n_corners = len(combos)
    subfolders = []
    if n_corners > 1:
        print('Imprecise parameters detected: running {} sub-campaigns of {} samples ({} runs total)'.format(
            n_corners, n_samples, n_corners*n_samples))

    for ci, combo in enumerate(combos):
        # Point every imprecise sampler at its corner for this sub-campaign; the
        # setting survives the per-sample deepcopy/pickle of base_titan.
        for (_j, sampler), corner in zip(epistemic, combo): sampler.corner = corner

        subfolder = campaign_subfolder(prime_seed, ci, n_corners)
        subfolders.append(subfolder)
        if n_corners > 1:
            label = '; '.join(
                '{} -> {}'.format(','.join(_sampler_param_names(base_titan.uq_mapper, j)),
                                  sampler.corner_label(corner))
                for (j, sampler), corner in zip(epistemic, combo))
            print('\nStarting sub-campaign {}/{}: {}'.format(ci+1, n_corners, label))

        if base_options.uncertainty.n_procs>1:
            with concurrent.futures.ProcessPoolExecutor(base_options.uncertainty.n_procs) as executor:
                output_futures = [executor.submit(wrapper,base_titan,base_options,i_sample,seed,subfolder) for i_sample, seed in enumerate(seedlist)]

                for i_sim, f in enumerate(concurrent.futures.as_completed(output_futures)):
                    if f._exception:
                        print('Error on result number {}: {}'.format(i_sim,f.exception()))
                    else:
                        print('Finished sim: '+str(i_sim+1)+' ('+str(round(100*(i_sim+1)/n_samples,4))+'%)')
                concurrent.futures.wait(output_futures)
        else:

            for i_sample, seed in enumerate(seedlist):
                new_titan = copy.deepcopy(base_titan)
                new_options = copy.deepcopy(base_options)
                wrapper(new_titan, new_options, i_sample, seed, subfolder)
                del new_titan, new_options
        collate_QoI(seedlist, base_folder, subfolder)
        if base_options.enable_failure_tree:
            collate_failures(seedlist, base_folder, subfolder)

    # With imprecise inputs there is no single failure probability, only the
    # interval the corners span, so bracket each component across them.
    if n_corners > 1 and base_options.enable_failure_tree:
        collate_failure_bounds(base_folder, campaign_folder(prime_seed), subfolders)

def wrapper(titan, options, i_sample, seed, subfolder):
    print('Starting run {} seed [{}]'.format(i_sample, str(seed).rjust(10,'0')))
    options.output_folder += subfolder + '/MC_' + str(i_sample)
    options.clean_up_folders()
    options.create_output_folders()
    component_list = []
    for _assembly in titan.assembly: 
        construct_state_vector(_assembly)
        [component_list.append(comp.name) for comp in _assembly.objects]
    state_info = titan.uq_mapper.map_from_seed(seed, titan, options)
    write_datafile(options.output_folder, i_sample, state_info)
    options.plot = True if i_sample % round(2*options.uncertainty.n_procs) == 0 else False
    options.filepath = ''
    try:
        loop(options,titan)

        with open(str(pathlib.Path(options.uncertainty.yaml).resolve()),'r') as f: 
            output_dict = yaml.load(f,Loader)['outputs']
        output_names, output_values = report_outputs_from_csv(options.output_folder, output_dict, component_list)

    except Exception as e: 
        log_error(options.output_folder, seed, e)
        raise e
    
    update_datafile(options.output_folder, seed, output_names, output_values)

    return output_names, output_values

def write_datafile(directory, i_sample, state_info):
    with open(directory + '/MCdata_'+str(state_info['seed']).rjust(10,'0'), 'w') as f:
        f.write('''__________________________________________________________________________________
__________________________________________________________________________________
          _        $$$$####| $$##| $$$###| //$#\\\\  $#|   $#|      $#| $#| //$#\\\\ 
        _-#-_         $#|     $#|    $#|   $#| $#| $#\\\\  $#|      $#| $#| $#| $#|  
       /#####\\        $#|     $#|    $#|   $$$###| $#| \\\\$#| $#|  $#| $#| $#| $#|
     /### ? ###\\      $#|     $#|    $#|   $#| $#| $#|   $#|      $#| $#| $#| $#|
__--#############--___$#|____$$##|___$#|___$#|_$#|_$#|___$#|______\\\\$#//__\\\\$#//
_____________________________(Still_in_development!)________________________\\\\$#|_

''')
        f.write('RUN | {} | SEED | {} |\n'.format(i_sample, str(state_info['seed']).rjust(10,'0')))
        i_param = 0
        for param, value in state_info.items():
            if param=='seed': continue
            f.write('PARAMETER #{} - {}:{}\n'.format(i_param, param, value))
            i_param+=1
        f.write('\n----------------------------------------------------------------------------------\n\n')

def update_datafile(directory, seed, output_names, output_values):
     with open(directory + '/MCdata_'+str(seed).rjust(10,'0'), 'a') as f:
         if len(output_names)<1: f.write('NO OUTPUT')
         i_out = 0
         for name, value in zip(output_names, output_values):
             f.write('OUTPUT #{} - {}:{}\n'.format(i_out, name, value))
             i_out+=1

def log_error(directory, seed, error):
    with open(directory + '/MCdata_'+str(seed).rjust(10,'0'), 'a') as f:
        f.write('ERROR - NO OUTPUT GENERATED\n')
        f.write('Exception: {}'.format(error))