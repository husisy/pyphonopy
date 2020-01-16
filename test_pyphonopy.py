import os
import numpy as np

from pyphonopy import get_vasp_factor, generate_dynamical_matrix_block, calculate_phonon_spectrum
from utils import parse_phonopy_band_data_yaml, parse_band_conf, parse_POSCAR, parse_vasprun_force_constant
from utils import element_symbol_to_info, phonopy_element_symbol_to_info


hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def general_task(directory, use_phonopy_mass=True, use_force_constant_full=True):
    hf_file = lambda *x: os.path.join(directory, *x)
    crystal_basis, atom_symbol, atom_position = parse_POSCAR(hf_file('POSCAR-unitcell'))
    element_info, s_atom_symbol, force_constant = parse_vasprun_force_constant(hf_file('vasprun.xml'))

    if use_phonopy_mass: #tiny difference between phonopy from periodictable
        atom_mass = [phonopy_element_symbol_to_info[x]['mass'] for x in atom_symbol]
    else:
        atom_mass = [element_symbol_to_info[x]['mass'] for x in atom_symbol]

    vasp_factor = get_vasp_factor()
    symprec = 1e-5
    (multi1,multi2,multi3),reciprocal_k = parse_band_conf(hf_file('band.conf'))

    reciprocal_basis = np.linalg.inv(crystal_basis).T #reciprocal_basis @ crystal_basis.T = 2 pi I
    tmp0 = reciprocal_k @ reciprocal_basis
    distance_on_path = np.cumsum(np.concatenate([[0], np.sqrt(np.sum((tmp0[1:] - tmp0[:-1])**2, axis=1))]))

    if not use_force_constant_full:
        force_constant = force_constant[:,::(multi1*multi2*multi3)]
    dynamical_matrix_block = generate_dynamical_matrix_block(crystal_basis,
            atom_position, multi1, multi2, multi3, force_constant, atom_mass, symprec=symprec)
    frequency = calculate_phonon_spectrum(dynamical_matrix_block, reciprocal_k) * vasp_factor

    yaml_data = parse_phonopy_band_data_yaml(hf_file('band.yaml'), tag_plot=False) #slow....
    # band.yaml file is generated using the below commands in ./data folder
    #   $ phonopy --fc vasprun.xml
    #   $ phonopy -c POSCAR-unitcell band.conf
    assert hfe(yaml_data['q-position'], reciprocal_k) < 1e-5
    assert hfe(yaml_data['distance'], distance_on_path) < 1e-5
    assert hfe(yaml_data['frequency'], frequency) < 1e-7


def test_data_ws00():
    general_task(os.path.join('data', 'ws00'))
    general_task(os.path.join('data', 'ws00'), use_force_constant_full=False)

def test_data_ws01():
    general_task(os.path.join('data', 'ws01'))
    general_task(os.path.join('data', 'ws01'), use_force_constant_full=False)

def test_data_ws02():
    general_task(os.path.join('data', 'ws02'))
    general_task(os.path.join('data', 'ws02'), use_force_constant_full=False)
