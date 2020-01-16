import os
import numpy as np

from pyphonopy import get_vasp_factor, generate_dynamical_matrix_block, calculate_phonon_spectrum
from utils import parse_phonopy_band_data_yaml, parse_band_conf, parse_POSCAR, parse_vasprun_force_constant, element_symbol_to_info


hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)
hf_interp = lambda A,B,num_point=50: np.stack([np.linspace(x,y,num_point) for x,y in zip(A,B)], axis=1)
hf_data = lambda *x: os.path.join('data', *x)

def test_example00():
    crystal_basis, atom_symbol, atom_position = parse_POSCAR(hf_data('POSCAR-unitcell'))
    multi1,multi2,multi3 = parse_band_conf(hf_data('band.conf'))
    element_info, s_atom_symbol, force_constant = parse_vasprun_force_constant(hf_data('vasprun.xml'))

    # atom_mass = [element_symbol_to_info[x]['mass'] for x in atom_symbol]
    atom_mass = [10.811, 95.96, 95.96] #atom mass used in phonopy, tiny different from periodictable

    vasp_factor = get_vasp_factor()
    symprec = 1e-5
    tmp0 = np.array([[0,0,0],[0,0.5,0],[-0.333,0.667,0],[0,0,0]])
    reciprocal_k = np.concatenate([hf_interp(x,y,40) for x,y in zip(tmp0[:-1], tmp0[1:])])

    reciprocal_basis = np.linalg.inv(crystal_basis).T #reciprocal_basis @ crystal_basis.T = 2 pi I
    tmp0 = reciprocal_k @ reciprocal_basis
    distance_on_path = np.cumsum(np.concatenate([[0], np.sqrt(np.sum((tmp0[1:] - tmp0[:-1])**2, axis=1))]))

    dynamical_matrix_block = generate_dynamical_matrix_block(crystal_basis,
            atom_position, multi1, multi2, multi3, force_constant, atom_mass, symprec=symprec)
    frequency = calculate_phonon_spectrum(dynamical_matrix_block, reciprocal_k) * vasp_factor

    yaml_data = parse_phonopy_band_data_yaml(hf_data('band.yaml'), tag_plot=False) #slow....
    # band.yaml file is generated using the below commands in ./data folder
    #   $ phonopy --fc vasprun.xml
    #   $ phonopy -c POSCAR-unitcell band.conf
    assert hfe(yaml_data['q-position'], reciprocal_k) < 1e-5
    assert hfe(yaml_data['distance'], distance_on_path) < 1e-5
    assert hfe(yaml_data['frequency'], frequency) < 1e-7
