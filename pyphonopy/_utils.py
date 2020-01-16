import numpy as np


def make_supercell(multi1, multi2, multi3, crystal_basis, atom_position, atom_symbol=None):
    '''
    multi1(int)

    multi2(int)

    multi3(int)

    crystal_basis(np,float,(3,3))

    atom_position(np,float,(N0,3))

    atom_symbol(list,str)
    atom_symbol(NoneType): optional

    (ret0)(list,(tuple,int),(N0,)): s_index_group

    (ret1)(np,float,(3,3)): s_crystal_basis

    (ret2)(np,float,(N1,3)): s_atom_position

    (ret3)(list,str,(N1)): s_atom_symbol, optional
    '''
    hf0 = lambda x: isinstance(x,int) and x>0
    assert hf0(multi1) and hf0(multi2) and hf0(multi3)
    tmp0 = np.array([(x,y,z) for z in range(multi3) for y in range(multi2) for x in range(multi1)])
    tmp1 = tmp0 @ crystal_basis
    num_atom = len(atom_position)
    s_index_group = np.tile(tmp0,(num_atom,1))
    s_atom_position = (atom_position[:,np.newaxis,:] + tmp1).reshape(-1,3)
    s_crystal_basis = crystal_basis * np.array([multi1,multi2,multi3])[:,np.newaxis]
    if atom_symbol is None:
        return s_index_group, s_crystal_basis, s_atom_position
    else:
        s_atom_symbol = [x for x in atom_symbol for y in range(multi1*multi2*multi3)]
        return s_index_group, s_crystal_basis, s_atom_position, s_atom_symbol


def get_vasp_factor()->float:
    '''
    (ret)(float)
    '''
    # vasp_unit see phonopy/interface/calculator.py/get_default_physical_units
    # length unit: Angstrom
    # force constant unit: eV/Angstrom^2
    EV = 1.60217733e-19  # [J]
    AMU = 1.6605402e-27  # [kg]
    # PlanckConstant = 4.13566733e-15  # [eV s]
    Angstrom = 1.0e-10   # [m]
    vasp_to_THZ = float(np.sqrt(EV/AMU)/Angstrom/(2*np.pi)/1e12)  # [THz] 15.633302
    return vasp_to_THZ
