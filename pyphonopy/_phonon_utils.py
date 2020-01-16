import itertools
import numpy as np

from ._niggli_utils import niggli_reduce_full
from ._utils import make_supercell

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))


def verify_supercell_symmetry(force_constant, multi1, multi2, multi3, primitive_atom_symbol=None):
    '''
    force_constant(np,float,(N0,N0,3,3))

    multi1(int)

    multi2(int)

    multi3(int)

    primitive_atom_symbol(list,str,(N1,))
    primitive_atom_symbol(NoneType): optional
    '''
    multi123 = multi1*multi2*multi3
    assert force_constant.ndim==4
    N0 = len(force_constant)
    assert N0%multi123==0 and force_constant.shape==(N0,N0,3,3)
    num_atom = N0 // multi123
    if primitive_atom_symbol is None:
        primitive_atom_symbol = [f'atom{x}-' for x in range(num_atom)]
    assert len(primitive_atom_symbol)==num_atom

    tmp0 = force_constant.transpose(0,2,1,3).reshape(N0*3,N0*3)
    assert hfe(tmp0, tmp0.T) < 1e-7, 'force_constant should be symmetrical'
    index_group = [(x,y,z) for z in range(multi3) for y in range(multi2) for x in range(multi1)]
    s_atom_str = [f'{x}({y0},{y1},{y2})' for x in primitive_atom_symbol for y0,y1,y2 in index_group]
    index_group_to_index = {y:list(range(x,N0,multi123)) for x,y in enumerate(index_group)}
    force_constant_dict = {(k0,k1):force_constant[v0][:,v1] for k0,v0 in index_group_to_index.items()
            for k1,v1 in index_group_to_index.items()}

    hf0 = lambda x,y: ((y[0]-x[0])%multi1, (y[1]-x[1])%multi2, (y[2]-x[2])%multi3)
    tmp0 = [(hf0(x1,y1),x1,y1) for x0,x1 in enumerate(index_group) for y0,y1 in enumerate(index_group)]
    all_delta_group = {x:[z[1:] for z in y] for x,y in itertools.groupby(sorted(tmp0, key=lambda x:x[0]), key=lambda x:x[0])}
    error_message = f'the atom order "atom-(x,y,z)" in force_constant should be: {s_atom_str}'
    for value in all_delta_group.values():
        if len(value)==1:
            continue
        tmp0 = force_constant_dict[value[0]]
        assert max(hfe(tmp0,force_constant_dict[x]) for x in value[1:])<1e-7, error_message



def generate_dynamical_matrix_block(crystal_basis, atom_position, multi1, multi2, multi3,
            force_constant, atom_mass, primitive_atom_symbol=None, symprec=1e-5, tag_verify=True):
    '''
    crystal_basis(np,float,(3,3))

    atom_position(np,float,(N0,3))

    multi1(int)

    multi2(int)

    multi3(int)

    force_constant(np,float,(N1,N1,3,3)): N1=N0*multi1*multi2*multi3

    atom_mass(list,float,(N0))

    primitive_atom_symbol(list,str,(N1,))
    primitive_atom_symbol(NoneType): optional

    symprec(float): optional

    tag_verify(bool): optional

    (ret)(dict): dynamical_matrix_block
        (%key)(tuple,int,3)
        (%value)(np,float,(N0*3,N0*3))
    '''
    multi123 = multi1*multi2*multi3
    num_atom = len(atom_position)
    if tag_verify: # maybe time-consuming
        verify_supercell_symmetry(force_constant, multi1, multi2, multi3, primitive_atom_symbol) # TODO
    s_index_group,s_crystal_basis,s_atom_position = make_supercell(multi1, multi2, multi3, crystal_basis, atom_position)
    s_atom_position_fraction = s_atom_position @ np.linalg.inv(s_crystal_basis)
    ind_primitive = np.arange(num_atom)*multi1*multi2*multi3

    tag_successs,trans_mat = niggli_reduce_full(s_crystal_basis, eps=symprec)
    assert tag_successs
    delta_niggli = np.rint(np.dot(s_atom_position_fraction, np.linalg.inv(trans_mat))).astype(np.int) @ trans_mat
    delta_niggli = delta_niggli - delta_niggli[np.repeat(ind_primitive, multi123, axis=0)]

    #TODO strange, see phonopy/structure/cells.py/get_smallest_vectors
    tmp0 = np.array(list(itertools.product([-1,0,1], [-1,0,1], [-1,0,1], [-1,0,1])))
    lattice_points = np.unique(tmp0 @ np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]), axis=0) @ trans_mat

    indI = []
    indJ = []
    scale_factor = []
    svec_int = []
    tmp0 = (s_atom_position_fraction - delta_niggli) @ s_crystal_basis
    tmp1 = s_atom_position_fraction[ind_primitive] @ s_crystal_basis
    tmp2 = lattice_points @ s_crystal_basis
    tmp3 = lattice_points * np.array([multi1,multi2,multi3])
    tmp4 = delta_niggli * np.array([multi1,multi2,multi3])
    tmp5 = np.zeros((num_atom,1,1), dtype=np.int)
    for ind0 in range(len(tmp0)):
        distance = np.sqrt(np.sum((tmp0[ind0] - tmp1[:,np.newaxis] + tmp2)**2, axis=2))
        ind1 = distance < (distance.min(axis=1, keepdims=True)+symprec)
        indI.append(np.ones(ind1.sum(), dtype=np.int)*ind0)
        indJ.append(np.nonzero(ind1)[0])
        scale_factor.append(np.array([1/x for x in ind1.sum(axis=1) for y in range(x)]))
        svec_int.append((s_index_group[ind0] + tmp3 - tmp4[ind0] - tmp5)[ind1])
    indI = np.concatenate(indI)
    indJ = np.concatenate(indJ)
    scale_factor = np.concatenate(scale_factor)
    svec_int = np.concatenate(svec_int)
    ## same code but use more memory
    # tmp0 = (s_atom_position_fraction - delta_niggli) @ s_crystal_basis
    # tmp1 = s_atom_position_fraction[ind_primitive] @ s_crystal_basis
    # tmp2 = lattice_points @ s_crystal_basis
    # distance = np.sqrt(np.sum((tmp0[:,np.newaxis,np.newaxis] - tmp1[np.newaxis,:,np.newaxis] + tmp2)**2, axis=3))
    # ind0 = distance < (distance.min(axis=2, keepdims=True)+symprec)
    # tmp3 = lattice_points*np.array([multi1,multi2,multi3])
    # tmp4 = (delta_niggli*np.array([multi1,multi2,multi3]))[:,np.newaxis,np.newaxis]
    # svec_int = (s_index_group[:,np.newaxis,np.newaxis] - np.zeros((num_atom,1,1),dtype=np.int) + tmp3 - tmp4)[ind0]
    # indI,indJ,_ = np.nonzero(ind0)
    # scale_factor = [1/x for x in np.sum(ind0,axis=2).reshape(-1) for y in range(x)]

    tmp1 = [(tuple(x0),x1,x2,x3) for x0,x1,x2,x3 in zip(svec_int,indI,indJ,scale_factor)]
    z0 = {key:[y[1:] for y in x] for key,x in itertools.groupby(sorted(tmp1, key=lambda x: x[0]), key=lambda x: x[0])}
    tmp0 = np.repeat(np.array(atom_mass), 3, axis=0)
    mass_factor = np.sqrt(tmp0[:,np.newaxis]*tmp0)
    ret = dict()
    for key,value in z0.items():
        tmp0 = [[np.zeros((3,3)) for _ in range(num_atom)] for _ in range(num_atom)]
        for indI,indJ,factor in value:
            tmp0[indI//multi123][indJ] = tmp0[indI//multi123][indJ] + force_constant[indI,indJ*multi123]*factor
        ret[tuple(-x for x in key)] = np.block(tmp0) / mass_factor
    # below should give the same result
    # ret = dict()
    # for key,value in z0.items():
    #     tmp0 = [[np.zeros((3,3)) for _ in range(num_atom)] for _ in range(num_atom)]
    #     for indI,indJ,factor in value:
    #         tmp0[indJ][indI//multi123] = tmp0[indJ][indI//multi123] + force_constant[indJ*multi123,indI]*factor
    #     ret[key] = np.block(tmp0)
    assert max(hfe(ret[k],ret[tuple(-x for x in k)].T) for k in ret.keys())<1e-7, 'dynamical matrix is NOT symmetry'
    return ret
