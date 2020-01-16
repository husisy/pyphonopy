import yaml
import numpy as np
import periodictable
from lxml import etree

element_number_to_info = {x.number:{'number':x.number, 'symbol':x.symbol, 'mass':x.mass} for x in periodictable.elements}
element_symbol_to_info = {x.symbol:{'number':x.number, 'symbol':x.symbol, 'mass':x.mass} for x in periodictable.elements}


def parse_band_conf(filepath:str):
    with open(filepath, 'r') as fid:
        z0 = [x.strip() for x in fid]
    tmp0 = [x.split('=',1)[1].strip() for x in z0 if x.startswith('DIM')]
    assert len(tmp0)==1
    mutli1,multi2,multi3 = [int(x) for x in tmp0[0].split()]
    return mutli1,multi2,multi3


def parse_phonopy_band_data_yaml(filepath:str, tag_plot:bool=False):
    assert filepath.endswith('.yaml')
    with open(filepath, encoding='utf-8') as fid:
        z0 = yaml.load(fid.read(), Loader=yaml.Loader)
    tmp0 = np.array([[[x3 for x2 in x1['eigenvector'] for x3 in x2] for x1 in x0['band']] for x0 in z0['phonon']])
    yaml_data = {
        'symbol': [x['symbol'] for x in z0['points']],
        'coordinates': [x['coordinates'] for x in z0['points']],
        'mass': [x['mass'] for x in z0['points']],
        'segment_nqpoint': list(z0['segment_nqpoint']),
        'q-position': np.array([x['q-position'] for x in z0['phonon']]),
        'distance': np.array([x['distance'] for x in z0['phonon']]),
        'frequency': np.sort(np.array([[y['frequency'] for y in x['band']] for x in z0['phonon']]), axis=1),
        'group_velocity': np.array([[y['group_velocity'] for y in x['band']] for x in z0['phonon']]),
        'eigenvector': tmp0[:,:,:,0] + tmp0[:,:,:,1]*1j, #(np,complex)
    }
    tmp0 = np.concatenate([[0],np.cumsum(np.array(yaml_data['segment_nqpoint']))-1])
    yaml_data['special_kpoint'] = yaml_data['distance'][tmp0]
    yaml_data['special_kpoint_str'] = [x[0] for x in z0['labels']] + [z0['labels'][-1][-1]]
    if tag_plot:
        import matplotlib.pyplot as plt
        _,ax = plt.subplots(1, 1)
        ax.plot(yaml_data['distance'], yaml_data['frequency'])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for x in yaml_data['special_kpoint']:
            ax.plot([x,x], ylim, color='r')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(yaml_data['special_kpoint'])
        ax.set_xticklabels(['${}$'.format(x) for x in yaml_data['special_kpoint_str']])
    return yaml_data


def parse_vasprun_force_constant(filepath:str):
    '''
    element_info, atom_symbol, force_constant = parse_vasprun_force_constant('vasprun.xml')

    filepath(str)

    (ret0)element_info(dict): (str)->(dict)

    (ret1)atom_symbol(list,str,N0)

    (ret2)force_constant(np,float,(N0,N0,3,3))
    '''
    with open(filepath, 'rb') as fid:
        # XML can only have ONLY one root element, see https://en.wikipedia.org/wiki/Root_element
        # vasprun is not standard XML file
        z1 = etree.HTML(fid.read())
    atom_symbol = [x.strip() for x in z1.xpath('//array[@name="atoms"]/set/rc/c/text()')[::2]]
    num_atom_total = len(atom_symbol)

    tmp0 = [x.xpath('./c/text()') for x in z1.xpath('//array[@name="atomtypes"]/set/rc')]
    tmp1 = [(int(x[0].strip()), x[1].strip(), float(x[2].strip()), float(x[3].strip()), x[4].strip()) for x in tmp0]
    element_info = {x[1]: {'num_atom':x[0], 'mass':x[2], 'valence':x[3], 'pseudopotential':x[4]} for x in tmp1}
    assert num_atom_total==sum(x['num_atom'] for x in element_info.values())

    hf0 = lambda x: [float(y) for y in x.strip().split()]
    hessian_matrix = np.array([hf0(x) for x in z1.xpath('//dynmat/varray[@name="hessian"]/v/text()')])
    assert hessian_matrix.shape==(num_atom_total*3, num_atom_total*3)

    tmp0 = np.array([element_info[x]['mass'] for x in atom_symbol])
    tmp1 = - np.sqrt(tmp0[:,np.newaxis] * tmp0)
    tmp2 = hessian_matrix.reshape((num_atom_total,3,num_atom_total,3)).transpose(0,2,1,3)
    force_constant = tmp1[:,:,np.newaxis,np.newaxis] * tmp2

    return element_info, atom_symbol, force_constant


def parse_vasp_force_constant(filepath:str)->np.ndarray:
    '''
    force_constant = parse_vasp_force_constant('FORCE_CONSTANTS')

    filepath(str)

    (ret)force_constant(np,float,(N0,N0,3,3))
    '''
    with open(filepath, encoding='utf-8') as fid:
        z0 = [x.strip() for x in fid]
    tmp0 = z0[0].split()
    if len(tmp0)==2: #strange, but two file formats do exists
        assert tmp0[0]==tmp0[1]
    num_atom = int(tmp0[0])
    assert len(z0) == (num_atom**2*4 + 1)
    tmp0 = [(str(x),str(y)) for x in range(1,num_atom+1) for y in range(1,num_atom+1)]
    assert all(x==tuple(y.split()) for x,y in zip(tmp0, z0[1::4]))
    hf0 = lambda x: [float(x) for x in x.strip().split()]
    tmp0 = [hf0(x) for x in z0[2::4]+z0[3::4]+z0[4::4]]
    force_constant = np.array(tmp0).reshape((3,-1,3)).transpose(1,0,2).reshape((num_atom,num_atom,3,3))
    return force_constant


def parse_POSCAR(filepath:str):
    '''
    crystal_basis, atom_symbol, atom_position = parse_POSCAR('POSCAR-unitcell')

    filepath(str)

    (ret0)(np,float,(3,3)): crystal_basis

    (ret1)(list,str,N0): atom_symbol

    (ret2)(np,float,(N0,3)): atom_position
    '''
    with open(filepath, encoding='utf-8') as fid:
        z0 = [x.strip() for x in fid]
    ind0 = [ind0 for ind0,x in enumerate(z0) if x.startswith('CONTCAR')][0] + 1
    assert abs(float(z0[ind0])-1)<1e-7, 'absolutely NOT support scale!=1'
    crystal_basis = np.array([[float(y) for y in x.split()] for x in z0[(ind0+1):(ind0+4)]])
    tmp0 = z0[ind0+4].split()
    tmp1 = [int(x) for x in z0[ind0+5].split()]
    atom_symbol = [y for x0,x1 in zip(tmp0,tmp1) for y in [x0]*x1]

    ind0 = [ind0 for ind0,x in enumerate(z0) if x.startswith('Direct')][0] + 1
    atom_position = np.array([[float(y) for y in x.split()] for x in z0[ind0:(ind0+len(atom_symbol))]]) @ crystal_basis
    return crystal_basis, atom_symbol, atom_position
