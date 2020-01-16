import yaml
import numpy as np
import periodictable
from lxml import etree

hf_interp = lambda A,B,num_point=50: np.stack([np.linspace(x,y,num_point) for x,y in zip(A,B)], axis=1)

element_number_to_info = {x.number:{'number':x.number, 'symbol':x.symbol, 'mass':x.mass} for x in periodictable.elements}
element_symbol_to_info = {x.symbol:{'number':x.number, 'symbol':x.symbol, 'mass':x.mass} for x in periodictable.elements}


# copy from phonopy/structure/atoms.py/line353/atom_data
tmp0 = [
    [0, "X", "X", None],  # 0
    [1, "H", "Hydrogen", 1.00794],  # 1
    [2, "He", "Helium", 4.002602],  # 2
    [3, "Li", "Lithium", 6.941],  # 3
    [4, "Be", "Beryllium", 9.012182],  # 4
    [5, "B", "Boron", 10.811],  # 5
    [6, "C", "Carbon", 12.0107],  # 6
    [7, "N", "Nitrogen", 14.0067],  # 7
    [8, "O", "Oxygen", 15.9994],  # 8
    [9, "F", "Fluorine", 18.9984032],  # 9
    [10, "Ne", "Neon", 20.1797],  # 10
    [11, "Na", "Sodium", 22.98976928],  # 11
    [12, "Mg", "Magnesium", 24.3050],  # 12
    [13, "Al", "Aluminium", 26.9815386],  # 13
    [14, "Si", "Silicon", 28.0855],  # 14
    [15, "P", "Phosphorus", 30.973762],  # 15
    [16, "S", "Sulfur", 32.065],  # 16
    [17, "Cl", "Chlorine", 35.453],  # 17
    [18, "Ar", "Argon", 39.948],  # 18
    [19, "K", "Potassium", 39.0983],  # 19
    [20, "Ca", "Calcium", 40.078],  # 20
    [21, "Sc", "Scandium", 44.955912],  # 21
    [22, "Ti", "Titanium", 47.867],  # 22
    [23, "V", "Vanadium", 50.9415],  # 23
    [24, "Cr", "Chromium", 51.9961],  # 24
    [25, "Mn", "Manganese", 54.938045],  # 25
    [26, "Fe", "Iron", 55.845],  # 26
    [27, "Co", "Cobalt", 58.933195],  # 27
    [28, "Ni", "Nickel", 58.6934],  # 28
    [29, "Cu", "Copper", 63.546],  # 29
    [30, "Zn", "Zinc", 65.38],  # 30
    [31, "Ga", "Gallium", 69.723],  # 31
    [32, "Ge", "Germanium", 72.64],  # 32
    [33, "As", "Arsenic", 74.92160],  # 33
    [34, "Se", "Selenium", 78.96],  # 34
    [35, "Br", "Bromine", 79.904],  # 35
    [36, "Kr", "Krypton", 83.798],  # 36
    [37, "Rb", "Rubidium", 85.4678],  # 37
    [38, "Sr", "Strontium", 87.62],  # 38
    [39, "Y", "Yttrium", 88.90585],  # 39
    [40, "Zr", "Zirconium", 91.224],  # 40
    [41, "Nb", "Niobium", 92.90638],  # 41
    [42, "Mo", "Molybdenum", 95.96],  # 42
    [43, "Tc", "Technetium", 98],  # 43 (mass is from wikipedia)
    [44, "Ru", "Ruthenium", 101.07],  # 44
    [45, "Rh", "Rhodium", 102.90550],  # 45
    [46, "Pd", "Palladium", 106.42],  # 46
    [47, "Ag", "Silver", 107.8682],  # 47
    [48, "Cd", "Cadmium", 112.411],  # 48
    [49, "In", "Indium", 114.818],  # 49
    [50, "Sn", "Tin", 118.710],  # 50
    [51, "Sb", "Antimony", 121.760],  # 51
    [52, "Te", "Tellurium", 127.60],  # 52
    [53, "I", "Iodine", 126.90447],  # 53
    [54, "Xe", "Xenon", 131.293],  # 54
    [55, "Cs", "Caesium", 132.9054519],  # 55
    [56, "Ba", "Barium", 137.327],  # 56
    [57, "La", "Lanthanum", 138.90547],  # 57
    [58, "Ce", "Cerium", 140.116],  # 58
    [59, "Pr", "Praseodymium", 140.90765],  # 59
    [60, "Nd", "Neodymium", 144.242],  # 60
    [61, "Pm", "Promethium", 145],  # 61 (mass is from wikipedia)
    [62, "Sm", "Samarium", 150.36],  # 62
    [63, "Eu", "Europium", 151.964],  # 63
    [64, "Gd", "Gadolinium", 157.25],  # 64
    [65, "Tb", "Terbium", 158.92535],  # 65
    [66, "Dy", "Dysprosium", 162.500],  # 66
    [67, "Ho", "Holmium", 164.93032],  # 67
    [68, "Er", "Erbium", 167.259],  # 68
    [69, "Tm", "Thulium", 168.93421],  # 69
    [70, "Yb", "Ytterbium", 173.054],  # 70
    [71, "Lu", "Lutetium", 174.9668],  # 71
    [72, "Hf", "Hafnium", 178.49],  # 72
    [73, "Ta", "Tantalum", 180.94788],  # 73
    [74, "W", "Tungsten", 183.84],  # 74
    [75, "Re", "Rhenium", 186.207],  # 75
    [76, "Os", "Osmium", 190.23],  # 76
    [77, "Ir", "Iridium", 192.217],  # 77
    [78, "Pt", "Platinum", 195.084],  # 78
    [79, "Au", "Gold", 196.966569],  # 79
    [80, "Hg", "Mercury", 200.59],  # 80
    [81, "Tl", "Thallium", 204.3833],  # 81
    [82, "Pb", "Lead", 207.2],  # 82
    [83, "Bi", "Bismuth", 208.98040],  # 83
    [84, "Po", "Polonium", None],  # 84
    [85, "At", "Astatine", None],  # 85
    [86, "Rn", "Radon", None],  # 86
    [87, "Fr", "Francium", None],  # 87
    [88, "Ra", "Radium", None],  # 88
    [89, "Ac", "Actinium", 227],  # 89 (mass is from wikipedia)
    [90, "Th", "Thorium", 232.03806],  # 90
    [91, "Pa", "Protactinium", 231.03588],  # 91
    [92, "U", "Uranium", 238.02891],  # 92
    [93, "Np", "Neptunium", 237],  # 93 (mass is from wikipedia)
    [94, "Pu", "Plutonium", None],  # 94
    [95, "Am", "Americium", None],  # 95
    [96, "Cm", "Curium", None],  # 96
    [97, "Bk", "Berkelium", None],  # 97
    [98, "Cf", "Californium", None],  # 98
    [99, "Es", "Einsteinium", None],  # 99
    [100, "Fm", "Fermium", None],  # 100
    [101, "Md", "Mendelevium", None],  # 101
    [102, "No", "Nobelium", None],  # 102
    [103, "Lr", "Lawrencium", None],  # 103
    [104, "Rf", "Rutherfordium", None],  # 104
    [105, "Db", "Dubnium", None],  # 105
    [106, "Sg", "Seaborgium", None],  # 106
    [107, "Bh", "Bohrium", None],  # 107
    [108, "Hs", "Hassium", None],  # 108
    [109, "Mt", "Meitnerium", None],  # 109
    [110, "Ds", "Darmstadtium", None],  # 110
    [111, "Rg", "Roentgenium", None],  # 111
    [112, "Cn", "Copernicium", None],  # 112
    [113, "Uut", "Ununtrium", None],  # 113
    [114, "Uuq", "Ununquadium", None],  # 114
    [115, "Uup", "Ununpentium", None],  # 115
    [116, "Uuh", "Ununhexium", None],  # 116
    [117, "Uus", "Ununseptium", None],  # 117
    [118, "Uuo", "Ununoctium", None],  # 118
]
phonopy_element_number_to_info = {x[0]:{'number':x[0], 'symbol':x[1], 'name':x[2], 'mass':x[3]} for x in tmp0}
phonopy_element_symbol_to_info = {x[1]:{'number':x[0], 'symbol':x[1], 'name':x[2], 'mass':x[3]} for x in tmp0}


def parse_band_conf(filepath:str):
    with open(filepath, 'r') as fid:
        z0 = [x.strip() for x in fid]
    tmp0 = [x.split('=',1)[1].strip() for x in z0 if x.startswith('DIM')]
    assert len(tmp0)==1
    multi123 = tuple(int(x) for x in tmp0[0].split())

    tmp0 = [x for x in z0 if x.startswith('BAND =')]
    assert len(tmp0)==1
    band_path = np.array([float(x) for x in tmp0[0].split('=',1)[1].strip().split()]).reshape(-1, 3)
    tmp0 = [x for x in z0 if x.startswith('BAND_POINTS')]
    assert len(tmp0)==1
    num_point = int(tmp0[0].split('=',1)[1].strip())
    reciprocal_k = np.concatenate([hf_interp(x,y,num_point) for x,y in zip(band_path[:-1], band_path[1:])])
    return multi123,reciprocal_k


def parse_phonopy_band_data_yaml(filepath:str, tag_plot:bool=False, with_eigenvector=False):
    assert filepath.endswith('.yaml')
    with open(filepath, encoding='utf-8') as fid:
        z0 = yaml.load(fid.read(), Loader=yaml.Loader)
    yaml_data = {
        'symbol': [x['symbol'] for x in z0['points']],
        'coordinates': [x['coordinates'] for x in z0['points']],
        'mass': [x['mass'] for x in z0['points']],
        'segment_nqpoint': list(z0['segment_nqpoint']),
        'q-position': np.array([x['q-position'] for x in z0['phonon']]),
        'distance': np.array([x['distance'] for x in z0['phonon']]),
        'frequency': np.sort(np.array([[y['frequency'] for y in x['band']] for x in z0['phonon']]), axis=1),
        'group_velocity': np.array([[y['group_velocity'] for y in x['band']] for x in z0['phonon']]),
    }
    if with_eigenvector:
        tmp0 = np.array([[[x3 for x2 in x1['eigenvector'] for x3 in x2] for x1 in x0['band']] for x0 in z0['phonon']])
        yaml_data['eigenvector'] = tmp0[:,:,:,0] + tmp0[:,:,:,1]*1j, #(np,complex)
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
    ind0 = [ind0 for ind0,x in enumerate(z0) if x.startswith('C')][0] + 1 #some old file format
    assert abs(float(z0[ind0])-1)<1e-7, 'absolutely NOT support scale!=1'
    crystal_basis = np.array([[float(y) for y in x.split()] for x in z0[(ind0+1):(ind0+4)]])
    tmp0 = z0[ind0+4].split()
    tmp1 = [int(x) for x in z0[ind0+5].split()]
    atom_symbol = [y for x0,x1 in zip(tmp0,tmp1) for y in [x0]*x1]

    ind0 = [ind0 for ind0,x in enumerate(z0) if x.startswith('Direct')][0] + 1
    atom_position = np.array([[float(y) for y in x.split()] for x in z0[ind0:(ind0+len(atom_symbol))]]) @ crystal_basis
    return crystal_basis, atom_symbol, atom_position
