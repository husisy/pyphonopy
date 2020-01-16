import itertools
import numpy as np

def hf_map(l,m,n):
    i,j,k,r=1,1,1,-1
    if l==1: i = -1
    if l==0: r = 0
    if m==1: j = -1
    if m==0: r = 1
    if n==1: k = -1
    if n==0: r = 2
    if i*j*k==-1:
        if r==0: i = -1
        if r==1: j = -1
        if r==2: k = -1
    return i,j,k
_STEP3_SET = {(l,m,n) for l,m,n in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]) if ((l,m,n)!=(-1,-1,-1)) and (l*m*n in {-1,0})}
_STEP3_MAP = {x:hf_map(*x) for x in _STEP3_SET}


def set_parameter(lattice, eps):
    '''
    lattice(np,float,(3,3)): each row as a vector, [vec1; vec2; vec3]

    eps(float)

    (ret)(dict): (str)->(float/int)
    '''
    tmp0 = lattice @ lattice.T
    xi = 2*tmp0[1,2]
    eta = 2*tmp0[0,2]
    zeta = 2*tmp0[0,1]
    params = {
        'A': tmp0[0,0],
        'B': tmp0[1,1],
        'C': tmp0[2,2],
        'xi': xi,
        'eta': eta,
        'zeta': zeta,
        'l': -1 if xi<-eps else (1 if xi>eps else 0),
        'm': -1 if eta<-eps else (1 if eta>eps else 0),
        'n': -1 if zeta<-eps else (1 if zeta>eps else 0),
    }
    return params


def _one_step(tmat_i, tmat_old, lattice_old, eps):
    tmat_new = tmat_i @ tmat_old
    lattice_new = tmat_i @ lattice_old
    params = set_parameter(lattice_new, eps)
    return tmat_new, lattice_new, params


def niggli_reduce_full(lattice, eps=1e-5, max_iteration=100):
    '''see http://scripts.iucr.org/cgi-bin/paper?S010876730302186X

    lattice(np,float,(3,3)): each row as a vector, [vec1; vec2; vec3]

    eps(float)

    max_iteration(int)

    (ret0)(bool): tag_success

    (ret1)(np,int,(3,3)): tmat
    '''
    assert lattice.shape==(3,3)
    params = set_parameter(lattice, eps)
    tmat = np.eye(3, dtype=np.int)
    tag_success = False
    for _ in range(max_iteration):
        # j=0 step0
        if (params['A']>(params['B']+eps)) or ((abs(params['A']-params['B'])<=eps) and abs(params['xi'])>(abs(params['eta'])+eps)):
            tmp0 = np.array([[0,-1,0],[-1,0,0],[0,0,-1]])
            tmat, lattice, params = _one_step(tmp0, tmat, lattice, eps)

        # j=1 step1
        if (params['B']>(params['C']+eps)) or ((abs(params['B']-params['C'])<=eps) and abs(params['eta'])>(abs(params['zeta'])+eps)):
            tmp0 = np.array([[-1,0,0],[0,0,-1],[0,-1,0]])
            tmat, lattice, params = _one_step(tmp0, tmat, lattice, eps)
            continue

        # j=2 step2
        if params['l']*params['m']*params['n']==1:
            tmp0 = -1 if params['l']==-1 else 1
            tmp1 = -1 if params['m']==-1 else 1
            tmp2 = -1 if params['n']==-1 else 1
            tmp3 = np.array([[tmp0,0,0],[0,tmp1,0],[0,0,tmp2]])
            tmat, lattice, params = _one_step(tmp3, tmat, lattice, eps)

        # j=3 step3
        if (params['l'],params['m'],params['n']) in _STEP3_SET:
            tmp0 = np.diag(_STEP3_MAP[(params['l'],params['m'],params['n'])])
            tmat, lattice, params = _one_step(tmp0, tmat, lattice, eps)

        # j=4 step4
        if ((abs(params['xi'])>(params['B']+eps)) or
                ((abs(params['B']-params['xi'])<=eps) and ((2*params['eta'])<(params['zeta']-eps))) or
                ((abs(params['B']+params['xi'])<=eps) and (params['zeta']<-eps))):
            tmp0 = -1 if params['xi']>0 else (1 if params['xi']<0 else 0)
            tmp1 = np.array([[1,0,0],[0,1,0],[0,tmp0,1]])
            tmat, lattice, params = _one_step(tmp1, tmat, lattice, eps)
            continue

        # j=5 step5
        if ((abs(params['eta'])>(params['A']+eps)) or
                ((abs(params['A']-params['eta'])<=eps) and ((2*params['xi'])<(params['zeta']-eps))) or
                ((abs(params['A']+params['eta'])<=eps) and (params['zeta']<-eps))):
            tmp0 = -1 if params['eta']>0 else (1 if params['eta']<0 else 0)
            tmp1 = np.array([[1,0,0],[0,1,0],[tmp0,0,1]])
            tmat, lattice, params = _one_step(tmp1, tmat, lattice, eps)
            continue

        # j=6 step6
        if ((abs(params['zeta'])>(params['A']+eps)) or
                ((abs(params['A']-params['zeta'])<=eps) and ((2*params['xi'])<(params['eta']-eps))) or
                ((abs(params['A']+params['zeta'])<=eps) and (params['eta']<-eps))):
            tmp0 = -1 if params['zeta']>0 else (1 if params['zeta']<0 else 0)
            tmp1 = np.array([[1,0,0],[tmp0,1,0],[0,0,1]])
            tmat, lattice, params = _one_step(tmp1, tmat, lattice, eps)
            continue

        # j=7 step7
        tmp0 = params['xi']+params['eta']+params['A']+params['B']
        if (tmp0<eps) or ((abs(tmp0)<=eps) and ((2*params['A']+2*params['eta']+params['zeta'])>eps)):
            tmp1 = np.array([[1,0,0],[0,0,1],[1,1,1]])
            tmat, lattice, params = _one_step(tmp1, tmat, lattice, eps)
            continue
        tag_success = True
        break
    return tag_success,tmat


def niggli_reduce(lattice, eps=1e-5, max_iteration=100):
    '''
    warning: the niggli_reduce() api is DIFFERENT from the niggli_reduce_full() mainly due to the backward compatlibity
        the current pyniggli.niggli_reduce() should be a drop-in replacement for the niggli.niggli_reduce()

    lattice(np,float,(3,3)): each column as a vector, [vec1, vec2, vec3]

    eps(float)

    max_iteration(int)

    (ret0)(np,float,(3,3)): each column as a vector, [vec1, vec2, vec3]
    (ret0)(NoneType): when fail
    '''
    lattice = lattice.reshape(3, 3)
    tag_success,tmat = niggli_reduce_full(lattice.T, eps, max_iteration)
    if tag_success:
        return lattice @ tmat.T
