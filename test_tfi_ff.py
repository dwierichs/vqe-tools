import pytest
import numpy as np
import tfi_ff


# cases for rotation matrix tests
Rz_cases = [(0,np.eye(2)),(np.pi/2,[[-1j,0],[0,1j]])]
Rax_cases = [(0,0,np.eye(2)),(np.pi/2,0,[[-1j,0],[0,1j]]),
    (0,np.pi/4,np.eye(2)),
    (np.pi/3,2*np.pi/3,[[0.5+0.4330127j,-0.75],[ 0.75,0.5-0.4330127j]])
    ]
# analytic solution for TFI-QAOA energy:
def tfi_energy(par, N, t=0.5):
    c = np.cos(2*par)
    s = np.sin(2*par)
    if len(par)==0:
        E = -t
    if len(par)==2:
        if N==2:
            E = -s[0]*s[1]-t*c[0]
        elif N==3:
            E = -s[0]*s[1]/2-np.sin(par[0])**2*np.sin(par[1])**2\
                    -t*np.cos(par[0])**2
    if len(par)==4:
        if N==2:
            E = -c[0]*s[2]*s[3]-s[0]*(c[1]*c[2]*s[3]+s[1]*c[3]-t*c[1]*s[2])\
                    -t*c[0]*c[2]
    return E

def tfi_grad(par, N, t=0.5):
    c = np.cos(2*par)
    s = np.sin(2*par)

    if len(par)==0:
        g = np.array([])
    elif len(par)==2:
        if N==2:
            g = 2*np.array([-c[0]*s[1]+t*s[0],-s[0]*c[1]])
        elif N==3:
            g = np.array([-c[0]*s[1]-2*np.cos(par[0])*np.sin(par[0])\
                    *np.sin(par[1])**2+2*t*np.sin(par[0])*np.cos(par[0]),
                    -s[0]*c[1]-2*np.sin(par[0])**2*np.cos(par[1])*\
                            np.sin(par[1])])
    elif len(par)==4:
        if N==2:
            g = 2*np.array([s[0]*s[2]*s[3]+t*s[0]*c[2]\
                    -c[0]*(c[1]*c[2]*s[3]+s[1]*c[3]-t*c[1]*s[2]),
                    -s[0]*(-s[1]*c[2]*s[3]+c[1]*c[3]+t*s[1]*s[2]),
                    -c[0]*c[2]*s[3]-s[0]*(-c[1]*s[2]*s[3]-t*c[1]*c[2])\
                            +t*c[0]*s[2],
                    -c[0]*s[2]*c[3]-s[0]*(c[1]*c[2]*c[3]-s[1]*s[3])])
    return g

def tfi_fubini(par, N, t=0.5):
    c = np.cos(2*par)
    s = np.sin(2*par)

    if len(par)==0:
        F = np.array([])
    elif len(par)==2:
        if N==2:
            F = np.array([[1,0],[0,s[0]**2]])
        elif N==3:
            F = np.array([[3,-3*np.sin(par[0])**2],
                [-3*np.sin(par[0])**2,
                    3+6*np.cos(par[0])**2-9*np.cos(par[0])**4]])/4

    elif len(par)==4:
        if N==2:
            f21 = -0.5*np.sin(4*par[0])*s[1]
            f22 = -s[1]**2*s[0]**2+1
            f31 = 1/4*(np.cos(4*par[0]-2*par[2])-np.cos(4*par[0]+2*par[2]))\
                    *c[1]\
                    -1/4*(np.cos(4*par[0]-2*par[2])+np.cos(4*par[0]+2*par[2]))\
                    +0.5*c[2]
            f32 = -s[0]*(c[0]*c[2]*s[1]-np.sin(4*par[1])*s[2]*s[0]/2)
            f33 = 1-1/4*(-c[1]
                    *(np.cos(2*(par[0]-par[2]))-np.cos(2*(par[0]+par[2])))\
                    +(np.cos(2*(par[0]-par[2]))+np.cos(2*(par[0]+par[2]))))**2
            F = np.array([[1.,0.,c[1],s[1]*s[2]],
                          [0.,np.sin(2*par[0])**2,f21,f31],
                          [c[1],f21,f22,f32],
                          [s[1]*s[2],f31,f32,f33]])
    return F

par1 = np.random.random(2)
par2 = np.random.random(4)
energy_cases = [([],0),([],1),([],2),([],3),
        (par1,0),(par1,1),(par1,2),(par1,3),
        (par2,2)]


@pytest.mark.parametrize("theta, Rz", Rz_cases)
def test_Rot_z(theta, Rz):
    assert np.allclose(tfi_ff.Rot_z(theta), Rz)

@pytest.mark.parametrize("theta, alpha, Rax", Rax_cases)
def test_Rot_z(theta, alpha, Rax):
    assert np.allclose(tfi_ff.Rot_ax(theta,np.cos(alpha),np.sin(alpha)), Rax)

@pytest.mark.parametrize("par, N", energy_cases)
def test_eval_energy(par, N):
    t = 0.5
    try:
        E = tfi_ff.eval_energy(par, N, t)
    except AssertionError:
        assert N<2
    else:
        assert np.allclose(tfi_energy(par, N, t), E)

@pytest.mark.parametrize("par, N", energy_cases)
def test_eval_all(par,N):
    t = 0.5
    try:
        E, g, F = tfi_ff.eval_all(par, N, t)
    except AssertionError:
        assert N<2
    else:
        dim = 2*(len(par)//2)
        print(np.round(np.abs(F-tfi_fubini(par,N,t)),5))
        assert type(E)==np.float64 and g.shape==(dim,) and F.shape==(dim,dim)
        assert np.allclose(tfi_energy(par, N, t), E)
        assert np.allclose(tfi_grad(par, N, t), g)
        assert np.allclose(tfi_fubini(par, N, t), F)
