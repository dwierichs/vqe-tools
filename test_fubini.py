import pytest
import util
import fubini
import tfi_ff
import numpy as np
from projectq.ops import H, X, Y, Z, Rx, Ry, Rz, Rxx, Ryy, Rzz, BasicGate, CX
from _add_gates import XX, YY, ZZ
from itertools import product

def gen_tfi_specs(N, p, y_pos, t):
    """ Generate standard gate set, random parameters and Paulis for
    Hamiltonian of the TFI model 
    
    """
    gates = [[H,[i],None] for i in range(N)]
    shift = 0
    for j in range(2*p):
        if j in y_pos:
            gates += [[Ry,[i],j+shift] for i in range(N)]
            shift += 1
        if j%2==0:
            gates += [[Rzz,[i,(i+1)%N],j+shift] 
                    for i in range(N)]
        else:
            gates += [[Rx,[i],j+shift] for i in range(N)]
    par = np.random.random(2*p+len(y_pos))
    paulis = [[ZZ,-1.,[0,1]],[X,-t,[0]]]

    return  gates, par, paulis

def gen_yama_specs(alpha=0.4, beta=0.2):
    """ Generate gate set, random parameters and paulis as well as benchmark
    results for 'H_2' example in Yamamoto [1909.05074]

    """
    par = np.random.random(4)
    c = np.cos(par); s = np.sin(par)
    # analytic gradient and Fubini matrix
    grad = np.array([
        alpha*(-c[2]*s[0]-s[2]*s[1]*c[0]-c[3]*c[1]*s[0])+\
        beta*(c[3]*c[2]*c[0]-c[3]*s[2]*s[1]*s[0]),
        alpha*(-s[2]*c[1]*s[0]-c[3]*s[1]*c[0]-s[3]*c[1])+\
        beta*(c[3]*s[2]*c[1]*c[0]-s[3]*s[2]*s[1]),
        alpha*(-s[2]*c[0]-c[2]*s[1]*s[0])+\
        beta*(-c[3]*s[2]*s[0]+c[3]*c[2]*s[1]*c[0]+s[3]*c[2]*c[1]),
        alpha*(-s[3]*c[1]*c[0]-c[3]*s[1])+\
        beta*(-s[3]*c[2]*s[0]-s[3]*s[2]*s[1]*c[0]+c[3]*s[2]*c[1])
        ])
    F = 1/4*np.array([[1,0,s[1],0],
                       [0,1,0,c[0]],
                       [s[1],0,1,-s[0]*c[1]],
                       [0,c[0],-s[0]*c[1],1]])
    # gates in the "H_2" ansatz
    gates = [[Ry,[0],0],[Ry,[1],1],[CX,[0,1],None],[Ry,[0],2],[Ry,[1],3]]
    # Hamiltonian Pauli decomposition
    paulis = [[Z, alpha, [0]],[Z, alpha, [1]],[XX, beta, [0,1]]]

    return gates, par, paulis, grad, F

# test group_gates function
param = np.random.random(4)
# gates to be grouped
gates = [[H,[0],None],[Rx,[2],0],[CX,[1,0],None],[Ry,[1],0],
         [Ry,[0],1],[CX,[0,1],None],[Rxx,[1,2],2],[Rx,[1],3]]
# scenarios of keeping some parameters fixed
var_pars = [[0,1,2,3], [0,2], [0,1], [3]]
# correct groupings by index
correct_indices = {
        str([0,1,2,3]): [[0],[1,2,3],[4,5],[6],[7]],
        str([0,2]): [[0],[1,2,3,4,5],[6,7]],
        str([0,1]): [[0],[1,2,3],[4,5,6,7]],
        str([3]): [[0,1,2,3,4,5,6],[7]],
        }
# correct gate groups with plugged in parameters where possible
correct_groups = {str(var_par): [[
    [(gates[i][0](param[gates[i][2]]) if gates[i][2] is not None
    else gates[i][0]), gates[i][1], gates[i][2]] 
    for i in group] 
    for group in grouping] 
    for var_par, grouping in correct_indices.items()}

# test fubini regarding its correctness on analytically known examples
N_tfi = 4; t=0.5
tfi_gates, tfi_par, tfi_paulis = gen_tfi_specs(N_tfi, N_tfi//2, [], t)
_, tfi_g, tfi_F = tfi_ff.eval_all(tfi_par, t=t)  # interdependence to be removed

models = [
        # Yamamoto "H_2" example
        [2, *gen_yama_specs()],
        # transverse field Ising model example
        [N_tfi, tfi_gates, tfi_par, tfi_paulis, tfi_g, tfi_F],
        ]

@pytest.mark.parametrize("model", models)
def test_fubini_correctness(model):
    N, gates, par, paulis, out_g, out_F = model
    F, g = fubini.fubini(gates, par, N, incl_grad=True, h_paulis=paulis)
    assert np.allclose(F, out_F)
    assert np.allclose(g, out_g)

# Test fubini with symmetries, e.g. on the TFI
y_posis = [[], [0], [0,2]]
var_pars = [[0,1,2,3], [0,2], [0,1], [3]]

@pytest.mark.parametrize("N, p, y_pos, var_par", 
        product([4,5], [2,3], y_posis, var_pars))
def test_fubini_symmetry(N, p, y_pos, var_par):
    gates, param, paulis = gen_tfi_specs(N, p, y_pos, 0.5)
    gate_groups = util.group_gates(gates, param, var_par)
    print(gates, gate_groups)
    fixed_par = [j for j in range(2*p+len(y_pos)) if j not in var_par]
    Fs = []; gs = []
    for groups, incl_grad, sym_translation, sym_reflection \
        in product([None, gate_groups], [True,False], 
            [False, True],[False,True]):
        try:
            F, g = fubini.fubini(gates, param, N, fixed_par, 
                groups, incl_grad, paulis, sym_translation,
                sym_reflection)
        except AssertionError:
            assert N%2==1 or int(sym_reflection)>int(sym_translation)
        Fs.append(F)
        gs.append(g)
        assert np.all(np.shape(F)==(len(var_par),len(var_par)))
        assert np.allclose(F,Fs[0]), f'F={F},F0={Fs[0]}'
        assert g is None or np.allclose(g,gs[0]), f'g={g}\ng0={gs[0]}'


# test the function against itself with varied options
#   additional qubits, and precomputing the gate_groups should not change
#   anything. incl_grad should not change F.
# arbitrary Hamiltonian
h_paulis = [[XX, -1/2, [0,1]],[Y, -1., [2]],[ZZ, -3/4, [2,3]]]
# test systems sizes
Ns = [4,5]
@pytest.mark.parametrize("var_par, gates, N, gate_groups, incl_grad", 
        product(var_pars, [gates], Ns, [None, correct_groups],
            [True,False]))
def test_fubini_options(var_par, gates, N, gate_groups, incl_grad):
    fixed_par = [j for j in range(4) if j not in var_par]
    if gate_groups is not None:
        gate_groups = gate_groups[str(var_par)]
    F0, g0 = fubini.fubini(gates, param, Ns[0], fixed_par, gate_groups,
                True, h_paulis)
    F, g = fubini.fubini(gates, param, N, fixed_par, gate_groups,
                incl_grad, h_paulis)
    assert np.allclose(F,F0)
    assert g is None or np.allclose(g,g0), f'g={g}\ng0={g0}'

