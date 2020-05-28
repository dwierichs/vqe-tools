import circuits
import _add_gates as add_g
from itertools import product
import util
from projectq.ops import *
import fubini

import pytest

N_set = [3,4]
bool_set = [True,False]
bounds_set = [(0.,1.),(-3, -2.4)]
shots_set = [0, 1, 2, 10]
model_set = [
        ['TFI',{'t':0.3}],
        ['XXZ',{'Delta':2.1}],
        ['J1J2',{'J2':-0.2}],
        ['H_2',{}],
        ['Ham',{'H':QubitOperator('X0 Z1',0.2)-QubitOperator('X2 Y0',1.4)}],
        ['Ham',{'H':QubitOperator('Z0 Z2',0.2), 
                'H_paulis': [[add_g.ZZ,0.2,[0,2]]]}],
        ]
gates_set = [
        [[[X,[0],None],[Ry,[2],0],[H,[1],None],[Rx,[0],0],[Rxx,[0,1],1]],2],
        [[[Rx,[0],0],[Ry,[2],1],[Rzz,[0,2],2],[Rx,[0],3],[Rxx,[0,1],4]],5],
        [[],0],
        ]
layers_set = [
        ['Rx','Ryy','Rz','Rxx'],
        [],
        ['Rzz','Ry'],
        ]
initgates_set = [
        [],
        None,
        [[Rx(np.pi/3),[0],None],[Y,[1],None],[add_g.ZZ,[0,2],None]],
        ]

@pytest.mark.parametrize("specs", product(N_set, bool_set, bool_set, 
    bounds_set, shots_set, bool_set, bool_set, gates_set, model_set))
def test_circuit_base_class(specs):
    N, verbose, drawing, bounds, shots, sym_trans, \
        sym_reflect, [gates, n], [model, model_pars] = specs
    try:
        circ = circuits.Circuit(N, verbose, drawing, bounds, shots, 
            sym_trans, sym_reflect)
    except AssertionError:
        assert int(sym_reflect)>int(sym_trans) or (N%2==1 and sym_reflect)
    else:
        for init_bounds in [None, bounds]:
            circ.init_param(n, init_bounds)
        if model!='H_2' or N==2:
            circ.set_hamiltonian(model, **model_pars)
        else:
            return None
        # This part has to be changed once there are meas_eval functions added
        # to other Hamiltonians
        if model=='TFI':
            assert circ.meas_eval([-1]*N, [0]*N)==-1.
            assert circ.meas_eval([0]*N, [1]*N)==-model_pars.pop('t',1.)
        else:
            assert circ.meas_eval is None
        circ.gates = gates
        circ._run(circ.param)
        circ._deallocate()
        if drawing:
            tex = circ.draw()
            tex = circ.draw('test.tex')
        else:
            if callable(circ.meas_eval) or shots==0:
                circ.eval(circ.param)
                circ.eval(None)

            if circ.H_paulis is None:
                incl_grad = False
            else:
                incl_grad = True
            # The testing of fubini is not very nice because it uses another
            # utility, namely fubini. 
            # If this test fails without having changed circuits.py, 
            # you probably made changes to fubini.py! 
            F1, grad1 = circ.fubini(circ.param, incl_grad=incl_grad)
            F2, grad2 = fubini.fubini(circ.gates, circ.param, N,
                    incl_grad=incl_grad, h_paulis=circ.H_paulis, 
                    sym_translation=sym_trans, sym_reflection=sym_reflect)
            assert np.allclose(F1,F2)
            if incl_grad:
                assert np.allclose(grad1,grad2), f'{grad1}\n{grad2}'

@pytest.mark.parametrize("specs", product(N_set, layers_set, gates_set,
    initgates_set, shots_set, bounds_set, bool_set, bool_set))
def test_custom_subclass(specs):
    N, layers, [gates, n], initgates, shots, bounds, \
            sym_trans, sym_reflect = specs
    try:
        circ = circuits.Custom(N, layers=layers, gates=gates, 
            initgates=initgates, shots=shots, sym_translation=sym_trans, 
            sym_reflection=sym_reflect, bounds=bounds)
    except AssertionError:
        assert int(sym_reflect)>int(sym_trans) or (N%2==1 and sym_reflect)
    else:
        circ._run(circ.param)
        circ._deallocate()

