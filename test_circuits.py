import circuits
import _add_gates as add_g
from itertools import product
import util
from projectq.ops import *

import pytest

#layers = ['Rx','Ry','CX','H','Ry']
#initg = [[H,[0],None],[X,[3],None],[CX,[3,4],None]]
#circ = circuits.Custom(layers=layers,initgates=initg, bounds=(0.,1.),
        #verbose=True, N=6)
#print(circ.gates)
#print(circ.param)
#circ.set_hamiltonian('TFI', t=1.)
#print(circ.eval(circ.param))
#print(circ.fubini(circ.param,incl_grad=True))

N_set = [3,4]
bool_set = [True,False]
bounds_set = [(0.,1.),(-3, -2.4)]
shots_set = [0, 1, 2, 10]
model_set = [
        ['TFI',{}],['TFI',{'t':1.}],
        ['XXZ',{'Delta':2.1}],['XXZ',{'Delta':0}],
        ['J1J2',{}],['J1J2',{'J2':-0.2}],
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
@pytest.mark.parametrize("specs", product(N_set, bool_set, bool_set, 
    bounds_set, shots_set, bool_set, bool_set, gates_set))
def test_circuit_base_class(specs):
    N, verbose, drawing, bounds, shots, sym_trans, \
        sym_reflect, [gates, n] = specs
    try:
        circ = circuits.Circuit(N, verbose, drawing, bounds, shots, 
            sym_trans, sym_reflect)
    except AssertionError:
        assert int(sym_reflect)>int(sym_trans)
    else:
        for init_bounds in [None, bounds]:
            circ.init_param(n, init_bounds)
        for model, model_pars in model_set:
            if model!='H_2' or N==2:
                circ.set_hamiltonian(model, **model_pars)
            if model=='TFI':
                assert circ.meas_eval([1]*N, [0]*N)==-1.
            else:
                assert circ.meas_eval()==None
        circ.gates = gates
        circ._run(circ.param)
        circ._deallocate()
        if drawing:
            tex = circ.draw()
            tex = circ.draw('test.tex')
        else:
            if model=='TFI' or shots==0:
                circ.eval(circ.param)
                circ.eval(None)





