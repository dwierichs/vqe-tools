import util 
import pytest
import numpy as np
from projectq.ops import *
from itertools import product

# Test _DERIVATIVES dictionary for correct formatting
@pytest.mark.parametrize("k,v",util._DERIVATIVES.items())
def test_derivative_lookup(k,v):
    assert type(k)==str and type(v)==list
    assert np.all([type(g)==list and len(g)==2 for g in v])
    assert np.all([isinstance(g[0], BasicGate) for g in v])
    assert np.all([type(g[1]) in [complex, float] for g in v])

# Test _LAYERS dictionary for correct formatting
N_set = [4,5]
@pytest.mark.parametrize("item,N",product(util._LAYERS.items(),N_set))
def test_layer_lookup(item,N):
    k, v = item
    assert callable(v)
    assert v.__code__.co_argcount==1
    gates = v(N)
    assert np.all([type(g)==list and len(g)==3 for g in gates])
    for g, qub, par in gates:
        # If the layer contains rotation gates, plug in dummy parameter to get
        # BasicGate
        assert isinstance(g, BasicGate) or isinstance(g(0),BasicGate)
        assert type(qub)==list
        assert type(par)==int or par is None

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

@pytest.mark.parametrize("var_par",var_pars)
def test_group_gates(var_par):
    gate_groups = util.group_gates(gates,param,var_par)
    assert len(gate_groups)==len(var_par)+1
    assert np.sum([len(group) for group in gate_groups])==len(gates)
    assert np.all(gate_groups==correct_groups[str(var_par)])

