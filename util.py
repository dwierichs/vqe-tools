""" Utility functions 
"""
import _add_gates as add_g
from projectq.ops import *

# {parametrized gate: generators} mapping
_DERIVATIVES = {
                'Rx'   : [[X, -1j/2]],
                'Ry'   : [[Y, -1j/2]],
                'Rz'   : [[Z, -1j/2]],
                'Rxx'  : [[add_g.XX,-1j/2]],
                'Ryy'  : [[add_g.YY,-1j/2]],
                'Rzz'  : [[add_g.ZZ,-1j/2]],
                }
# {layer name: lambda N->gates} mapping
# Structure of layer-lambda output: <<list of <Gate,qubits,par-index>>, gate count, depth, 2-qubit gate count>
_LAYERS = {     
    # Parameter free one-qubit gates
    'X': lambda N: [[X,[i],None] for i in range(N)],
    'Y': lambda N: [[Y,[i],None] for i in range(N)],
    'Z': lambda N: [[Z,[i],None] for i in range(N)],
    'H': lambda N: [[H,[i],None] for i in range(N)],
    'T': lambda N: [[T,[i],None] for i in range(N)],

    # parameter free two-qubit gates
    'CX': lambda N: [[CX,[i,(i+1)%N],None] for i in range(N)],

    ## parametrized one-qubit gates
    #  dense layer
    'Ph': lambda N: [[Ph,[i],0] for i in range(N)],
    'Rx': lambda N: [[Rx,[i],0] for i in range(N)],
    'Ry': lambda N: [[Ry,[i],0] for i in range(N)],
    'Rz': lambda N: [[Rz,[i],0] for i in range(N)],
    'R' : lambda N: [[R,[i],0] for i in range(N)], 

    #  sparse layer only acting on even sites
    'Rx2': lambda N: [[Rx,[i*2],0] for i in range(N//2)],
    'Ry2': lambda N: [[Ry,[i*2],0] for i in range(N//2)],
    'Rz2': lambda N: [[Rz,[i*2],0] for i in range(N//2)],

    #  sparse layer only acting on odd sites
    'Rx3': lambda N: [[Rx,[i*2+1],0] for i in range(N//2)],
    'Ry3': lambda N: [[Ry,[i*2+1],0] for i in range(N//2)],
    'Rz3': lambda N: [[Rz,[i*2+1],0] for i in range(N//2)],

    ## parametrized two-qubit gates
    #  dense layer
    'Rxx': lambda N: [[Rxx, [i,(i+1)%N], 0] for i in range(N)], 
    'Ryy': lambda N: [[Ryy, [i,(i+1)%N], 0] for i in range(N)], 
    'Rzz': lambda N: [[Rzz, [i,(i+1)%N], 0] for i in range(N)], 
    #  reuse previous variational parameters
    'Ryy*': lambda N: [[Ryy, [i,(i+1)%N], -1] for i in range(N)], 
    }

def group_gates(gates, param, var_par):
    """ Groups the gates of a circuit (without reordering them) into blocks 
        connected to a varied parameter.
        Fixed parameters are treated like unparametrized gates. 
        All parameters of callable gates are plugged in.

    Args:
        gates (iterable): list of gates composing the circuit with each gate
                          in the format [pq.gate, qubits, parameter index]
        param (iterable): parameters for the circuit
        var_par (iterable): indices of parameters that are to be varied
    Returns:
        gate_groups (list): list of groups of gates with gates in the same
                            format as in the input. gate_groups[I] contains
                            the gates to be executed before applying the 
                            var_par[I]-th derivative operator. Parameters 
                            are plugged into all gates.
    Comments:
        Warning! This function assumes that 
          - gates[i][2]>=gates[j][2] if i>j
          - gates[i][0], gates[j][0] commute if gates[i][2]==gates[j][2]

    """
    # initialization
    gate_groups = []; I = 0; i = var_par[0]; group = []
    # run through gates in circuit
    for gate, qub, j in gates:
        # if the gate is parametrized by the current (non-fixed) parameter
        if j==i:
            # write all _previous_ gates to the previous parameter gate group
            gate_groups.append(group)
            # reset collection of gates for current parameter
            group = []
            # advance the parameter index
            I += 1; 
            # if the parameter index exceeds the parameter list we arrived at 
            # gates that are executed after the first gate that depends on the 
            # last parameter, correspondingly we only will append all gates and
            # don't need to advance in the parameter list anymore
            if I==len(var_par):
                # this is never matched
                i = -1
            # else we assign the new parameter index to i
            else:
                i = var_par[I]

        # append gate if not parametrized
        if not callable(gate):
            group.append([gate, qub, j])
        # plug in parameter into controlled gate and append
        elif isinstance(gate, ControlledGate):
            group.append([C(gate._gate(param[j])), qub, j])
        # plug in parameter into gate and append
        else:
            group.append([(gate(param[j])), qub, j])

    # the last gate group collects all gates from the first occurence of the 
    # last parameter onwards
    gate_groups.append(group)

    return gate_groups
