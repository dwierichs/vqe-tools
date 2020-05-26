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
