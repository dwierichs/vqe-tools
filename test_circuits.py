import circuits
import util
from projectq.ops import *

layers = ['Rx','Ry','CX','H','Ry']
initg = [[H,[0],None],[X,[3],None],[CX,[3,4],None]]
circ = circuits.Custom(layers=layers,initgates=initg, bounds=(0.,1.),
        verbose=True, N=6)
#print(circ.gates)
print(circ.param)
circ.set_hamiltonian('TFI', t=1.)
print(circ.eval(circ.param))
print(circ.fubini(circ.param,incl_grad=True))
