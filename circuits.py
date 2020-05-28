""" Circuit class based on projectQ and subclasses 
"""
import os
import numpy as np
import projectq as pq
from projectq.ops import *
from projectq.ops import QubitOperator as QuOp
import _add_gates as add_g
import util

class Circuit():
    """ Basic ProjectQ Circuit class 
        - Subclasses have to provide the self.gates variable encoding the
          variational ansatz class This variable is structured as list of lists
          with self.gates[i] = [ gate, qubit_indices, parameter_index]
          where gate is the gate itself, qubit_indices is an iterable with the
          qubit indices the gate acts on and parameter_index gives the id of
          the used parameter for the gate.  The derivative of the gate is found
          automatically (see util._DERIVATIVES)
    """

    def __init__(self, N, verbose=False, drawing=False, bounds=(-0.05,0.05),
            shots=0, sym_translation=False, sym_reflection=False,
            name='Unnamed'):
        """

        Args:
            N (int): number of qubits
            verbose (bool): whether or not to output messages about the circuit
            drawing (bool): whether to draw the circuit instead of computing
            bounds (tuple): lower and upper bound for random initial parameters
            shots (int): number of shots for evaluation
            sym_translation (bool): whether or not the circuit and the 
                                    Hamiltonian are invariant under translation
                                    
            sym_reflection (bool): whether or not the circuit and Hamiltonian
                                   are invariant under reflection. requires 
                                   sym_translation=True 
            name (str): name of the circuit
        Returns:

        Sets:
            N (arg)
            verbose (arg)
            drawing (arg)
            bounds (arg)
            shots (arg)
            sym_translation (arg)
            sym_reflection (arg)
            eng (projectq.MainEngine)
            name (arg)
        Comments:
            draw() requires drawing=True
            sym_reflection=True requires sym_translation=True

        """
        # if we use reflection symmetry, require translation symmetry and N%2=0
        if sym_reflection: 
            assert sym_translation, ('Only use sym_reflection together with '
                'sym_translation')
            assert N%2==0, ('Only use sym_reflection for even N')
        self.N = N
        self.verbose = verbose
        self.drawing = drawing
        self.bounds = bounds
        self.shots = shots
        self.sym_translation = sym_translation
        self.sym_reflection = sym_reflection
        self.name = name
        if drawing:
            self.eng = pq.MainEngine(backend=pq.backends.CircuitDrawer(),
                    engine_list=[])
        else:
            self.eng = pq.MainEngine(backend=pq.backends.Simulator(),
                    engine_list=[])

    def init_param(self, n=None, bounds=None):
        """ Initialize the circuit parameters 

        Args:
            n (int): number of parameters
            bounds (tuple): lower and upper bound for random initial parameters
        Returns:
            param (array): n randomly sampled parameters within bounds
        Sets:
            bounds (arg)
            param (ret)

        """
        if n is None: n = self.n
        if bounds is None: 
            bounds = self.bounds
        else:
            self.bounds = bounds
        np.random.seed(int.from_bytes(os.urandom(4),byteorder='little'))
        self.param = np.random.rand(n)*(bounds[1]-bounds[0])+bounds[0]

        return self.param

    def set_hamiltonian(self, model, **kwargs):
        """ Set Hamiltonian observable to be measured

        Args:
            model (str): name of the system/Hamiltonian
            kwargs (): optional parameters for the Hamiltonian
        Returns:
        Sets:
            model_H (QubitOperator): model Hamiltonian
            measurement_bases (iterable): gatesets to required for measurements
                                          of the model Hamiltonian
            meas_eval (callable): measured strings to energy value decoder
            H_paulis (iterable): pauli decomposition of Hamiltonian as gateset

        """
        # initialize to be able to add QuOps later on
        self.model_H = QuOp('Z0', 0.)

        if model == 'TFI':
            t = kwargs.pop('t', 1.)
            # pauli decomposition as gateset for gradient within fubini
            self.H_paulis = [[add_g.ZZ, -1., [0,1]], [X, -t, [0]]]
            if self.sym_translation:
                self.model_H = QuOp(f'Z0 Z1', -1.)+QuOp(f'X0', -t)
            else:
                for i in range(self.N):
                    self.model_H += QuOp(f'Z{i} Z{(i+1)%self.N}', -1./self.N)
                    self.model_H += QuOp(f'X{i}', -t/self.N)
            # gate sets for sampled measurements
            self.measurement_bases = [[],[[H,i] for i in range(self.N)]]
            def meas_eval(z,x):
                ev = -1/self.N*np.sum([z[i]*z[(i+1)%self.N] for i in \
                    range(self.N)])
                ev += -t/self.N*np.sum(x)
                return ev

        elif model == 'XXZ':
            delta = kwargs.pop('Delta', 1.)
            self.H_paulis = [
                    [add_g.XX, 1., [0,1]],
                    [add_g.YY, 1., [0,1]],
                    [add_g.ZZ, delta, [0,1]],
                    ]
            if self.sym_translation:
                self.model_H = QuOp(f'X0 X1', 1.)+\
                               QuOp(f'Y0 Y0', 1.)+\
                               QuOp(f'Z0 Z0', delta)
            else:
                for i in range(self.N):
                    self.model_H += QuOp(f'X{i} X{(i+1)%self.N}', 1./self.N)
                    self.model_H += QuOp(f'Y{i} Y{(i+1)%self.N}', 1./self.N)
                    self.model_H += QuOp(f'Z{i} Z{(i+1)%self.N}', delta /self.N)
            # We are not using this, so it is a place holder
            meas_eval = None

        elif model == 'J1J2':
            J2 = kwargs.pop('J2',1.)
            self.H_paulis = [
                     [add_g.XX, 1., [0,1]],
                     [add_g.YY, 1., [0,1]],
                     [add_g.ZZ, 1., [0,1]],
                     [add_g.XX, J2, [0,2]],
                     [add_g.YY, J2, [0,2]],
                     [add_g.ZZ, J2, [0,2]],
                     ]
            for i in range(self.N):
                # Nearest neighbour
                self.model_H += QuOp(f'X{i} X{(i+1)%self.N}', 1./self.N)
                self.model_H += QuOp(f'Y{i} Y{(i+1)%self.N}', 1./self.N)
                self.model_H += QuOp(f'Z{i} Z{(i+1)%self.N}', 1./self.N) 
                # Next-to-nearest neighbour
                self.model_H += QuOp(f'X{i} X{(i+2)%self.N}', J2/self.N)
                self.model_H += QuOp(f'Y{i} Y{(i+2)%self.N}', J2/self.N)
                self.model_H += QuOp(f'Z{i} Z{(i+2)%self.N}', J2/self.N)
            # We are not using this, so it is a place holder
            meas_eval = None

        elif model == 'H_2':
            assert self.N == 2, 'N should be 2 for the Hydrogen model.'
            op = ['', 'Z0', 'Z1', 'Z0 Z1', 'Y0 Y1', 'X0 X1']
            g = [0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
            for i in range(len(g)):
                self.model_H += QuOp(op[i], g[i]) 
            self.H_paulis = [
                     [Z, g[1], [0]],
                     [Z, g[2], [1]],
                     [add_g.ZZ, g[3], [0,1]],
                     [add_g.YY, g[4], [0,1]],
                     [add_g.XX, g[5], [0,1]],
                     ]
            # We are not using this, so it is a place holder
            meas_eval = None

        elif model == 'Ham':
            self.model_H = kwargs.pop('H')
            self.H_paulis = kwargs.pop('H_paulis', None)
            # We are not using this, so it is a place holder
            meas_eval = None

        self.model_H.compress()
        self.meas_eval = meas_eval
        if self.verbose: print( f'Appended the {model} expectation values to '
                f'the {self.name} gate. '
                f'Using parameter(s) {kwargs}'*int(len(kwargs)>0) )

        return None


    def _run(self, param=None):
        """ Run the circuit without measurement or expectation value evaluation

        Args:
            param (array): gate parameters, if None use stored ones

        """
        param = param if param is not None else self.param
        # initialize qubit register
        self.qureg = self.eng.allocate_qureg(self.N)
        # iterate over circuit
        for gate, qub, par in self.gates:
            # syntax slightly differs for controlled gates
            if isinstance(gate, ControlledGate):
                # parametrized controlled gate
                if par is not None: 
                    C(gate._gate(param[par])) | (self.qureg[qub[0]], 
                            [self.qureg[i] for i in qub[1:]]) 
                # parameter-free controlled gate
                else: 
                    gate | (self.qureg[qub[0]], 
                            [self.qureg[i] for i in qub[1:]])
            # not a controlled gate
            else:
                # parametrized gate
                if par is not None: 
                    gate(param[par]) | [self.qureg[i] for i in qub] 
                # parametrized-free gate
                else:
                    try:
                        gate | [self.qureg[i] for i in qub] 
                    except IndexError: #? fix!
                        gate | tuple(self.qureg[i] for i in qub)

    def _deallocate(self):
        """ Ending function that measures all qubits and deallocates them.
        This is required by the projectq syntax before finishing in any case.

        """
        All(Measure) | self.qureg
        self.eng.flush(deallocate_qubits=True)

        return None

    def eval(self, param):
        """ Run the circuit and compute the expectation value of the
        Hamiltonian, either exactly or via samples

        Args:
            param (array): circuit parameters
        Returns:
            ev (float): expectation value
        Comments:
            requires self.measurement_bases to be set if self.shots>0.

        """
        try:
            # run the circuit
            self._run(param)

            # compute the exact expectation value
            if self.shots==0:
                ev = self.eng.backend.get_expectation_value(
                        self.model_H, self.qureg)

            # compute the expectation value with samples
            # to be precise, we waste one state reconstruction at the end
            else:
                # secondary engine for buffering the state
                engine_list = []
                buf_eng = pq.MainEngine(pq.backends.Simulator(), engine_list)
                buf_qureg = buf_eng.allocate_qureg(self.N)
                buf_eng.backend.set_wavefunction(self.eng.backend.cheat()[1], 
                    buf_qureg)
                energies = []
                for _ in range(self.shots):
                    # measured strings in various bases
                    measurements = []
                    # iterate through measurement bases
                    for basis in self.measurement_bases:
                        # apply basis change gates
                        for gate, qub in basis:
                            try:
                                gate | [self.qureg[qub]]
                            # ?
                            except IndexError:
                                gate | tuple(self.qureg[qub])
                        # measure qubits
                        All(Measure) | self.qureg
                        self.eng.flush(deallocate_qubits=False)
                        # read out collapsed state
                        measurements.append(
                                [-int(q)*2+1 for q in self. qureg])
                        # restore buffered state for next measurement
                        self.eng.backend.set_wavefunction(
                                buf_eng.backend.cheat()[1], self.qureg)
                    # decode the measured strings into a value of H
                    energies.append(self.meas_eval(*measurements))
                ev = np.mean(energies)
                All(Measure) | buf_qureg
                buf_eng.flush(deallocate_qubits=True)
            self._deallocate() 

        # catch interrupts during circuit evaluation to avoid projectq errors
        except AttributeError as e:
            raise AttributeError(f'{e}\nDid you forget to set the '
                    f'model Hamiltonian via Circuit.set_hamiltonian?')
        except KeyboardInterrupt:
            self._deallocate()
            if self.shots>0:
                All(Measure) | buf_qureg
                buf_eng.flush(deallocate_qubits=True)
            raise KeyboardInterrupt('Derived KeyboardInterrupt '
                    'from Circuit.eval()')

        return ev

    def fubini(self, param, fixed_par=None, gate_groups=None, incl_grad=False):
        """ Compute the Fubini-Study matrix, c.f. fubini.py for readability

        Args:
            param (iterable): circuit parameters
            fixed_par (iterable): list of indices of fixed parameters
            gate_groups (iterable): groups of gates, see output format of 
                                    util.group_gates, computed if not provided
            incl_grad (bool): whether or not to compute the gradient by reusing 
                              circuits
        Returns:
            F (array): Fubini-Study matrix of the circuit at param
            grad (array): gradient at param if requested via incl_grad, 
                          else None

        """
        param = param if param is not None else self.param

        # if there are no fixed parameters, all parameters are varied
        if fixed_par is None:
            n = len(param)
            var_par = list(range(n))
        # else only non-fixed parameters are varied
        else:
            var_par = [i for i in range(len(param)) if i not in fixed_par]
            n = len(var_par)

        # if not given, we can create the gate_groups here
        if gate_groups is None:
            if var_par==[]:
                gate_groups = self.gates
            else:
                gate_groups = util.group_gates(self.gates, param, var_par)

        # initialization
        F = np.zeros((n,n)) # F is a real matrix, c.f. end of function
        der_overlap = np.zeros(n).astype(complex)
        if incl_grad:
            grad = np.zeros(n) # the gradient is real-valued
        else:
            grad = None

        der_ops = []
        for I, i in enumerate(var_par):
            # this is just a lookup for generator gates of the gates
            # that are parametrized with the parameter i:
            der = []
            for gate, qub, par in gate_groups[I+1]:
                # select only the gates containing the derived parameter
                if par==i:
                    for der_op, coeff in util._DERIVATIVES[
                            str(gate).split('(')[0]]:
                        der.append([der_op, coeff, qub])
            der_ops.append(der)

        try:
            self.qureg = self.eng.allocate_qureg(self.N+1)
            ancilla = self.qureg[self.N]
            # secondary engine for buffering
            buf_eng = pq.MainEngine(pq.backends.Simulator(), [])
            buf_qureg = buf_eng.allocate_qureg(self.N+1)

            # begin algorithm described e.g. in 1804.03023v4, fig.5
            H | [ancilla]
            # run through rows of F-matrix
            for I, i in enumerate(var_par):
                # run through gates in the group to be executed _before_ the
                # first gate with current parameter i
                for gate, qub, _ in gate_groups[I]:
                    # apply the gate depending on whether it's controlled
                    if isinstance(gate, ControlledGate): 
                        gate | (self.qureg[qub[0]], 
                                [self.qureg[x] for x in qub[1:]])
                    else:
                        gate | [self.qureg[x] for x in qub] 
                # store state in buffer: applied original circuit up to
                # (excluding) current parameter i - this obviously does not
                # work on a QC
                buf_eng.backend.set_wavefunction(self.eng.backend.cheat()[1],
                        buf_qureg)

                # initialize generator sum, treating constants conveniently
                constant = 0.
                generators = QuOp('Z0', 0.)
                # run over generators of gates parametrized by i and add them
                for k, [der_op, coeff_i, qub] in enumerate(der_ops[I]): 
                    if str(der_op)=='':
                        constant += 1.
                    else:
                        generators += QuOp(' '.join([(f'{str(der_op)[x]}'
                        f'{qub[x]}') for x in range(len(qub))]), 1.)
                            
                    # for translation symmetry there will be m*N terms for some
                    # m in der_i, we only need the first m generators
                    if self.sym_translation and k==len(der_ops[I])//self.N-1:
                        break
                # <\psi|\partial_i\psi> corresponds to the expectation value of
                # the generators in the current state
                der_overlap[I] = constant+coeff_i \
                        *self.eng.backend.get_expectation_value(
                        generators, self.qureg)
                # for translation symmetry we account for the skipped terms
                if self.sym_translation:
                    der_overlap[I] *= self.N

                # run over generators of gates parametrized by i
                for k, [der_op_i, coeff_i, qub_i] in enumerate(der_ops[I]):
                    # apply controlled generator gate according to algorithm
                    C(der_op_i) | (ancilla, [self.qureg[x] for x in qub_i])
                    # run over columns of F-matrix (upper right triangle)
                    for J in range(I, n):
                        j = var_par[J]
                        # run over generators of gates parametrized by j
                        for der_op_j, coeff_j, qub_j in der_ops[J]:
                            # make use of reflection symmetry properties. this 
                            # was tested on layers of one- and two-qubit gates
                            if self.sym_reflection:
                                if len(qub_i)==len(qub_j):
                                    if k>0 and k<self.N//2:
                                        fac = 2.
                                    elif k==0 or k==self.N//2:
                                        fac = 1.
                                    else:
                                        fac = 0
                                elif len(qub_i)==1:
                                    if k>0 and k<=self.N//2:
                                        fac = 2.
                                    else:
                                        fac = 0
                                elif len(qub_i)==2:
                                    if k<self.N//2:
                                        fac = 2.
                                    else:
                                        fac = 0
                            else:
                                fac = 1.

                            # this enables skipping some computations when 
                            # reflection symmetry is exploited
                            if fac>0:
                                # apply the ancilla-controlled generator gate 
                                C(der_op_j)|(ancilla,
                                        [self.qureg[x] for x in qub_j])
                                # obtain the matrix entry via expectation value
                                # on the ancilla
                                exp_val = np.abs(coeff_i)*np.abs(coeff_j)\
                                    *self.eng.backend.get_expectation_value(
                                                QuOp('X0',1.), [ancilla])
                                # for translation symmetry, we just multiply
                                # the contribution by the number of qubits
                                if self.sym_translation:
                                    exp_val *= self.N
                                    # for reflection symmetry we additionally
                                    # skip some terms and in return can include
                                    # the factor here
                                    if self.sym_reflection:
                                        exp_val *= fac

                                # add the contribution for this combination of
                                # generators of the i-parametrized gates and
                                # the j-parametrized gates
                                F[I,J] += exp_val
                                # undo the controlled generator for the
                                # j-parametrized generator in order to reuse
                                # the state - impossible on QC of course
                                C(der_op_j).get_inverse() | (ancilla, 
                                        [self.qureg[x] for x in qub_j])
                            # no need to go on if the parameter influence is
                            # only a translation invariant layer
                            if self.sym_translation:
                                break

                        # run through gate_group attributed to parameter j and
                        # apply all gates, generating the state for the next j
                        for gate, qub, _ in gate_groups[J+1]:
                            if isinstance(gate, ControlledGate): 
                                gate | (self.qureg[qub[0]], 
                                        [self.qureg[x] for x in qub[1:]])
                            else:
                                gate | [self.qureg[x] for x in qub] 
                    # at this point, all columns in the given row have been 
                    # treated, the full circuit (incl. C(idergate)) has been 
                    # applied and we can use this state to produce the energy 
                    # derivative (unlike on a QC) via controlled gates of the 
                    # Hamiltonian Pauli terms, see e.g. 1804.03023v4, fig.5
                    if incl_grad:
                        # run over Paulis in Hamiltonian
                        for h_op, h_coeff, h_qub in self.H_paulis:
                            C(h_op) | (ancilla, [self.qureg[x] for x in h_qub])
                            # compute gradient contribution via EV on ancilla
                            grad[I] += -2*coeff_i.imag*h_coeff\
                                    *self.eng.backend.get_expectation_value(
                                            QuOp('Y0',1.), [ancilla])
                            # undo the current Pauli (self-inverse)
                            C(h_op) | (ancilla, [self.qureg[x] for x in h_qub])

                    # reset the state to before applying the controlled
                    # generator gate corresponding to parameter i
                    self.eng.backend.set_wavefunction(
                            buf_eng.backend.cheat()[1], self.qureg)

            # formalities to make projectq happy: measure the qubits
            All(Measure) | buf_qureg
            buf_eng.flush(deallocate_qubits=True)
            self._deallocate()

            # add second term of Fubini matrix. In rotation gate based ansatze
            # the term will be real as der_overlap will be purely imaginary
            # -> F is a real matrix -> symmetric 
            for I in range(n):
                for J in range(I, n):
                    F[I,J] -= (np.conj(der_overlap[I])*der_overlap[J]).real
                    F[J,I] = F[I,J] 

        # catch interrupts during circuit evaluation to avoid projectq errors
        except KeyboardInterrupt:
            self._deallocate()
            All(Measure) | buf_qureg
            buf_eng.flush(deallocate_qubits=True)
            raise KeyboardInterrupt('Derived KeyboardInterrupt from fubini()')

        return F, grad

    def draw(self, filename=None, param=None):
        """ Output a drawing of the circuit to a tex file 

        Args:
            filename (str): file name to store the latex drawing code
            param (array): gate parameters, if None use stored ones
        Returns:
            tex (str): latex code to draw the circuit
        Comments:
            filename can be given w/o file extension
        """

        assert self.drawing, ('Enable the drawing option to activate the '
                'CircuitDrawer backend.')
        self._run(param)
        self.eng.flush(deallocate_qubits=True)
        tex = self.eng.backend.get_latex()
        if filename is not None:
            # add file ending if missing
            if filename[-4:] != '.tex':
                filename += '.tex'
            # write to file
            with open(filename, 'w') as f:
                f.write(tex)

        return tex


class Custom(Circuit):
    """ Custom circuit subclass """

    def __init__(self, *args, layers, gates=None, initgates=None, **kwargs):
        """ 

        Args:
            layers (iterable): layertypes of the circuit, chosen from 
                               util._LAYERS.keys()
            gates (iterable): gates composing the circuit instead of layers,
                              overriding layers option. Each gate has format 
                              [gate,qubits,parameter index]
            initgates (iterable): prepend some initial gates in order to 
                                  prepare psi_0; these gates are not 
                                  parametrized
        Sets:
            name (str): fixed as Custom Circuit
            layers (arg)
            gates (arg/iterable): either the argument gates or a gate set 
                                  constructed from layers and initgates
            n (int): number of parameters

        """
        self.name = 'Custom Circuit'
        self.layers = layers

        Circuit.__init__(self, *args, **kwargs) 

        # set up gate structure if gates are given
        if gates not in [None, []]:
            self.gates = gates
            par_shift = np.max([g[2] for g in gates if g[2] is not None])+1
        # of if gates are not given via layers
        else:
            par_shift = 0
            self.gates = []
            for L in layers:
                # look up gates for layer and append them to self.gates
                try:
                    # if only None-parametrized gates exist in this layer,
                    # this will lead to no increment in par_shift
                    par_num = -1 
                    layer = util._LAYERS[L]
                    for gate, qub, par in layer(self.N):
                        self.gates.append([gate,qub,
                            (None if par is None else par+par_shift)])
                        if par is not None and par>par_num:
                            par_num = par
                    par_shift += par_num+1
                except KeyError:
                    raise KeyError(f'Unknown gate in custom circuit: {L}.'
                            '\nThese are the available gates:'
                            f'\n{util._LAYERS.keys()}')

        if initgates not in [None, []]:
            self.gates = initgates + self.gates
        # memorize number of parameters in ansatz
        self.n = par_shift
        # initialize parameters
        self.init_param()

        if self.verbose and gates is None: 
            print((f'\nSet up a custom circuit with {self.N} qubits and '
                f'layer structure \n{layers}.\n'))
        elif self.verbose: 
            gate_strings = []
            for gate, qub, par in self.gates:
                gate_strings.append(
                    (f'[{str(gate(par) if par is not None else gate)}, '
                        f'{qub},{par}'))
            print((f'\nSet up a custom circuit with {self.N} qubits and gate '
                f'structure \n{" ".join(gate_strings)}.\n'))
