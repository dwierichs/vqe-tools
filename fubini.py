""" Implementation of the Fubini-Study matrix for a circuit given as gate set.
Self-contained version of the code implemented in the Circuit class in
circuits.py
"""
import numpy as np
import _add_gates as add_g
import projectq as pq
from projectq.ops import *
from projectq.ops import QubitOperator as QuOp
import util


""" In the following code, the word gate might refer to either a ProjectQ gate
    instance or a list with the format 
        [pq-gate, qubit indices, parameter index] 
    Assumptions:
        1. If sym_reflection we also have sym_translation (non-trivial)
        2. <\psi|\partial_i\psi> \in i\mathbb{R} \forall i
        3. all generators of each gate parametrized by a given parameter 
           commute with all other gates parametrized by that parameter
    
    """

def fubini(gates, par, N, fixed_par=None, gate_groups=None, incl_grad=False, 
        h_paulis=None, sym_translation=False, sym_reflection=False):
    """ Compute the Fubini-Study matrix

    Args:
        gates (iterable): list of gates composing the circuit with each gate
                          in the format [pq.gate, qubits, parameter index]
        par (iterable): circuit parameters
        N (int): number of qubits
        fixed_par (iterable): list of indices of fixed parameters
        gate_groups (iterable): groups of gates, see output of 
                                util.group_gates() for details
        incl_grad (bool): whether or not to compute the gradient by reusing 
                          circuits
        h_paulis (iterable): pauli decomposition of the Hamiltonian for
                             gradient computation (incl_grad=True)
        sym_translation (bool): whether or not the circuit and the Hamiltonian
                                are invariant under translation
        sym_reflection (bool): whether or not the circuit and the Hamiltonian
                               are invariant under reflection. requires 
                               sym_translation=True and N%2=0 for 
                               implementation reasons (provides only rather 
                               small additional speedup)
    Returns:
        F (array): Fubini-Study matrix of the circuit at par
        grad (array): gradient at par if requested via incl_grad, else None

    """
    # if we use reflection symmetry we require translation symmetry and N%2=0
    if sym_reflection: 
        assert sym_translation, ('Only use sym_reflection together with '
                'sym_translation')
        assert N%2==0, ('Only use sym_reflection for even N')

    # if there are no fixed parameters, all parameters are varied
    if fixed_par is None:
        n = len(par)
        var_par = list(range(n))
    # else only non-fixed parameters are varied
    else:
        var_par = [i for i in range(len(par)) if i not in fixed_par]
        n = len(var_par)

    # if not given, we can create the gate_groups here
    if gate_groups is None:
        if var_par==[]:
            gate_groups = gates
        else:
            gate_groups = util.group_gates(gates, par, var_par)

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
        # that are parametrized with the current parameter i:
        der = []
        for gate, qub, par in gate_groups[I+1]:
            # select only the gates containing the derived parameter
            if par==i:
                for der_op, coeff in util._DERIVATIVES[str(gate).split('(')[0]]:
                    der.append([der_op, coeff, qub])
        der_ops.append(der)

    try:
        engine_list = []
        # main engine including one ancilla qubit
        eng = pq.MainEngine(pq.backends.Simulator(), engine_list)
        qureg = eng.allocate_qureg(N+1)
        ancilla = qureg[N]
        # secondary engine for buffering
        buf_eng = pq.MainEngine(pq.backends.Simulator(), engine_list)
        buf_qureg = buf_eng.allocate_qureg(N+1)

        # begin algorithm described e.g. in 1804.03023v4, fig.5
        H | [ancilla]
        # run through rows of F-matrix
        for I, i in enumerate(var_par):
            # run through gates in the group to be executed _before_ the first
            # gate with current parameter i
            for gate, qub, _ in gate_groups[I]:
                # apply the gate depending on whether it's controlled
                if isinstance(gate, ControlledGate): 
                    gate | (qureg[qub[0]], [qureg[x] for x in qub[1:]])
                else:
                    gate | [qureg[x] for x in qub] 
            # store state in buffer: applied original circuit up to (excluding)
            # current parameter i - this obviously does not work on a QC
            buf_eng.backend.set_wavefunction(eng.backend.cheat()[1], buf_qureg)

            # initialize generator sum, treating constants conveniently
            constant = 0.
            generators = QuOp('Z0', 0.)
            # run over generators of gates parametrized by i and add them
            for k, [der_op, coeff_i, qub] in enumerate(der_ops[I]): 
                if str(der_op)=='':
                    constant += 1.
                else:
                    generators += QuOp(' '.join(
                        [f'{str(der_op)[x]}{qub[x]}' for x in range(len(qub))]
                        ), 1.)
                # for translation symmetry there will be m*N terms for some m
                # in der_i, we only need the first m generators
                if sym_translation and k==len(der_ops[I])//N-1:
                    break
            # <\psi|\partial_i\psi> corresponds to the expectation value of the
            # generators in the current state
            der_overlap[I] = coeff_i*eng.backend.get_expectation_value(
                    generators, qureg
                    )+constant
            # for translation symmetry we need to account for the skipped terms
            if sym_translation:
                der_overlap[I] *= N

            # run over generators of gates parametrized by i
            for k, [der_op_i, coeff_i, qub_i] in enumerate(der_ops[I]):
                # apply controlled generator gate according to the algorithm
                C(der_op_i) | (ancilla, [qureg[x] for x in qub_i])
                # run over columns of F-matrix (upper right triangle)
                for J in range(I, n):
                    j = var_par[J]
                    # run over generators of gates parametrized by j
                    for der_op_j, coeff_j, qub_j in der_ops[J]:
                        # make use of reflection symmetry properties. this 
                        # was tested on layers of one- and two-qubit gates only
                        if sym_reflection:
                            if len(qub_i)==len(qub_j):
                                if k>0 and k<N//2:
                                    fac = 2.
                                elif k==0 or k==N//2:
                                    fac = 1.
                                else:
                                    fac = 0
                            elif len(qub_i)==1:
                                if k>0 and k<=N//2:
                                    fac = 2.
                                else:
                                    fac = 0
                            elif len(qub_i)==2:
                                if k<N//2:
                                    fac = 2.
                                else:
                                    fac = 0
                        else:
                            fac = 1.

                        # this enables skipping some computations when 
                        # reflection symmetry is exploited
                        if fac>0:
                            # apply the ancilla-controlled generator gate 
                            C(der_op_j) | (ancilla, [qureg[x] for x in qub_j])
                            # obtain the matrix entry via expectation value on 
                            # the ancilla
                            exp_val = np.abs(coeff_i)*np.abs(coeff_j)\
                                    *eng.backend.get_expectation_value(
                                            QuOp('X0',1.), [ancilla])
                            # for translation symmetry, we just multiply the 
                            # contribution by the number of qubits
                            if sym_translation:
                                exp_val *= N
                                # for reflection symmetry we additionally skip
                                # some terms and in return can include the 
                                # factor here
                                if sym_reflection:
                                    exp_val *= fac

                            # add the contribution for this combination of 
                            # generators of the i-parametrized gates and the 
                            # j-parametrized gates
                            F[I,J] += exp_val
                            # undo the controlled generator for the 
                            # j-parametrized generator in order to reuse the 
                            # state - impossible on QC of course
                            C(der_op_j).get_inverse() | (ancilla, 
                                    [qureg[x] for x in qub_j])
                        # no need to go on if the parameter influence is only a
                        # translation invariant layer
                        if sym_translation:
                            break

                    # run through gate_group attributed to parameter j
                    # and apply all gates, generating the state for the next j
                    for gate, qub, _ in gate_groups[J+1]:
                        if isinstance(gate, ControlledGate): 
                            gate | (qureg[qub[0]], [qureg[x] for x in qub[1:]])
                        else:
                            gate | [qureg[x] for x in qub] 
                # at this point, all columns in the given row have been 
                # treated, the full circuit (incl. C(idergate)) has been 
                # applied and we can use this state to produce the energy 
                # derivative (unlike on a QC) via controlled gates of the 
                # Hamiltonian Pauli terms, see e.g. 1804.03023v4, fig.5
                if incl_grad:
                    # run over Paulis in Hamiltonian
                    for h_op, h_coeff, h_qub in h_paulis:
                        # apply the current Pauli if it's not the identity
                        if (not isinstance(h_op, QuOp) 
                                or h_op.terms != {():1.0}):
                            C(h_op) | (ancilla, [qureg[x] for x in h_qub])
                        # compute contribution to gradient via EV on ancilla
                        grad[I] += -2*coeff_i.imag*h_coeff\
                                *eng.backend.get_expectation_value(
                                        QuOp('Y0',1.), [ancilla])
                        # undo the current Pauli (self-inverse)
                        if (not isinstance(h_op, QuOp) 
                                or h_op.terms != {():1.0}):
                            C(h_op) | (ancilla, [qureg[x] for x in h_qub])

                # reset the state to before applying the controlled generator
                # gate corresponding to parameter i
                eng.backend.set_wavefunction(buf_eng.backend.cheat()[1], qureg)

        # formalities to make projectq happy: measure the qubits
        All(Measure) | qureg
        eng.flush(deallocate_qubits=True)
        All(Measure) | buf_qureg
        buf_eng.flush(deallocate_qubits=True)

        # add second term of Fubini matrix. In rotation gate based ansatze
        # the term will be real as der_overlap will be purely imaginary
        # -> F is a real matrix -> symmetric 
        for I in range(n):
            for J in range(I, n):
                F[I,J] -= (np.conj(der_overlap[I])*der_overlap[J]).real
                F[J,I] = F[I,J] 

    # In case of KeyboardInterrupt, exit properly by deallocating the quregs
    except KeyboardInterrupt:
        All(Measure) | qureg
        eng.flush(deallocate_qubits=True)
        All(Measure) | buf_qureg
        buf_eng.flush(deallocate_qubits=True)
        raise KeyboardInterrupt('Derived KeyboardInterrupt from fubini()')

    return F, grad

