""" Transverse field Ising model functionalities for fast free fermion
simulation """
import numpy as np

# For convenience we define rotation matrices locally here
def Rot_z(theta):
    """ Rotation matrix about the Z axis for a qubit

    Args:
        theta (float): rotation angle
    Returns:
        _ (array): rotation matrix

    """
    return np.array([[np.exp(-1j*theta), 0],
                     [0, np.exp(1j*theta)]])

def Rot_ax(theta, C, S):
    """ Rotation of a qubit about the axis C*Z + S*Y
    Args:
        theta (float): rotation angle
        C (float): cosine of the angle between z-axis and rotation axis 
        S (float): sine of the angle between z-axis and rotation axis 
    Returns:
        _ (array): rotation matrix

    """
    return np.array([[np.cos(theta)-1j*np.sin(theta)*C, -S*np.sin(theta)],
                     [S*np.sin(theta), np.cos(theta)+1j*np.sin(theta)*C]])

# Pauli matrices
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]]).astype(complex)

def eval_energy(par, N=None, t=1.):
    """ Evaluate the energy for the QAOA TFI circuit

    Args:
        par (iterable): parameters for the circuit
        N (int): number of qubits
        t (float): transverse field for model
    Returns:
        energy (float): energy at par
    Raises:
    Comments:
        - Note that the number of qubits is found via the size of the parameter
          input if it is not given
        - For the derivation see e.g. our paper 2004.14666, appendix, where the
          same notation is used but which is *not* original content
          
    """
    # initialization: number of qubits, fermions and parameters
    if N is None: N = len(par)
    assert N>1, 'Computations with 1 or less qubits not supported'
    # number of blocks with 2 layers each
    p = len(par)//2 
    # number of fermions
    K = N//2
    # compute angle between the z-axis and the rotation axis for ZZ-layers 
    # per fermion
    if N%2==0:
        Thetas = [(2*k+1)/N*np.pi for k in range(K)]
    else:
        Thetas = [(2*k+2)/N*np.pi for k in range(K)]
    # precompute the sine and cosine of these angles
    C = np.cos(Thetas)
    S = np.sin(Thetas)
    # precompute all operators appearing in the circuit, there are p*(K+1)
    ops = {layer: [Rot_ax(par[layer], C[k], S[k]) for k in range(K)] \
            if layer%2==0 \
            else [Rot_z(par[layer])] for layer in range(2*p)}
    energy = 0.

    # iterate through the fermions
    for k in range(K): 
        state = np.array([1.,0.]).astype(complex)
        for i in range(p):
            # evolve the state with ZZ layer
            state = np.dot(ops[2*i][k], state)
            # evolve the state with X layer
            state = np.dot(ops[2*i+1][0], state)
        # set Hamiltonian
        H = -2*(Z*(C[k]+t)+Y*S[k])
        # energy is EV of Hamiltonian in fully state
        energy += np.vdot(state, np.dot(H, state)).real
    # add offset energy for odd N
    if N%2==1:
        energy -= 1+t
    # rescale energy and gradient to enery per site
    energy /= N

    return energy

def eval_all(par, N=None, t=1.):
    """ Evaluate the energy, the gradient and the fubini matrix for the 
        QAOA TFI circuit

    Args:
        par (iterable): parameters for the circuit
        N (int): number of qubits
        t (float): transverse field for model
    Returns:
        energy (float): energy at par
        grad (array): gradient at par
        F (array): Fubini-Study matrix at par
    Raises:
    Comments:
        - Note that the number of qubits is found via the size of the parameter
          input if N is not given
        - For the derivation see e.g. our paper 2004.14666, appendix

    """
    # initialization: number of qubits, fermions and parameters
    if N is None: N = len(par)
    assert N>1, 'Computations with 1 or less qubits not supported'
    # number of blocks with 2 layers each
    p = len(par)//2 
    # number of fermions
    K = N//2
    # compute angle between the z-axis and the rotation axis for ZZ-layers 
    # per fermion
    if N%2==0:
        Thetas = [(2*k+1)/N*np.pi for k in range(K)]
    else:
        Thetas = [(2*k+2)/N*np.pi for k in range(K)]
    # precompute the sine and cosine of these angles
    C = np.cos(Thetas)
    S = np.sin(Thetas)
    # precompute all operators appearing in the circuit, there are p*(K+1)
    ops = {layer: [Rot_ax(par[layer], C[k], S[k]) for k in range(K)] \
            if layer%2==0 \
            else [Rot_z(par[layer])] for layer in range(2*p)}
    # initialize dictionary of all required states and output quantities
    # the dictionary uses the keys {fermion}_{derivative parameter index}
    states = {}
    energy = 0
    grad = np.zeros(2*p)
    F = np.zeros((2*p,2*p)) 
    # initialize helper overlap 
    der_overlaps = np.zeros((2*p, K))

    # iterate through the fermions
    for k in range(K): 
        state = np.array([1.,0.]).astype(complex)
        for i in range(2*p):
            if i%2==0:
                # apply generator of derivative parameter
                # Warning: we skip a factor -1j here
                der_state = np.dot(C[k]*Z+S[k]*Y, state) 
                # evolve derivative-free state
                state = np.dot(ops[i][k], state)
            else:
                # apply generator of derivative parameter
                # Warning: we skip a factor -1j here
                der_state = np.dot(Z, state)
                # evolve derivative-free state
                state = np.dot(ops[i][0], state)
            # evolve derivative state 
            for j in range(i, 2*p):
                if j%2==0:
                    der_state = np.dot(ops[j][k], der_state)
                else:
                    der_state = np.dot(ops[j][0], der_state)
            states[f'{k}_{i}'] = der_state
        # store fully evolved, derivative-free state
        states[f'{k}'] = state

        # set Hamiltonian
        H = -2*(Z*(C[k]+t)+Y*S[k])
        # act with it on the fully evolved state
        ham_state = np.dot(H, states[f'{k}'])
        # energy is EV of Hamiltonian in fully evolved derivative-free state
        energy += np.vdot(states[f'{k}'], ham_state).real

        for i in range(2*p):
            # overlaps between derivative state and derivative-free state.
            # they are real because we skipped -1j above and they would be 
            # purely imaginary
            der_overlaps[i,k] = np.vdot(states[f'{k}'],states[f'{k}_{i}']).real
            # gradient as overlap of derivative state and Hamiltonian acting on
            # derivative-free state
            # take imaginary part instead of real part because of the left out
            # factor -1j
            grad[i] += 2*np.vdot(ham_state, states[f'{k}_{i}']).imag

    # sum derivative overlaps, as they form the second term of F (missing -1j)
    der_overlap = np.sum(der_overlaps, axis=1)

    # construct F
    for i in range(2*p):
        for j in range(i,2*p):
            # first term of F
            # those terms where the indices of the direct sums coincide
            for k1 in range(K):
                # overlaps miss a factor 1j and a factor -1j, which cancel
                F[i,j] += np.vdot(states[f'{k1}_{i}'], 
                        states[f'{k1}_{j}']).real

            # those terms where the indices of the direct sums differ 
                for k2 in range(K):
                    if k1!=k2:
                        # again the missing factors cancel
                        F[i,j] += der_overlaps[i,k1]*der_overlaps[j,k2]
            # second term of F, yet again the missing factors cancel
            F[i,j] -= der_overlap[i]*der_overlap[j] 

        # F is a hermitian matrix, but in this circuit it is real
        for j in range(i):
            F[i,j] = F[j,i]

    # add offset energy for odd N
    if N%2==1:
        energy -= 1+t
    # rescale energy and gradient to enery per site
    energy /= N
    grad /= N

    return energy, grad, F

