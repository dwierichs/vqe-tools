# vqe-tools
Here we collect projectq-based circuit simulation tools and optimization routines for variational quantum eigensolvers, which we used for [our paper](https://www.arxiv.org/abs/2004.14666) on the natural gradient descent optimizer.

# Requirements
The code uses standard python packages and *projectQ*, see requirements.txt for tested versions

# Contents
The collection contains multiple modules:
1. Free fermion simulation of the QAOA ansatz on the transverse field Ising model
2. Quantum computing circuits using projectQ as backend. Various useful methods for optimization routines are available
3. Optimizers: Vanilla Gradient Descent (GD), Adam, Quantum Natural Gradient Descent (QNG/NatGrad)
4. A decoupled version of the Fubini-Study matrix implementation in the circuit class. 
   This is the place to look at in order to review our implementation as it is self-contained.
5. Tests are provided for the functionalities via pytest

# Usage
Have a look at the interactive notebooks to explore the functionalities implemented so far. We recommend cloning the repository and starting an own notebook to assure proper rendering of the formulae in the comments.


