import util 
import pytest
import numpy as np
from projectq.ops import BasicGate

# Test _DERIVATIVES dictionary for correct formatting
@pytest.mark.parametrize("k,v",util._DERIVATIVES.items())
def test_derivative_lookup(k,v):
    assert type(k)==str and type(v)==list
    assert np.all([type(g)==list and len(g)==2 for g in v])
    assert np.all([isinstance(g[0], BasicGate) for g in v])
    assert np.all([type(g[1]) in [complex, float] for g in v])

