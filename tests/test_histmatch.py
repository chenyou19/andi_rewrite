import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('histmatch',Path(__file__).resolve().parents[1]/'scripts/prepare_histmatch.py')
h=importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def test_affine_target_and_background():
    a=np.linspace(.01,1,600,dtype=np.float32).reshape(2,3,10,10)
    a[0,:,0,:]=0
    target=np.stack([2*h.landmarks(a[:,c][a[:,c]>0])+.3 for c in range(3)])
    b,_=h.transform(a,target)
    assert np.array_equal(b[a==0],a[a==0])
    np.testing.assert_allclose(b[a>0],2*a[a>0]+.3,rtol=1e-6)

def test_ties_do_not_reverse_order():
    q=np.repeat(np.arange(1,5.),[1024,1024,1024,1025])
    x,y=h.mapping(q,np.linspace(.1,2,4097))
    assert len(x)==4 and np.all(np.diff(y)>0)
    a=np.array([1,1,2,4.]);b=np.interp(a,x,y)
    assert b[0]==b[1] and np.all(np.diff(b)>=0)

def test_degenerate_or_invalid_inputs_fail():
    with pytest.raises(ValueError): h.landmarks(np.ones(10))
    with pytest.raises(ValueError): h.landmarks(np.array([1,np.nan]))
    with pytest.raises(ValueError): h.transform(np.full((1,3,2,2),-1.),np.ones((3,4097)))

def test_reference_cdf_has_zero_distance():
    assert h.distances(h.P,np.linspace(.01,2,4097))=={'cdf_max_grid_error':0.,'wasserstein_grid_over_reference_iqr':0.}
