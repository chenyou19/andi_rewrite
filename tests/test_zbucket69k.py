import importlib.util
from pathlib import Path
import numpy as np
from collections import Counter
s=importlib.util.spec_from_file_location('z',Path(__file__).resolve().parents[1]/'scripts/prepare_zbucket69k.py');z=importlib.util.module_from_spec(s);s.loader.exec_module(z)

def test_bins_cover_once():
    assert [v for a,b in z.BINS for v in range(a,b+1)]==list(range(146))
    assert len(z.BINS)==29

def test_largest_remainder_ties():
    assert z.allocate([1,1,1],5).tolist()==[2,2,1]

def test_draw_balanced_and_reproducible():
    a=z.draw(list(range(7)),25,np.random.default_rng(73))
    assert a==z.draw(list(range(7)),25,np.random.default_rng(73))
    c=Counter(a);assert max(c.values())-min(c.values())<=1 and len(a)==25
    b=z.draw(list(range(7)),4,np.random.default_rng(73));assert len(set(b))==4
