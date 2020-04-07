import sparray as sp

def test_init__default_arguments__should_set_properties_correctly():
    shape = (2, 7, 9)

    s = sp.sparray(shape)

    assert s.shape == (2, 7, 9)
    assert s.ndim == 3
    assert s.dtype == float


def test_init__different_default_value__should_set_for_all_elements():
    x = sp.sparray((3, 3), default=7)
    
    assert x[2, 2] == 7
    assert x[0, 2] == 7
    assert x[1, 1] == 7

