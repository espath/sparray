import sparray as sp

def test_init__default_arguments__should_set_properties_correctly():
    shape = (2, 7, 9)

    s = sp.sparray(shape)

    assert s.shape == (2, 7, 9)
    assert s.ndim == 3
    assert s.dtype == float



