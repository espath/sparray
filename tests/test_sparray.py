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


def test_getitem_setitem__indices_within_shape__should_work_correctly():
    A = sp.sparray((3, 3))

    A[0, 0] = 10.0
    assert A[0, 0] == 10.0

    A[2, 1] = 47.98
    assert A[2, 1] == 47.98

    A[0, 2] = -1.32
    assert A[0, 2] == -1.32


def test_delitem__indices_within_shape__should_work_correctly():
    s = sp.sparray((25, 14))

    s[5, 2] = 87.5

    del s[5, 2]

    assert s[5, 2] == 0.0
