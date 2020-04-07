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


def test_add__two_matrices_with_default_values__should_be_sum_of_default_values():
    A = sp.sparray((3, 3), default=2.52)
    B = sp.sparray((3, 3), default=7.33)

    C = A + B

    assert C[0, 0] == 2.52 + 7.33
    assert C[0, 2] == 2.52 + 7.33
    assert C[1, 0] == 2.52 + 7.33
    assert C[1, 1] == 2.52 + 7.33
    assert C[2, 2] == 2.52 + 7.33


def test_add__two_custom_matrices__should_be_sum():
    A = sp.sparray((3, 3), default=2.52)
    B = sp.sparray((3, 3), default=7.33)

    A[0, 2] = 1.5
    A[1, 1] = 3.0
    B[0, 0] = 8.3
    B[0, 2] = 0.3

    C = A + B

    assert C[0, 0] == 2.52 + 8.30
    assert C[0, 1] == 2.52 + 7.33
    assert C[0, 2] == 1.80
    assert C[1, 0] == 2.52 + 7.33
    assert C[1, 1] == 3.00 + 7.33
    assert C[1, 2] == 2.52 + 7.33
    assert C[2, 0] == 2.52 + 7.33
    assert C[2, 1] == 2.52 + 7.33
    assert C[2, 2] == 2.52 + 7.33


def test_sub__two_custom_matrics__should_subtract_correctly():
    A = sp.sparray((3, 3), default=4.32)
    B = sp.sparray((3, 3), default=1.10)

    A[0, 0] = 7.83
    B[0, 0] = 10.32

    B[1, 2] = 3.3

    C = A - B

    assert C[0, 0] == 7.83 - 10.32
    assert C[0, 1] == 4.32 - 1.10
    assert C[0, 2] == 4.32 - 1.10
    assert C[1, 0] == 4.32 - 1.10
    assert C[1, 1] == 4.32 - 1.10
    assert C[1, 2] == 4.32 - 3.30
    assert C[2, 0] == 4.32 - 1.10
    assert C[2, 1] == 4.32 - 1.10
    assert C[2, 2] == 4.32 - 1.10


def test_mult__A_times_A_with_default_value__should_square_default_value():
    A = sp.sparray((2, 1), default=7)

    C = A*A

    assert C[0, 0] == 7**2
    assert C[0, 1] == 7**2
    assert C[0, 2] == 7**2
    assert C[1, 0] == 7**2
    assert C[1, 1] == 7**2
    assert C[1, 2] == 7**2
    assert C[2, 0] == 7**2
    assert C[2, 1] == 7**2
    assert C[2, 2] == 7**2


def test_mult__two_custom_matrices__should_multiply_elementwise():
    A = sp.sparray((3, 3), default=1.1)
    B = sp.sparray((3, 3), default=2.5)

    A[0, 0] = 3.5
    A[1, 0] = 5.7
    B[2, 2] = 8.9
    B[0, 0] = 2.1

    C = A * B

    assert C[0, 0] == 3.5 * 2.1
    assert C[0, 1] == 1.1 * 2.5
    assert C[0, 2] == 1.1 * 2.5
    assert C[1, 0] == 5.7 * 2.5
    assert C[1, 1] == 1.1 * 2.5
    assert C[1, 2] == 1.1 * 2.5
    assert C[2, 0] == 1.1 * 2.5
    assert C[2, 1] == 1.1 * 2.5
    assert C[2, 2] == 1.1 * 8.9


def test_div__two_custom_matrics__should_divide_elementwise():
    A = sp.sparray((3, 3), default=1.1)
    B = sp.sparray((3, 3), default=2.5)

    A[0, 0] = 3.5
    A[1, 0] = 5.7
    B[2, 2] = 8.9
    B[0, 0] = 2.1

    C = A / B

    assert C[0, 0] == 3.5 / 2.1
    assert C[0, 1] == 1.1 / 2.5
    assert C[0, 2] == 1.1 / 2.5
    assert C[1, 0] == 5.7 / 2.5
    assert C[1, 1] == 1.1 / 2.5
    assert C[1, 2] == 1.1 / 2.5
    assert C[2, 0] == 1.1 / 2.5
    assert C[2, 1] == 1.1 / 2.5
    assert C[2, 2] == 1.1 / 8.9


def test_mod__default_matrices__should_compute_correctly():
    A = sp.sparray((3, 3))
    B = sp.sparray((3, 3), default=2.0)

    A[0, 0] = 5

    C = A % B

    assert C[0, 0] == 1.0
    assert C[0, 1] == 0.0
    assert C[0, 2] == 0.0
    assert C[1, 0] == 0.0
    assert C[1, 1] == 0.0
    assert C[1, 2] == 0.0
    assert C[2, 0] == 0.0
    assert C[2, 1] == 0.0
    assert C[2, 2] == 0.0


def test_power__default_matrices__should_compute_correctly():
    A = sp.sparray((2, 2), default=2.0)
    B = sp.sparray((2, 2), default=3.0)

    C = A**B

    assert C[0, 0] == 8.0
    assert C[0, 1] == 8.0
    assert C[1, 0] == 8.0
    assert C[1, 1] == 8.0


def test_iadd__default_matrix__should_double_when_add_itself():
    A = sp.sparray((2, 2), default=1.3)

    A += A

    assert A[0, 0] == 2.6
    assert A[0, 1] == 2.6
    assert A[1, 0] == 2.6
    assert A[1, 1] == 2.6


def test_sum__default_matrix__should_compute_sum_correctly():
    A = sp.sparray((2, 2), default=1.3)

    assert A.sum() == 2 * 2 * 1.3


def test_sum__custom_matrix__should_compute_sum_correctly():
    A = sp.sparray((2, 2), default=1.3)

    A[0, 1] = 99.0

    assert A.sum() == (2 * 2 - 1) * 1.3 + 99.0


