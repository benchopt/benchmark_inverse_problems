import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    pass


def check_test_dataset_get_data(benchmark, dataset_class):
    if sys.platform == "darwin":
        pytest.skip(
            "Skipping test_dataset_get_data on MacOS."
        )
