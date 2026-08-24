import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: test hits the network (deselect with -m 'not network')"
    )
