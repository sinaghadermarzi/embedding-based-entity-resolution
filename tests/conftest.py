import socket

import pytest

_network_ok: bool | None = None  # session cache: probe at most once per run


def _network_available() -> bool:
    global _network_ok
    if _network_ok is None:
        try:
            socket.create_connection(("s3.amazonaws.com", 443), timeout=3).close()
            _network_ok = True
        except OSError:
            _network_ok = False
    return _network_ok


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: test hits the network (deselect with -m 'not network')"
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    # network-marked tests auto-skip when the network is unreachable, so a plain
    # offline 'uv run pytest' stays green (the marker still allows -m deselection)
    if item.get_closest_marker("network") and not _network_available():
        pytest.skip("network unreachable (probe to s3.amazonaws.com:443 failed)")
