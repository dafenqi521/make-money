"""Pytest configuration — marker registration."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: test that requires live network access to financial APIs",
    )
