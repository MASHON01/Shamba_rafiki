"""
Test suite for the Shamba Rafiki backend.

Marks `tests` as a package so test module names are unique across the
parallel unit/, integration/, and api/ sub-packages (avoiding pytest
import-file-mismatch on same-named files). pytest discovers and runs
these via the configuration in pytest.ini; conftest.py holds the
shared fixtures.
"""