import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


from litellm.proxy.auth.litellm_license import LicenseCheck


def test_read_public_key_loads_successfully():
    """Ensure public_key.pem is valid PEM with no leading whitespace."""
    license_check = LicenseCheck()
    assert (
        license_check.public_key is not None
    ), "public_key.pem could not be loaded — check for leading whitespace or malformed PEM header"


def test_spark_fork_has_no_license_gate():
    license_check = LicenseCheck()
    license_check.license_str = None
    assert license_check.is_premium() is True
    for data in ({"max_users": 100, "max_teams": 1}, {}, None):
        license_check.airgapped_license_data = data
        assert license_check.is_over_limit(101) is False
        assert license_check.is_team_count_over_limit(2) is False
