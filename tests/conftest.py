import warnings

from pydantic import PydanticDeprecatedSince20
import pytest


@pytest.fixture(autouse=True)
def suppress_external_warnings():
    """
    suppresses specific warnings from external libraries
    (Marker/Surya) that we cannot fix ourselves.
    """
    # Filter only the specific Pydantic warning coming from the 'marker' or 'surya' modules
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="marker.*")

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="surya.*")
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20, module="surya.*")
    warnings.filterwarnings("ignore", message="Support for class-based `config` is deprecated")
