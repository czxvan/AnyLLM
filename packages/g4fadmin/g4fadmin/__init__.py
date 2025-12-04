"""
G4FAdmin - GPT4Free Provider and Model Management Tool

Quickly find which provider/model combinations in g4f actually work

Core Features:
  - Provider scanning and recommendation
  - Model listing and searching
  - Authentication detection
  - Real API testing
  - Batch testing and export

Usage Example:
    >>> from g4fadmin import G4FAdmin
    >>> admin = G4FAdmin()
    >>> admin.print_summary()  # View summary
    >>> providers = admin.get_recommended_providers(5)  # Get recommendations
    >>> success, resp, time = admin.test_provider("ApiAirforce", "gpt-4")  # Test
"""

from .admin import G4FAdmin, ProviderInfo, ModelInfo, TestResult, AuthType

__version__ = "1.0.0"
__all__ = ["G4FAdmin", "ProviderInfo", "ModelInfo", "TestResult", "AuthType"]
