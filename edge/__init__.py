"""Edge service package."""

from importlib import import_module

app = import_module('edge.app')

__all__ = ['app']
