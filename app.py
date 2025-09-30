"""Legacy compatibility module exposing edge.app symbols."""

from importlib import import_module

_edge_app = import_module("edge.app")
app = _edge_app.app

__all__ = ["app"]

for _name in dir(_edge_app):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_edge_app, _name)
    if _name not in __all__:
        __all__.append(_name)

_del_names = ["_edge_app", "_name", "import_module"]
for _symbol in _del_names:
    globals().pop(_symbol, None)
