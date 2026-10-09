"""Minimal stand-in for pointcept.models.builder: registries are no-op decorators here."""
class _Registry:
    def __init__(self, name): self.name = name; self.module_dict = {}
    def register_module(self, name=None, module=None, force=False):
        def _reg(cls):
            self.module_dict[name or cls.__name__] = cls; return cls
        return _reg(module) if module is not None else _reg
MODELS = _Registry("models"); MODULES = _Registry("modules")
