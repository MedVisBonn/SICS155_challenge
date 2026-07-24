def update_from(cfg, ref, inplace=False):
    """
    Minimal utility used by FACT.models.blocks to fill missing cfg fields from a base.
    Works on simple attribute namespaces (objects with __dict__).
    """
    if not inplace:
        import types
        new = types.SimpleNamespace()
        for k, v in ref.__dict__.items():
            setattr(new, k, getattr(cfg, k, v))
        return new
    else:
        for k, v in ref.__dict__.items():
            if not hasattr(cfg, k) or getattr(cfg, k, v) is None:
                setattr(cfg, k, v)
        return cfg