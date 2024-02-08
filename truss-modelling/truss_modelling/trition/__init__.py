try:
    import tritonclient

    del tritonclient
except ImportError as e:
    raise ImportError(
        "Please install `triton` or `triton-cuda` dependency group "
        "to use any utils in this sub-package."
    ) from e
