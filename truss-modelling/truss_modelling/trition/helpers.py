"""Helpers for interacting with nvidia triton inference server."""
from typing import List

import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.utils as triton_utils


def _fill_inputs(
    name: str,
    input_data,
    dtype: np.dtype,
    mutable_inputs: List[triton_grpc.InferInput],
    make_2d: bool = True,
) -> None:
    if input_data is None:
        return

    array_input = np.asarray(input_data, dtype=dtype)
    if make_2d:
        array_input = np.atleast_2d(array_input)
    t = triton_grpc.InferInput(
        name, array_input.shape, triton_utils.np_to_triton_dtype(dtype)
    )
    t.set_data_from_numpy(array_input)
    mutable_inputs.append(t)
