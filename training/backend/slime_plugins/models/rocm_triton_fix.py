"""ROCm/CDNA Triton fix for the fla GatedDeltaNet kernels.

On ROCm (gfx942 / MI300X), Triton software pipelining (``num_stages > 1``) in the
flash-linear-attention (fla) GatedDeltaNet autotune configs triggers a HIP
illegal memory access (``hipErrorIllegalAddress``, code 700) during autotune
benchmarking of ``recompute_w_u_fwd_kernel`` and friends. The GDN inference path
(sglang's own triton kernels) is unaffected; only the fla training kernels crash.

Clamping ``num_stages`` to 1 for every Triton autotune ``Config`` on ROCm avoids
the bad pipelined configs while remaining functionally correct (``num_stages`` is
a performance knob, not a correctness one). This module MUST be imported BEFORE
``fla.ops.*`` so the ``@triton.autotune`` decorators build clamped configs.

No-op on CUDA/NVIDIA.
"""

import torch

try:
    import triton

    if getattr(torch.version, "hip", None) is not None and not getattr(
        triton.Config, "_rocm_num_stages_clamped", False
    ):
        _orig_config_init = triton.Config.__init__

        def _clamped_config_init(self, *args, **kwargs):
            num_stages = kwargs.get("num_stages", None)
            if num_stages is not None and num_stages > 1:
                kwargs["num_stages"] = 1
            _orig_config_init(self, *args, **kwargs)

        triton.Config.__init__ = _clamped_config_init
        triton.Config._rocm_num_stages_clamped = True
except Exception:
    # Triton unavailable or API changed: leave the default behavior untouched.
    pass
