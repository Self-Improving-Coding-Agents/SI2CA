# the file to manage all sglang deps in the megatron actor
try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except (ImportError, AttributeError):
    # These fp8/ue8m0 quantization helpers are only used for fp8-quantized weight sync;
    # bf16 training never calls them. On some ROCm images the transitive sglang->aiter
    # import races and raises AttributeError ("module 'aiter' has no attribute 'dtypes'")
    # instead of ImportError, which must not kill MegatronTrainRayActor import.
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions


try:
    from sglang.srt.managers.io_struct import DeltaEncoding, DeltaParam, DeltaSpec
except (ImportError, AttributeError):
    # Older sglang images don't have delta-sync io_struct. Only --update-weight-mode=delta
    # needs these; the default full-sync path runs without them.
    # AttributeError: io_struct transitively re-imports the quantization->aiter chain; when
    # the concurrent-actor aiter JIT race (see the fp8_utils guard above) has already left
    # aiter without `dtypes` in this process, that chain raises AttributeError here too.
    DeltaEncoding = None
    DeltaParam = None
    DeltaSpec = None

from sglang.srt.utils import MultiprocessingSerializer


try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

__all__ = [
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
    "DeltaEncoding",
    "DeltaParam",
    "DeltaSpec",
]
