import warnings

from core.lower.lower_linear import lower_tl
from .stateful_operator import load_generated_callable

class LinearAttentionEngine:
    """Deprecated modifier adapter for the IR-first StatefulOperator seam."""

    def __init__(
        self,
        qkv_meta,
        q_mod=None,
        k_mod=None,
        v_mod=None,
        decay_mod=None,
        custom_io=None,
        tune=False,
        tune_filename="tune_result",
        tune_bwd=False,
    ):
        warnings.warn(
            "LinearAttentionEngine is deprecated; construct AlgorithmIR and StatefulOperator instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._compile_tl(
            qkv_meta,
            q_mod,
            k_mod,
            v_mod,
            decay_mod,
            custom_io,
            tune=tune,
            tune_filename=tune_filename,
            tune_bwd=tune_bwd,
        )

    def __call__(self, *args, **kargs):
        return self.attention(*args, **kargs)


    def _compile_tl(
        self,
        qkv_meta,
        q_mod,
        k_mod,
        v_mod,
        decay_mod,
        custom_io,
        tuned_config=None,
        tune=False,
        tune_filename="",
        tune_bwd=False,
    ):
        tl_code = lower_tl(
            qkv_meta,
            q_mod,
            k_mod,
            v_mod,
            decay_mod,
            custom_io,
            tuned_config,
            tune=tune,
            tune_filename=tune_filename,
            tune_bwd=tune_bwd,
        )
        self.tl_code = tl_code
        self.attention = load_generated_callable(tl_code, "linear_attention")
