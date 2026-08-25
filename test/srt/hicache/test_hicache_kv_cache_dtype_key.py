"""Unit tests for kv_cache_dtype inclusion in HiCache storage keys.

Verifies that HiCacheStorageConfig.kv_cache_dtype is threaded into the
key derivation of each backend so that runs with different --kv-cache-dtype
do not silently reuse each other's cache entries.

Issue: https://github.com/sgl-project/sglang/issues/33268
"""

import tempfile
import pytest

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
)


def _make_config(kv_cache_dtype=None):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=8,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="DeepSeek-V4-Flash",
        kv_cache_dtype=kv_cache_dtype,
    )


class TestHiCacheDtypeKey:
    """Verify that different kv_cache_dtype values produce different storage keys."""

    def test_different_dtype_produces_different_suffix(self):
        """Two configs differing only in kv_cache_dtype must have different suffixes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_bf16 = HiCacheFile(_make_config("bf16"), file_path=tmpdir)
            cache_fp8 = HiCacheFile(_make_config("fp8_e4m3"), file_path=tmpdir)

            assert cache_bf16.config_suffix != cache_fp8.config_suffix, (
                f"Suffix collision: bf16={cache_bf16.config_suffix}, "
                f"fp8={cache_fp8.config_suffix}"
            )
            assert "dtype_bf16" in cache_bf16.config_suffix
            assert "dtype_fp8_e4m3" in cache_fp8.config_suffix

    def test_dtype_in_suffixed_key(self):
        """The suffixed key must contain the dtype segment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(_make_config("fp8_e4m3"), file_path=tmpdir)
            key = cache._get_suffixed_key("page_001")
            assert "dtype_fp8_e4m3" in key, f"dtype missing from key: {key}"

    def test_none_dtype_backward_compatible(self):
        """When kv_cache_dtype is None (not set), suffix must not contain dtype.

        This ensures backward compatibility: existing cache files created
        before the fix remain accessible with the old key format.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(_make_config(None), file_path=tmpdir)
            assert "dtype" not in cache.config_suffix, (
                f"dtype leaked into suffix when kv_cache_dtype is None: "
                f"{cache.config_suffix}"
            )
            assert cache.config_suffix == "_DeepSeek-V4-Flash", (
                f"Unexpected suffix for None dtype: {cache.config_suffix}"
            )

    def test_same_dtype_produces_same_suffix(self):
        """Two configs with the same kv_cache_dtype must produce the same suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = HiCacheFile(_make_config("bf16"), file_path=tmpdir)
            cache2 = HiCacheFile(_make_config("bf16"), file_path=tmpdir)
            assert cache1.config_suffix == cache2.config_suffix

    def test_non_mla_model_dtype_in_suffix(self):
        """Non-MLA models should also include dtype in the suffix."""
        cfg_bf16 = HiCacheStorageConfig(
            tp_rank=0, tp_size=4, pp_rank=0, pp_size=1,
            attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
            enable_storage_metrics=False, is_page_first_layout=False,
            model_name="Qwen2.5-7B", kv_cache_dtype="bf16",
        )
        cfg_fp8 = HiCacheStorageConfig(
            tp_rank=0, tp_size=4, pp_rank=0, pp_size=1,
            attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
            enable_storage_metrics=False, is_page_first_layout=False,
            model_name="Qwen2.5-7B", kv_cache_dtype="fp8_e5m2",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_bf16 = HiCacheFile(cfg_bf16, file_path=tmpdir)
            cache_fp8 = HiCacheFile(cfg_fp8, file_path=tmpdir)
            assert cache_bf16.config_suffix != cache_fp8.config_suffix
            assert "dtype_bf16" in cache_bf16.config_suffix
            assert "dtype_fp8_e5m2" in cache_fp8.config_suffix


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
