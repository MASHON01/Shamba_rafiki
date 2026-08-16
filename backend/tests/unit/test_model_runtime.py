"""
Unit tests for the model runtime (, ).

Everything here runs on fakes, offline, in CI: a hand-built in-memory
GGUF header, a monkeypatched downloader, and an injectable readiness
clock. No real model, no network - real RAM/latency numbers come from
 on the target machine, never from here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from app.config.settings import settings
from app.core.exceptions import RegistryError
from app.model_runtime import (
    build_model_info,
    default_model,
    get_model,
    list_models,
    resolved_model_path,
    wait_for_ready,
)
from app.model_runtime import downloader as dl
from app.model_runtime import server_manager as sm
from app.model_runtime.gguf_validator import GGUFValidationError, validate_gguf
from app.model_runtime.registry import ModelSpec

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers: build a minimal, valid GGUF byte string in memory
# ---------------------------------------------------------------------------


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def build_gguf_bytes(
    architecture: str = "llama",
    file_type: int = 15,  # 15 == Q4_K_M
    pad_to: int | None = None,
) -> bytes:
    """A valid GGUF v3 header with general.architecture + general.file_type."""
    out = bytearray
    out += b"GGUF"  # magic
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 0)  # tensor_count
    out += struct.pack("<Q", 2)  # kv_count

    # KV 1: general.architecture (string)
    out += _gguf_string("general.architecture")
    out += struct.pack("<I", 8)  # value type: STRING
    out += _gguf_string(architecture)

    # KV 2: general.file_type (uint32)
    out += _gguf_string("general.file_type")
    out += struct.pack("<I", 4)  # value type: UINT32
    out += struct.pack("<I", file_type)

    if pad_to is not None and pad_to > len(out):
        out += b"\x00" * (pad_to - len(out))
    return bytes(out)


def write_gguf(path: Path, **kwargs) -> Path:
    path.write_bytes(build_gguf_bytes(**kwargs))
    return path


def _spec(**overrides) -> ModelSpec:
    base = dict(
        id="test-model",
        display_name="Test Model",
        repo_id="acme/Test-GGUF",
        filename="Test-Q4_K_M.gguf",
        quantization="Q4_K_M",
        file_size_mb=0,  # size check off unless a test sets it
    )
    base.update(overrides)
    return ModelSpec(**base)

    # ---------------------------------------------------------------------------
    # registry
    # ---------------------------------------------------------------------------


def test_registry_lists_and_has_default():
    models = list_models
    assert models, "registry should not be empty"
    assert default_model.id == settings.MODEL_ID


def test_get_model_unknown_raises_with_ids():
    with pytest.raises(RegistryError) as exc:
        get_model("no-such-model")
        # The error names the registered ids so a typo is easy to fix.
    assert "llama-3.2-3b-instruct-q4_k_m" in str(exc.value)


def test_resolved_model_path_is_models_dir_plus_filename(tmp_path):
    spec = _spec(filename="Some-Model.gguf")
    assert resolved_model_path(spec, tmp_path) == tmp_path / "Some-Model.gguf"


def test_default_filenames_are_uppercase_q4_k_m():
    # bartowski filenames are case-sensitive; guard the convention.
    for spec in list_models:
        assert spec.filename.endswith(".gguf")
        assert "Q4_K_M" in spec.filename

        # ---------------------------------------------------------------------------
        # gguf_validator
        # ---------------------------------------------------------------------------


def test_validate_gguf_reads_arch_and_quant(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", architecture="llama", file_type=15)
    info = validate_gguf(path)
    assert info.version == 3
    assert info.architecture == "llama"
    assert info.quantization == "Q4_K_M"


def test_validate_gguf_missing_file(tmp_path):
    with pytest.raises(GGUFValidationError):
        validate_gguf(tmp_path / "nope.gguf")


def test_validate_gguf_empty_file(tmp_path):
    p = tmp_path / "empty.gguf"
    p.write_bytes(b"")
    with pytest.raises(GGUFValidationError):
        validate_gguf(p)


def test_validate_gguf_bad_magic(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GGUFValidationError):
        validate_gguf(p)


def test_validate_gguf_quant_mismatch(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", file_type=15)  # Q4_K_M
    with pytest.raises(GGUFValidationError):
        validate_gguf(path, expected=_spec(quantization="Q6_K"))


def test_validate_gguf_size_mismatch(tmp_path):
    path = write_gguf(tmp_path / "m.gguf")  # a few hundred bytes
    with pytest.raises(GGUFValidationError):
        validate_gguf(path, expected=_spec(file_size_mb=1000))


def test_validate_gguf_size_within_tolerance(tmp_path):
    # ~1 MB file, expected 1 MB -> within +/-15%.
    path = write_gguf(tmp_path / "m.gguf", pad_to=1024 * 1024)
    info = validate_gguf(path, expected=_spec(file_size_mb=1))
    assert info.quantization == "Q4_K_M"

    # ---------------------------------------------------------------------------
    # downloader
    # ---------------------------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "f.bin"
    p.write_bytes(b"hello shamba")
    assert dl.sha256_of(p) == hashlib.sha256(b"hello shamba").hexdigest()


def test_download_model_fetches_and_verifies(tmp_path, monkeypatch):
    spec = _spec(file_size_mb=0)  # skip size window, keep quant check

    def fake_hf(repo_id, filename, revision, local_dir):
        dest = Path(local_dir) / filename
        write_gguf(dest, file_type=15)  # Q4_K_M, matches spec
        return str(dest)

    monkeypatch.setattr(dl, "_load_hf_download", lambda: fake_hf)

    path = dl.download_model(spec, models_dir=tmp_path)
    assert path.exists()
    assert path.name == spec.filename


def test_download_model_checksum_mismatch_raises(tmp_path, monkeypatch):
    spec = _spec(sha256="deadbeef")  # wrong on purpose

    def fake_hf(repo_id, filename, revision, local_dir):
        dest = Path(local_dir) / filename
        write_gguf(dest, file_type=15)
        return str(dest)

    monkeypatch.setattr(dl, "_load_hf_download", lambda: fake_hf)

    with pytest.raises(dl.ModelDownloadError):
        dl.download_model(spec, models_dir=tmp_path)


def test_download_model_cached_skips_fetch(tmp_path, monkeypatch):
    spec = _spec
    # Pre-place a valid file; hf must NOT be called.
    write_gguf(tmp_path / spec.filename, file_type=15)

    def boom():
        raise AssertionError("should not download when cached")

    monkeypatch.setattr(dl, "_load_hf_download", boom)
    path = dl.download_model(spec, models_dir=tmp_path)
    assert path.exists()

    # ---------------------------------------------------------------------------
    # server_manager
    # ---------------------------------------------------------------------------


def test_build_command_has_core_args(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")  # exists + absolute -> used as-is
    config = sm.ServerConfig(
        model_path=tmp_path / "model.gguf",
        host="127.0.0.1",
        port=8080,
        context_size=4096,
        threads=8,
    )
    cmd = sm.build_command(config, binary=str(binary))

    assert cmd[0] == str(binary)
    assert "-m" in cmd and str(config.model_path) in cmd
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("-t") + 1] == "8"
    assert cmd[cmd.index("--port") + 1] == "8080"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_command_string_is_shell_safe(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("x")
    config = sm.ServerConfig(
        model_path=tmp_path / "m.gguf",
        host="127.0.0.1",
        port=8080,
        context_size=2048,
        threads=4,
    )
    s = sm.command_string(config, binary=str(binary))
    assert "--port 8080" in s and "-c 2048" in s


def test_config_from_settings_parses_url():
    config = sm.ServerConfig.from_settings
    # settings.LLM_SERVER_URL default is http://localhost:8080
    assert config.port == 8080
    assert config.host in ("localhost", "127.0.0.1")
    assert config.context_size == settings.MODEL_CONTEXT_SIZE
    assert config.threads == settings.MODEL_THREADS


def test_resolve_binary_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sm.shutil, "which", lambda _n: None)
    monkeypatch.setattr(sm.paths, "ROOT_DIR", tmp_path)  # no llama.cpp here
    with pytest.raises(sm.ServerManagerError):
        sm.resolve_server_binary("definitely-not-real-binary-xyz")

        # ---------------------------------------------------------------------------
        # readiness
        # ---------------------------------------------------------------------------


def test_wait_for_ready_success_first_try():
    result = wait_for_ready("http://x", probe=lambda: True)
    assert result.ready is True
    assert result.attempts == 1
    assert bool(result) is True


def test_wait_for_ready_succeeds_after_retries():
    calls = {"n": 0}

    def probe() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    result = wait_for_ready(
        "http://x",
        probe=probe,
        interval_s=0.01,
        timeout_s=100,
        sleep=lambda _s: None,
        clock=lambda: next(ticks),
    )
    assert result.ready is True
    assert result.attempts == 3


def test_wait_for_ready_times_out():
    result = wait_for_ready(
        "http://x",
        probe=lambda: False,
        timeout_s=0,
        sleep=lambda _s: None,
    )
    assert result.ready is False
    assert result.attempts >= 1

    # ---------------------------------------------------------------------------
    # model_info
    # ---------------------------------------------------------------------------


def test_model_info_configured_by_default():
    info = build_model_info(resident=True)
    assert info["configured"] is True
    assert info["id"] == settings.MODEL_ID
    assert info["resident"] is True
    assert "path" in info and "present" in info


def test_model_info_unconfigured_id_is_tolerant(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_ID", "not-a-real-id")
    info = build_model_info
    assert info["configured"] is False
    assert info["id"] == "not-a-real-id"


def test_model_info_present_flag(tmp_path):
    spec = _spec
    write_gguf(tmp_path / spec.filename)
    info = build_model_info(spec=spec, models_dir=tmp_path)
    assert info["present"] is True

    # ---------------------------------------------------------------------------
    # /health integration (model block surfaced, backward compatible)
    # ---------------------------------------------------------------------------


def test_health_includes_model_block(api_client):
    body = api_client.get("/health").json
    data = body["data"]
    assert data["llm_available"] is True  # unchanged field
    assert data["model"]["configured"] is True  # new in
    assert data["model"]["id"] == settings.MODEL_ID
