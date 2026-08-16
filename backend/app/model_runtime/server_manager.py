"""
llama-server process manager.

Turns "run llama-server with the right arguments" into one tested,
repeatable place. It does two jobs:

  1. Build the exact command. Given the active model and settings, it
     produces the precise llama-server argument list - the same command
     whether it is run here as a subprocess, printed for the kiosk
     operator, or baked into the Docker/compose deployment (Option B,
     where llama-server runs as its own service). `command_string` is
     that documented, copy-pasteable invocation.

  2. Run it for development. `ServerManager` can start llama-server as a
     child process and stop it cleanly, as a context manager, so tests
     and local runs can bring the real server up and down. On the kiosk,
     llama-server is launched separately (Option B); this manager is the
     dev convenience, not the production supervisor.

Deliberately out of scope for: crash supervision / auto-restart
and health-aware routing. Those belong to (Resilience), so we
do not build them twice. Here, start means start, stop means stop.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from app.config import paths
from app.config.settings import settings
from app.core.exceptions import ShambaRafikiError
from app.model_runtime.readiness import ReadinessResult, wait_for_ready
from app.model_runtime.registry import ModelSpec, default_model, resolved_model_path
from app.utils.logger import get_logger

logger = get_logger("ServerManager")


class ServerManagerError(ShambaRafikiError):
    """Raised when the llama-server process cannot be managed."""

    # Common locations a locally-built llama-server binary lands in, relative
    # to the repo's bundled llama.cpp/ checkout. Tried in order after PATH.


_LOCAL_BINARY_CANDIDATES = (
    "build/bin/llama-server",
    "build/bin/Release/llama-server.exe",
    "build/bin/llama-server.exe",
    "llama-server",
)


@dataclass(slots=True)
class ServerConfig:
    """Resolved inputs for one llama-server invocation."""

    model_path: Path
    host: str
    port: int
    context_size: int
    threads: int
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(
        cls,
        spec: ModelSpec | None = None,
        model_path: Path | None = None,
        extra_args: list[str] | None = None,
    ) -> "ServerConfig":
        """
        Build a config from application settings + the active model.

        Model file resolution order:
          1. an explicit ``model_path`` argument, else
          2. the resolved path of ``spec`` (or the default model), i.e.
             ``models/<filename>``.
        Host/port come from ``settings.LLM_SERVER_URL``; context size and
        threads from ``settings.MODEL_CONTEXT_SIZE`` / ``MODEL_THREADS``.
        """
        spec = spec or default_model
        resolved = model_path or resolved_model_path(spec)

        parsed = urlparse(settings.LLM_SERVER_URL)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080

        return cls(
            model_path=Path(resolved),
            host=host,
            port=port,
            context_size=settings.MODEL_CONTEXT_SIZE,
            threads=settings.MODEL_THREADS,
            extra_args=list(extra_args or []),
        )


def resolve_server_binary(binary: str | None = None) -> str:
    """
    Locate the llama-server executable.

    Order: an explicit/absolute path if it exists, then PATH lookup of
    ``settings.LLAMA_SERVER_BIN``, then the repo's bundled llama.cpp build
    directory. Raises with actionable guidance if nothing is found - a
    missing binary is a setup problem, and the message should say so.
    """
    name = binary or settings.LLAMA_SERVER_BIN

    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    llama_cpp_dir = paths.ROOT_DIR / "llama.cpp"
    for rel in _LOCAL_BINARY_CANDIDATES:
        path = llama_cpp_dir / rel
        if path.exists():
            return str(path)

    raise ServerManagerError(
        f"Could not find the llama-server binary ({name!r}). Build "
        f"llama.cpp (CMake, CPU-only) or set LLAMA_SERVER_BIN to its full "
        f"path. Looked on PATH and under {llama_cpp_dir}."
    )


def build_command(config: ServerConfig, binary: str | None = None) -> list[str]:
    """
    The exact llama-server argument list for ``config``.

    Pure and side-effect-free (does not touch the process) so it can be
    unit-tested and printed. Does not require the model file to exist -
    that check belongs to start, not to describing the command.
    """
    server_bin = resolve_server_binary(binary)
    return [
        server_bin,
        "-m",
        str(config.model_path),
        "-c",
        str(config.context_size),
        "-t",
        str(config.threads),
        "--host",
        config.host,
        "--port",
        str(config.port),
        *config.extra_args,
    ]


def command_string(config: ServerConfig, binary: str | None = None) -> str:
    """Shell-quoted form of:func:`build_command`, for docs and logs."""
    return shlex.join(build_command(config, binary))


class ServerManager:
    """
    Start and stop a llama-server child process for development.

    Use as a context manager so the process is always cleaned up:

        cfg = ServerConfig.from_settings
        with ServerManager(cfg) as mgr:
            mgr.wait_until_ready... # server is up on cfg.host:cfg.port

    On the kiosk, llama-server runs as its own service (Docker Option B);
    this class is the local convenience, not the production supervisor.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        binary: str | None = None,
    ) -> None:
        self.config = config or ServerConfig.from_settings
        self._binary = binary
        self._process: subprocess.Popen | None = None

    @property
    def server_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll is None

    def start(self) -> subprocess.Popen:
        """
        Launch llama-server. Requires the model file to exist.

        Raises ServerManagerError if already running, the model file is
        missing, or the process fails to spawn.
        """
        if self.is_running:
            raise ServerManagerError("llama-server is already running.")

        if not self.config.model_path.exists():
            raise ServerManagerError(
                f"Model file not found: {self.config.model_path}. Download "
                f"it first (scripts/download_model.py)."
            )

        command = build_command(self.config, self._binary)
        logger.info("server.start", command=shlex.join(command))

        try:
            self._process = subprocess.Popen(  # noqa: S603 - args are ours.
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (OSError, ValueError) as exc:
            raise ServerManagerError(f"Failed to start llama-server: {exc}") from exc

        return self._process

    def wait_until_ready(self, timeout_s: float | None = None) -> ReadinessResult:
        """Block until the server answers its health probe (see readiness)."""
        if not self.is_running:
            raise ServerManagerError("Cannot wait for readiness: server is not running.")
        return wait_for_ready(self.server_url, timeout_s=timeout_s)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Terminate the process, escalating to kill if it will not exit."""
        if self._process is None:
            return

        if self._process.poll is None:
            logger.info("server.stop")
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning("server.stop.kill", reason="terminate timed out")
                self._process.kill()
                self._process.wait(timeout=timeout_s)

        self._process = None

    def __enter__(self) -> "ServerManager":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
