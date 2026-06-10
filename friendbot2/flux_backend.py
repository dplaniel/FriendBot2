"""
Thin async wrapper around the local ``flux/generate.py`` script.

Rather than shelling out to ``generate.py`` for every prompt (which would reload
several GB of model weights each time), this loads the chosen pipeline once and
keeps it resident, calling ``generate.py``'s own ``load_pipeline`` and
``generate`` functions. The model ("sd35" or "flux") is passed straight through
to ``load_pipeline``, the same value generate.py's ``--model`` flag selects. All
torch/CUDA work runs on a single dedicated worker thread so the asyncio event
loop is never blocked and the CUDA context stays pinned to one thread.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Optional


class FluxBackend:
    """Loads and drives the flux generate.py pipeline for the bot."""

    def __init__(self, repo_path: Path, model: str = "sd35"):
        self.repo_path = Path(repo_path)
        self.model = model
        self._module: Optional[ModuleType] = None
        self._pipe = None
        # max_workers=1 keeps every torch call on the same thread (required for a
        # stable CUDA context) and naturally serializes GPU access.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux")

    # -- loading ------------------------------------------------------------
    def _import_generate(self) -> ModuleType:
        """Import the flux generate.py module by absolute file path."""
        gen_path = self.repo_path / "generate.py"
        if not gen_path.exists():
            raise FileNotFoundError(
                f"Could not find generate.py at {gen_path}. "
                "Set FLUX_REPO_PATH to your local flux checkout."
            )
        # Put the repo on sys.path so generate.py's MODELS_DIR/OUTPUT_DIR (which are
        # resolved relative to its own __file__) and any sibling imports work.
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        spec = importlib.util.spec_from_file_location("flux_generate", gen_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec for {gen_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    async def load(self) -> None:
        """Import generate.py and build the pipeline on the worker thread."""
        loop = asyncio.get_running_loop()

        def _load():
            module = self._import_generate()
            pipe = module.load_pipeline(self.model)
            return module, pipe

        self._module, self._pipe = await loop.run_in_executor(self._executor, _load)

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    # -- generation ---------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        caption: bool = True,
    ) -> Path:
        """Generate one image and return the path to the saved PNG."""
        if not self.ready:
            raise RuntimeError("FLUX pipeline is not loaded yet.")

        loop = asyncio.get_running_loop()

        def _gen() -> Path:
            return self._module.generate(
                self._pipe, prompt, seed=seed, caption=caption
            )

        return await loop.run_in_executor(self._executor, _gen)

    # -- teardown -----------------------------------------------------------
    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
