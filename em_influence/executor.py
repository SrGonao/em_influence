from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Command:
    id: str
    argv: tuple[str, ...]
    log_dir: Path
    cwd: Path | None = None
    env: dict[str, str] | None = None
    # How many GPUs this command needs visible via CUDA_VISIBLE_DEVICES. Most
    # commands need exactly one; a command that itself fans out across
    # multiple GPUs (e.g. bergson auto-detects and uses every visible device)
    # sets this higher so the executor reserves that many for it instead of
    # the usual single device.
    gpus: int = 1


@dataclass(frozen=True)
class CommandResult:
    id: str
    returncode: int
    gpu_group: tuple[int, ...]
    stdout_path: Path
    stderr_path: Path


class LocalGpuExecutor:
    """Allocates however many GPUs each chain asks for out of one shared
    pool, rather than pre-partitioning the pool into fixed-size groups. Most
    chains ask for one device, so this behaves exactly like the old
    fixed-group executor for them; a chain that asks for more (see
    Command.gpus) gets that many currently-free devices reserved for its
    whole duration, and nothing else runs on them until it releases."""

    def __init__(
        self,
        cuda_devices: list[int],
        *,
        jobs_per_gpu_group: int = 1,
    ) -> None:
        if not cuda_devices:
            raise ValueError("at least one CUDA device is required")
        if jobs_per_gpu_group < 1:
            raise ValueError("jobs_per_gpu_group must be positive")
        self.cuda_devices = list(cuda_devices)
        self.capacity = jobs_per_gpu_group
        self._condition = threading.Condition()
        self._occupancy = {device: 0 for device in self.cuda_devices}

    def _free_devices(self) -> list[int]:
        return [device for device in self.cuda_devices if self._occupancy[device] < self.capacity]

    def _acquire(self, count: int) -> tuple[int, ...]:
        if count > len(self.cuda_devices):
            raise ValueError(f"chain needs {count} GPUs but only {len(self.cuda_devices)} are configured")
        with self._condition:
            self._condition.wait_for(lambda: len(self._free_devices()) >= count)
            group = tuple(sorted(sorted(self._free_devices(), key=self._occupancy.__getitem__)[:count]))
            for device in group:
                self._occupancy[device] += 1
            return group

    def _release(self, group: tuple[int, ...]) -> None:
        with self._condition:
            for device in group:
                self._occupancy[device] -= 1
            self._condition.notify_all()

    def _execute(self, command: Command, group: tuple[int, ...]) -> CommandResult:
        safe_id = command.id.replace("/", "_")
        stdout_path = command.log_dir / "stdout" / f"{safe_id}.log"
        stderr_path = command.log_dir / "stderr" / f"{safe_id}.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(command.env or {})
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, group))
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(
                command.argv,
                cwd=command.cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        return CommandResult(
            id=command.id,
            returncode=completed.returncode,
            gpu_group=group,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def run(self, commands: Iterable[Command]) -> list[CommandResult]:
        """Run every command independently and concurrently across GPU groups."""
        return [chain[0] for chain in self.run_chains([[command] for command in commands])]

    def run_chains(self, chains: Iterable[list[Command]]) -> list[list[CommandResult]]:
        """Run each chain's commands in order on one GPU group (stopping the
        chain at its first failure); different chains run concurrently across
        groups. This is what makes independent jobs in a manifest run in
        parallel while a single job's own multi-command sequence (e.g.
        prepare -> train) still executes in order on one GPU."""
        chains = [list(chain) for chain in chains]
        results: list[list[CommandResult]] = [[] for _ in chains]

        def run_chain(index: int, chain: list[Command]) -> None:
            # One acquisition covers every command in the chain, sized to
            # the widest requirement any of them has (a chain's later
            # command needing more GPUs than its prep step still gets them
            # for the chain's whole duration - simpler than re-acquiring
            # mid-chain, and the extra reservation on cheap prep steps is
            # negligible next to the GPU-heavy command it's held for).
            group = self._acquire(max((command.gpus for command in chain), default=1))
            try:
                for command in chain:
                    result = self._execute(command, group)
                    results[index].append(result)
                    if result.returncode:
                        break
            finally:
                self._release(group)

        workers = len(self.cuda_devices) * self.capacity
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: list[Future[None]] = [
                pool.submit(run_chain, index, chain) for index, chain in enumerate(chains) if chain
            ]
            for future in as_completed(futures):
                future.result()
        return results

