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


@dataclass(frozen=True)
class CommandResult:
    id: str
    returncode: int
    gpu_group: tuple[int, ...]
    stdout_path: Path
    stderr_path: Path


class LocalGpuExecutor:
    def __init__(
        self,
        cuda_devices: list[int],
        *,
        gpus_per_job: int = 1,
        jobs_per_gpu_group: int = 1,
    ) -> None:
        if not cuda_devices:
            raise ValueError("at least one CUDA device is required")
        if gpus_per_job < 1 or len(cuda_devices) % gpus_per_job:
            raise ValueError("CUDA devices must divide evenly into GPU groups")
        if jobs_per_gpu_group < 1:
            raise ValueError("jobs_per_gpu_group must be positive")
        self.groups = [
            tuple(cuda_devices[start : start + gpus_per_job])
            for start in range(0, len(cuda_devices), gpus_per_job)
        ]
        self.capacity = jobs_per_gpu_group
        self._condition = threading.Condition()
        self._occupancy = {group: 0 for group in self.groups}

    def _acquire(self) -> tuple[int, ...]:
        with self._condition:
            self._condition.wait_for(
                lambda: any(
                    occupancy < self.capacity
                    for occupancy in self._occupancy.values()
                )
            )
            group = min(self.groups, key=self._occupancy.__getitem__)
            self._occupancy[group] += 1
            return group

    def _release(self, group: tuple[int, ...]) -> None:
        with self._condition:
            self._occupancy[group] -= 1
            self._condition.notify()

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
            group = self._acquire()
            try:
                for command in chain:
                    result = self._execute(command, group)
                    results[index].append(result)
                    if result.returncode:
                        break
            finally:
                self._release(group)

        workers = len(self.groups) * self.capacity
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: list[Future[None]] = [
                pool.submit(run_chain, index, chain) for index, chain in enumerate(chains) if chain
            ]
            for future in as_completed(futures):
                future.result()
        return results

