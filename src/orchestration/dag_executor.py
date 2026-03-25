from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
@dataclass
class Task:
    name: str
    fn: Callable
    deps: List[str] = field(default_factory=list)
    result: Any = None
    elapsed: float = 0.0
    status: str = "pending"
class DAGExecutor:
    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
    def register(self, name: str, fn: Callable, deps: Optional[List[str]] = None) -> None:
        self._tasks[name] = Task(name=name, fn=fn, deps=deps or [])
    def run(self, target: Optional[str] = None) -> Dict[str, Any]:
        order = self._topological_sort(target)
        results: Dict[str, Any] = {}
        for task_name in order:
            task = self._tasks[task_name]
            dep_results = {d: results[d] for d in task.deps}
            t0 = time.time()
            try:
                task.result  = task.fn(**dep_results)
                task.status  = "done"
            except Exception as exc:
                task.status  = "failed"
                task.result  = exc
                raise RuntimeError(f"Task '{task_name}' failed: {exc}") from exc
            finally:
                task.elapsed = time.time() - t0
            results[task_name] = task.result
        return results
    def _topological_sort(self, target: Optional[str] = None) -> List[str]:
        if target is not None:
            needed = self._reachable_deps(target)
        else:
            needed = set(self._tasks.keys())
        visited: Set[str] = set()
        order: List[str] = []
        def dfs(name: str) -> None:
            if name in visited:
                return
            for dep in self._tasks[name].deps:
                dfs(dep)
            visited.add(name)
            order.append(name)
        for name in needed:
            dfs(name)
        return [n for n in order if n in needed]
    def _reachable_deps(self, name: str) -> Set[str]:
        reached: Set[str] = {name}
        stack = [name]
        while stack:
            n = stack.pop()
            for dep in self._tasks[n].deps:
                if dep not in reached:
                    reached.add(dep)
                    stack.append(dep)
        return reached
    def status_report(self) -> str:
        lines = [f"{'Task':<30} {'Status':<10} {'Time':>8}"]
        lines.append("-" * 52)
        for name, task in self._tasks.items():
            lines.append(f"{name:<30} {task.status:<10} {task.elapsed:>7.2f}s")
        return "\n".join(lines)