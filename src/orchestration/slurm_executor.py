from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import subprocess
import time
@dataclass
class SlurmJobConfig:
    job_name:        str
    n_nodes:         int          = 1
    n_tasks_per_node: int         = 1
    n_gpus_per_node: int          = 1
    gpu_type:        str          = "a100"
    memory_gb:       int          = 64
    walltime_hours:  int          = 48
    partition:       str          = "gpu"
    account:         Optional[str] = None
    conda_env:       str          = "rgp"
    extra_modules:   List[str]    = field(default_factory=list)
@dataclass
class SlurmJob:
    job_id:      str
    name:        str
    status:      str   
    submit_time: float = 0.0
    script_path: Optional[str] = None
class SlurmExecutor:
    def __init__(
        self,
        output_dir:     Union[str, Path] = "logs/slurm",
        max_retries:    int  = 3,
        dry_run:        bool = False,
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.max_retries = max_retries
        self.dry_run     = dry_run
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, SlurmJob] = {}
    def generate_script(
        self,
        script_path:    Union[str, Path],
        config:         SlurmJobConfig,
        command:        str,
    ) -> Path:
        lines = [
            ,
            f"#SBATCH --job-name={config.job_name}",
            f"#SBATCH --nodes={config.n_nodes}",
            f"#SBATCH --ntasks-per-node={config.n_tasks_per_node}",
            f"#SBATCH --gres=gpu:{config.n_gpus_per_node}",
            f"#SBATCH --mem={config.memory_gb}G",
            f"#SBATCH --time={config.walltime_hours:02d}:00:00",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --output={self.output_dir}/{config.job_name}_%j.out",
            f"#SBATCH --error={self.output_dir}/{config.job_name}_%j.err",
        ]
        if config.account:
            lines.append(f"#SBATCH --account={config.account}")
        lines.append("")
        for mod in config.extra_modules:
            lines.append(f"module load {mod}")
        lines.append(f"conda activate {config.conda_env}")
        lines.append("")
        lines.append(command)
        lines.append("")
        script_path = Path(script_path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("\n".join(lines))
        return script_path
    def submit(
        self,
        script_path:  Union[str, Path],
        config:       Optional[SlurmJobConfig] = None,
    ) -> str:
        if self.dry_run:
            job_id = f"DRY_RUN_{time.time():.0f}"
            self._jobs[job_id] = SlurmJob(
                job_id=job_id,
                name=config.job_name if config else "unknown",
                status="COMPLETED",
                submit_time=time.time(),
                script_path=str(script_path),
            )
            return job_id
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")
        job_id = result.stdout.strip().split()[-1]
        self._jobs[job_id] = SlurmJob(
            job_id=job_id,
            name=config.job_name if config else "unknown",
            status="PENDING",
            submit_time=time.time(),
            script_path=str(script_path),
        )
        return job_id
    def status(self, job_id: str) -> str:
        if self.dry_run:
            return "COMPLETED"
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "-P", "--noheader"],
                capture_output=True, text=True,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            return lines[0] if lines else "UNKNOWN"
        except FileNotFoundError:
            return "UNKNOWN"
    def wait(
        self,
        job_id:        str,
        poll_interval: int = 60,
        timeout_s:     int = 259_200,  
    ) -> str:
        start = time.time()
        while True:
            s = self.status(job_id)
            if s in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                return s
            if time.time() - start > timeout_s:
                return "TIMEOUT"
            time.sleep(poll_interval)