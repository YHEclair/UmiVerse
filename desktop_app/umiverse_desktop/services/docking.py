from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import pandas as pd


AFFINITY_PATTERN = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+")
POCKET_KEYS = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")


class VinaDockingService:
    def run_batch(
        self,
        vina_executable: str,
        receptor_pdbqt: str,
        ligand_dir: str,
        config_file: str,
        output_dir: str,
        cpu: int | None = None,
        exhaustiveness: int | None = None,
        timeout_seconds: int | None = None,
    ) -> pd.DataFrame:
        vina_path = Path(vina_executable)
        receptor_path = Path(receptor_pdbqt)
        ligand_root = Path(ligand_dir)
        config_path = Path(config_file)
        output_root = Path(output_dir)
        self._validate_inputs(vina_path, receptor_path, ligand_root, config_path)
        output_root.mkdir(parents=True, exist_ok=True)
        logs_dir = output_root / "logs"
        poses_dir = output_root / "poses"
        logs_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)
        pocket = self.parse_pocket_config(config_path)
        sanitized_config_path = output_root / "vina_pocket_only_config.txt"
        self.write_pocket_config(pocket, sanitized_config_path)

        ligand_paths = sorted(ligand_root.glob("*.pdbqt"))
        if not ligand_paths:
            raise ValueError(f"No .pdbqt ligand files found in {ligand_root}")

        rows = []
        for ligand_path in ligand_paths:
            started = time.perf_counter()
            ligand_name = ligand_path.stem
            out_path = poses_dir / f"{ligand_name}_out.pdbqt"
            log_path = logs_dir / f"{ligand_name}.log"
            command = [
                str(vina_path),
                "--receptor",
                str(receptor_path),
                "--ligand",
                str(ligand_path),
                "--config",
                str(sanitized_config_path),
                "--out",
                str(out_path),
            ]
            if cpu is not None and int(cpu) > 0:
                command.extend(["--cpu", str(int(cpu))])
            if exhaustiveness is not None and int(exhaustiveness) > 0:
                command.extend(["--exhaustiveness", str(int(exhaustiveness))])

            status = "success"
            error_message = ""
            best_affinity = None
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=int(timeout_seconds) if timeout_seconds is not None else None,
                    check=False,
                )
                self._write_vina_log(log_path, command, completed.stdout, completed.stderr)
                if completed.returncode != 0:
                    status = "failed"
                    error_message = (completed.stderr or completed.stdout).strip()
                best_affinity = self._parse_affinity(log_path, completed.stdout)
                if best_affinity is None and status == "success":
                    status = "failed"
                    error_message = "Could not parse best affinity from Vina output."
            except Exception as exc:
                status = "failed"
                error_message = str(exc)
                self._write_vina_log(log_path, command, "", str(exc))
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "ligand_name": ligand_name,
                    "ligand_path": str(ligand_path),
                    "status": status,
                    "best_affinity_kcal_mol": best_affinity,
                    "output_pdbqt": str(out_path) if out_path.exists() else "",
                    "log_path": str(log_path) if log_path.exists() else "",
                    "elapsed_seconds": round(elapsed, 3),
                    "error_message": error_message,
                }
            )
        result = pd.DataFrame(rows)
        result.to_csv(output_root / "vina_batch_results.csv", index=False)
        return result

    @staticmethod
    def parse_pocket_config(config_path: str | Path) -> dict[str, float]:
        path = Path(config_path)
        values: dict[str, float] = {}
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            key = key.lower()
            if key in POCKET_KEYS:
                try:
                    values[key] = float(value)
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value for {key}: {value}") from exc
        missing = [key for key in POCKET_KEYS if key not in values]
        if missing:
            raise ValueError(f"Config is missing required pocket parameters: {', '.join(missing)}")
        return values

    @staticmethod
    def write_pocket_config(pocket: dict[str, float], output_path: Path) -> None:
        output_path.write_text(
            "\n".join(f"{key} = {pocket[key]}" for key in POCKET_KEYS) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _validate_inputs(
        vina_path: Path,
        receptor_path: Path,
        ligand_root: Path,
        config_path: Path,
    ) -> None:
        if not vina_path.exists():
            raise FileNotFoundError(f"Vina executable not found: {vina_path}")
        if not receptor_path.is_file():
            raise FileNotFoundError(f"Receptor PDBQT not found: {receptor_path}")
        if not ligand_root.is_dir():
            raise FileNotFoundError(f"Ligand folder not found: {ligand_root}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Vina config file not found: {config_path}")

    @staticmethod
    def _parse_affinity(log_path: Path, stdout: str) -> float | None:
        text = ""
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        if not text:
            text = stdout or ""
        for line in text.splitlines():
            match = AFFINITY_PATTERN.match(line)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _write_vina_log(log_path: Path, command: list[str], stdout: str, stderr: str) -> None:
        sections = [
            "Command:",
            " ".join(command),
            "",
            "STDOUT:",
            stdout or "",
            "",
            "STDERR:",
            stderr or "",
            "",
        ]
        log_path.write_text("\n".join(sections), encoding="utf-8", errors="ignore")
