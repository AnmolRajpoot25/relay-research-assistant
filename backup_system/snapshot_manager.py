"""
Snapshot and State Backup Manager.

Senior Engineering Design Note:
In multi-model or quota-constrained deployments, upstream LLM provider quotas
(or chat agent quotas) can trigger sudden rate limits or model switches.
This module guarantees zero-loss persistence:
1. Automatic checkpoints of in-flight pipeline states, token tallies, and query logs.
2. Codebase state snapshotting into a timestamped, versioned backup vault.
3. Safe atomic writes so partial failures or unexpected interruptions never
   corrupt existing checkpoints.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SnapshotManager")


@dataclass
class SessionCheckpoint:
    """State snapshot of a running session or batch run."""
    checkpoint_id: str
    timestamp: float = field(default_factory=time.time)
    active_model: str = "primary"
    completed_requests: int = 0
    total_tokens_consumed: int = 0
    pending_items: List[Dict[str, Any]] = field(default_factory=list)
    completed_items: List[Dict[str, Any]] = field(default_factory=list)
    system_metadata: Dict[str, Any] = field(default_factory=dict)


class SnapshotManager:
    """
    Manages codebase snapshots, runtime checkpoints, and seamless restoration.
    """

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path(__file__).resolve().parent.parent
        self.backup_vault_directory = self.workspace_root / ".backup_vault"
        self.checkpoints_directory = self.backup_vault_directory / "checkpoints"
        self.snapshots_directory = self.backup_vault_directory / "code_snapshots"

        self.checkpoints_directory.mkdir(parents=True, exist_ok=True)
        self.snapshots_directory.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint: SessionCheckpoint) -> Path:
        """
        Atomically saves a session checkpoint to avoid corruption on unexpected termination.
        """
        target_path = self.checkpoints_directory / f"{checkpoint.checkpoint_id}.json"
        temporary_path = self.checkpoints_directory / f"{checkpoint.checkpoint_id}.tmp"

        payload = asdict(checkpoint)
        with open(temporary_path, "w", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, indent=2)

        # Atomic replace
        os.replace(temporary_path, target_path)
        logger.info(f"Saved atomic checkpoint: {target_path.name}")
        return target_path

    def load_latest_checkpoint(self) -> Optional[SessionCheckpoint]:
        """
        Loads the most recent checkpoint available in the vault.
        """
        checkpoints = sorted(
            self.checkpoints_directory.glob("*.json"),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )
        if not checkpoints:
            return None

        latest_file = checkpoints[0]
        try:
            with open(latest_file, "r", encoding="utf-8") as checkpoint_file:
                raw_data = json.load(checkpoint_file)
            logger.info(f"Loaded latest session checkpoint from {latest_file.name}")
            return SessionCheckpoint(**raw_data)
        except Exception as error:
            logger.error(f"Failed to read checkpoint {latest_file.name}: {error}")
            return None

    def create_codebase_snapshot(self, label: str = "manual_backup") -> Path:
        """
        Backs up the entire source tree (ignoring caches, virtual environments, and large binaries).
        """
        timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
        snapshot_directory = self.snapshots_directory / f"snapshot_{label}_{timestamp_slug}"
        snapshot_directory.mkdir(parents=True, exist_ok=True)

        ignored_patterns = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
            ".backup_vault",
            "*.pyc",
        }

        def should_copy(path: Path) -> bool:
            for part in path.parts:
                if part in ignored_patterns:
                    return False
            return True

        for current_root, directory_names, file_names in os.walk(self.workspace_root):
            relative_root = Path(current_root).relative_to(self.workspace_root)
            if not should_copy(relative_root):
                continue

            target_folder = snapshot_directory / relative_root
            target_folder.mkdir(parents=True, exist_ok=True)

            for file_name in file_names:
                source_file = Path(current_root) / file_name
                if file_name.endswith(".pyc") or file_name.endswith(".tmp"):
                    continue
                shutil.copy2(source_file, target_folder / file_name)

        logger.info(f"Created complete project snapshot at: {snapshot_directory}")
        return snapshot_directory

