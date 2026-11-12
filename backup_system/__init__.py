"""
Backup and State Persistence System.
Provides fault-tolerant persistence across model switches and unexpected quota exhaustion.
"""

from backup_system.snapshot_manager import SnapshotManager, SessionCheckpoint

__all__ = ["SnapshotManager", "SessionCheckpoint"]

