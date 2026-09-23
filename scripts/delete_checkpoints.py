"""Delete saved model checkpoints under logs/ to reclaim disk space.

Checkpoints live at ``logs/<model>/<run_name>/<seed>/checkpoints/`` and are
written by the training loop for later evaluation (see ``scripts/evaluate_logs.py``).
They are not needed once a run's metrics have been collected, but they are
large (hundreds of MB to low GBs per run), so this script defaults to a dry
run and only deletes when explicitly told to.

Usage:
    # Preview what would be deleted, with total size.
    python scripts/delete_checkpoints.py --logs-dir logs

    # Actually delete every checkpoints/ directory found under logs/.
    python scripts/delete_checkpoints.py --logs-dir logs --execute

    # Only delete checkpoints for a specific model.
    python scripts/delete_checkpoints.py --logs-dir logs/gem --execute
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_checkpoint_directories(logs_root: Path) -> list[Path]:
    """Find every ``checkpoints`` directory under a logs tree.

    Args:
        logs_root: Root directory to search, typically the repository's
            ``logs/`` folder or a subdirectory of it.

    Returns:
        Sorted list of paths to directories named ``checkpoints``.

    Usage:
        >>> find_checkpoint_directories(Path("logs/gem"))
        [PosixPath('logs/gem/.../0/checkpoints'), ...]
    """
    return sorted(path for path in logs_root.rglob("checkpoints") if path.is_dir())


def directory_size_in_bytes(directory: Path) -> int:
    """Compute the total size of all files under a directory, recursively.

    Args:
        directory: Directory whose contents should be measured.

    Returns:
        Total size in bytes of every regular file under ``directory``.

    Usage:
        >>> directory_size_in_bytes(Path("logs/gem/run/0/checkpoints"))
        297795584
    """
    return sum(
        file_path.stat().st_size
        for file_path in directory.rglob("*")
        if file_path.is_file()
    )


def format_bytes_as_human_readable(size_in_bytes: int) -> str:
    """Format a byte count using the largest sensible binary unit.

    Args:
        size_in_bytes: Number of bytes to format.

    Returns:
        A human-readable string such as ``"1.5 GiB"``.

    Usage:
        >>> format_bytes_as_human_readable(1610612736)
        '1.5 GiB'
    """
    size = float(size_in_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def delete_checkpoint_directories(
    checkpoint_directories: list[Path], execute: bool
) -> int:
    """Report on and optionally delete a list of checkpoint directories.

    Args:
        checkpoint_directories: Directories to remove, as found by
            `find_checkpoint_directories`.
        execute: If True, actually delete each directory. If False, only
            print what would be deleted.

    Returns:
        Total size in bytes of the directories (deleted or previewed).

    Usage:
        >>> dirs = find_checkpoint_directories(Path("logs"))
        >>> delete_checkpoint_directories(dirs, execute=False)
    """
    total_size_in_bytes = 0
    for checkpoint_directory in checkpoint_directories:
        directory_size = directory_size_in_bytes(checkpoint_directory)
        total_size_in_bytes += directory_size
        action = "Deleting" if execute else "Would delete"
        print(
            f"{action} {checkpoint_directory} ({format_bytes_as_human_readable(directory_size)})"
        )
        if execute:
            shutil.rmtree(checkpoint_directory)
    return total_size_in_bytes


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the checkpoint deletion script.

    Returns:
        Parsed arguments with ``logs_dir`` and ``execute`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root directory to search for checkpoints/ subdirectories (default: logs).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the checkpoint directories. Without this flag, only a preview is printed.",
    )
    return parser.parse_args()


def main() -> None:
    """Find and (optionally) delete all checkpoints/ directories under logs/."""
    arguments = parse_command_line_arguments()
    if not arguments.logs_dir.is_dir():
        raise SystemExit(f"logs directory not found: {arguments.logs_dir}")

    checkpoint_directories = find_checkpoint_directories(arguments.logs_dir)
    if not checkpoint_directories:
        print(f"No checkpoints/ directories found under {arguments.logs_dir}")
        return

    total_size_in_bytes = delete_checkpoint_directories(
        checkpoint_directories, execute=arguments.execute
    )
    summary_verb = "Deleted" if arguments.execute else "Would delete"
    print(
        f"\n{summary_verb} {len(checkpoint_directories)} checkpoints/ directories "
        f"totaling {format_bytes_as_human_readable(total_size_in_bytes)}."
    )
    if not arguments.execute:
        print("Re-run with --execute to actually delete them.")


if __name__ == "__main__":
    main()
