#!/usr/bin/env python3
"""Reproduce the Q6_K share from an Nsight Systems SQLite export.

Export first with:
    nsys export --type sqlite --output trace.sqlite trace.nsys-rep

The database is opened read-only. No profiler data is changed.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def read_kernel_totals(database: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (row count, duration ns) for all graph nodes and Q6_K MMVQ."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        all_kernels = connection.execute(
            """
            SELECT COUNT(*), SUM(end - start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE graphNodeId IS NOT NULL
            """
        ).fetchone()
        q6k_kernels = connection.execute(
            """
            SELECT COUNT(*), SUM(kernel.end - kernel.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds AS strings ON strings.id = kernel.demangledName
            WHERE kernel.graphNodeId IS NOT NULL
              AND strings.value LIKE 'void mul_mat_vec_q<(ggml_type)14,%'
            """
        ).fetchone()
    finally:
        connection.close()

    if all_kernels[1] is None or q6k_kernels[1] is None:
        raise RuntimeError("the export contains no matching graph-node kernel rows")
    return all_kernels, q6k_kernels


def read_largest_groups(database: Path) -> list[tuple[object, ...]]:
    """Return the eight graph-node kernel groups with the greatest duration."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        return connection.execute(
            """
            SELECT strings.value,
                   kernel.gridX,
                   kernel.blockX,
                   kernel.blockY,
                   COUNT(*) AS launches,
                   SUM(kernel.end - kernel.start) AS duration_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds AS strings ON strings.id = kernel.demangledName
            WHERE kernel.graphNodeId IS NOT NULL
            GROUP BY strings.value, kernel.gridX, kernel.blockX, kernel.blockY
            ORDER BY duration_ns DESC
            LIMIT 8
            """
        ).fetchall()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Q6_K MMVQ time in an Nsight Systems graph-node export"
    )
    parser.add_argument("database", type=Path, help="Nsight Systems SQLite export")
    args = parser.parse_args()

    database = args.database.resolve()
    if not database.is_file():
        raise SystemExit(f"database does not exist: {database}")

    (all_rows, all_ns), (q6k_rows, q6k_ns) = read_kernel_totals(database)
    share = 100.0 * q6k_ns / all_ns

    print(f"All graph-node kernels: {all_rows:4d} rows, {all_ns / 1e6:7.3f} ms")
    print(f"Q6_K MMVQ kernels:      {q6k_rows:4d} rows, {q6k_ns / 1e6:7.3f} ms")
    print(f"Q6_K share:             {share:7.2f}%")
    print("\nLargest graph-node groups:")

    for name, grid_x, block_x, block_y, launches, duration_ns in read_largest_groups(database):
        short_name = str(name).split("(", 1)[0]
        if "mul_mat_vec_q" in str(name) and "(ggml_type)14" in str(name):
            short_name = "Q6_K " + ("fused" if "(bool)1" in str(name) else "nonfused")
        print(
            f"{int(duration_ns) / 1e6:8.3f} ms  {int(launches):4d} launches  "
            f"gridX={int(grid_x):<6d} block=({int(block_x)},{int(block_y)},1)  {short_name}"
        )


if __name__ == "__main__":
    main()
