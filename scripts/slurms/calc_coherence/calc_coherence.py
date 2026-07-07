"""
calc_coherence.py
-----------------
Wrapper script that calculates UAVSAR coherence from geocoded complex TIF files.

Supports two modes:
  interferometric  – nearest-neighbor temporal pairs sharing the same
                     site / flight / polarization / segment.
  copol            – cross-polarization pairs (HV vs. VH) sharing the same
                     site / flight / date / segment.

Usage
-----
python calc_coherence/calc_coherence.py \
    --input_dir /path/to/geoslcs \
    --out_dir   /path/to/coherence \
    --mode      interferometric \
    --window_size 5 11 \
    --polarization HH
"""

import sys
import re
import logging
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the repo root importable so that `scripts.coherence` can be found
# regardless of the current working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.coherence import calculate_coherence  # noqa: E402


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Example: lowman_05208_21021_007_210322_L090HV_01_BU_s2_2x8.tif
_FNAME_RE = re.compile(
    r"^(?P<site>[^_]+)"        # site  (e.g. lowman)
    r"_(?P<flight>\d+)"        # flight (e.g. 05208)
    r"_\d+"                    # stack id (e.g. 21021)
    r"_\d+"                    # pass   (e.g. 007)
    r"_(?P<date>\d+)"          # date   (e.g. 210322)
    r"_L\d+(?P<pol>HH|HV|VH|VV)"  # polarization band (e.g. HV from L090HV)
    r"_\d+"                    # segment id (e.g. 01)
    r"_\w+"                    # tag    (e.g. BU)
    r"_(?P<segment>s\d+)"      # segment (e.g. s2)
    r"_"                       # looks separator
)


def parse_filename(path: Path) -> dict | None:
    """
    Parse a UAVSAR geocoded TIF filename into its component parts.

    Parameters
    ----------
    path : Path
        Full path to the TIF file.

    Returns
    -------
    dict or None
        Dictionary with keys ``site``, ``flight``, ``date``, ``pol``,
        ``segment``, and ``path``, or *None* if the filename does not match
        the expected pattern.
    """
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return {**m.groupdict(), "path": path}


# ---------------------------------------------------------------------------
# Coherence calculation modes
# ---------------------------------------------------------------------------

def run_interferometric(
    files: list[dict],
    out_dir: Path,
    window_size: tuple[int, int],
    polarization: str | None = None,
) -> None:
    """
    Calculate nearest-neighbor temporal coherence.

    Files are grouped by (site, flight, pol, segment).  Within each group
    they are sorted chronologically by *date* and coherence is calculated
    for every consecutive pair that has distinct acquisition dates.

    Parameters
    ----------
    files : list[dict]
        Parsed file records (output of :func:`parse_filename`).
    out_dir : Path
        Root output directory.
    window_size : tuple[int, int]
        Rectangular window size for the coherence calculation as
        ``(row_window_size, col_window_size)``.
    polarization : str or None
        If provided, only process groups whose polarization matches this value
        (e.g. ``"HH"``, ``"HV"``).  When *None* all polarizations are processed.
    """
    logger = logging.getLogger(__name__)

    groups: dict = defaultdict(list)
    for f in files:
        # Apply polarization filter when requested
        if polarization is not None and f["pol"] != polarization:
            continue
        key = (f["site"], f["flight"], f["pol"], f["segment"])
        groups[key].append(f)

    for (site, flight, pol, segment), members in groups.items():
        # Sort chronologically (date string YYMMDD sorts correctly as-is)
        members.sort(key=lambda x: x["date"])

        if len(members) < 2:
            logger.warning(
                "Group %s/%s/%s/%s has only one file – skipping.",
                site, flight, pol, segment,
            )
            continue

        group_out_dir = Path(out_dir) / site / flight
        group_out_dir.mkdir(parents=True, exist_ok=True)

        for f1, f2 in itertools.combinations(members, 2):
            date1 = f1["date"]
            date2 = f2["date"]
                        
            # Prevent self-coherence (same date)
            if date1 == date2:
                logger.warning(
                    "Skipping self-coherence pair for %s/%s/%s/%s: both dates are %s.",
                    site, flight, pol, segment, date1,
                )
                continue

            out_name = (
                f"{site}_{flight}_{date1}_{date2}_{pol}_{segment}_w{window_size[0]}x{window_size[1]}_int_coh.tif"
            )
            out_path = group_out_dir / out_name

            if out_path.exists():
                logger.info("Skipping (already exists): %s", out_path.name)
                continue

            logger.info(
                "Calculating interferometric coherence: %s + %s -> %s",
                f1["path"].name,
                f2["path"].name,
                out_name,
            )
            try:
                calculate_coherence(
                    f1["path"], f2["path"], out_path, window_size=window_size
                )
            except Exception as exc:
                logger.error(
                    "Failed for %s: %s", out_name, exc, exc_info=True
                )


def run_crosspol(
    files: list[dict],
    out_dir: Path,
    window_size: tuple[int, int],
) -> None:
    """
    Calculate cross-polarization (HV/VH) coherence.

    Files are grouped by (site, flight, date, segment).  A coherence image
    is produced only when **both** HV and VH polarizations are present.

    Parameters
    ----------
    files : list[dict]
        Parsed file records (output of :func:`parse_filename`).
    out_dir : Path
        Root output directory.
    window_size : tuple[int, int]
        Rectangular window size for the coherence calculation as
        ``(row_window_size, col_window_size)``.
    """
    logger = logging.getLogger(__name__)

    groups: dict = defaultdict(dict)
    for f in files:
        key = (f["site"], f["flight"], f["date"], f["segment"])
        groups[key][f["pol"]] = f

    for (site, flight, date, segment), pol_map in groups.items():
        if "HV" not in pol_map or "VH" not in pol_map:
            logger.warning(
                "Group %s/%s/%s/%s is missing HV or VH – skipping.",
                site, flight, date, segment,
            )
            continue

        group_out_dir = Path(out_dir) / site / flight
        group_out_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{site}_{flight}_{date}_HV-VH_{segment}_w{window_size[0]}x{window_size[1]}_crosspol_coh.tif"
        out_path = group_out_dir / out_name

        if out_path.exists():
            logger.info("Skipping (already exists): %s", out_path.name)
            continue

        f_hv = pol_map["HV"]
        f_vh = pol_map["VH"]
        logger.info(
            "Calculating copol coherence: %s + %s -> %s",
            f_hv["path"].name,
            f_vh["path"].name,
            out_name,
        )
        try:
            calculate_coherence(
                f_hv["path"], f_vh["path"], out_path, window_size=window_size
            )
        except Exception as exc:
            logger.error(
                "Failed for %s: %s", out_name, exc, exc_info=True
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the appropriate coherence mode."""
    parser = argparse.ArgumentParser(
        description="Calculate UAVSAR coherence from geocoded TIF files.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Base directory containing the geocoded .tif files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Base directory to save the output .tif files.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interferometric", "crosspol"],
        required=True,
        help="Coherence calculation mode: 'interferometric' or 'crosspol'.",
    )
    parser.add_argument(
        "--window_size",
        nargs=2,
        type=int,
        default=(5, 5),
        help=(
            "Rectangular window size for coherence as two integers: "
            "row_window_size col_window_size (default: 5 5)."
        ),
    )
    parser.add_argument(
        "--polarization",
        type=str,
        choices=["HH", "HV", "VH", "VV"],
        default=None,
        help=(
            "Optional polarization filter for interferometric mode "
            "(e.g. HH, HV, VH, VV).  When omitted, all polarizations are processed."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Discover all TIF files recursively
    all_tifs = list(input_dir.rglob("*.tif"))
    logger.info("Found %d .tif files in %s", len(all_tifs), input_dir)

    # Parse filenames, skipping those that don't match
    parsed = []
    for tif in all_tifs:
        record = parse_filename(tif)
        if record is None:
            logger.warning("Could not parse filename, skipping: %s", tif.name)
        else:
            parsed.append(record)

    logger.info(
        "Successfully parsed %d / %d filenames.", len(parsed), len(all_tifs)
    )

    window_size = tuple(args.window_size)

    if args.mode == "interferometric":
        run_interferometric(parsed, out_dir, window_size, polarization=args.polarization)
    else:
        run_crosspol(parsed, out_dir, window_size)

    logger.info("Done.")


if __name__ == "__main__":
    main()
