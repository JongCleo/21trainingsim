import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LrcLine:
    timestamp: float
    text: str


@dataclass
class AdLib:
    timestamp: float  # in seconds
    text: str


def parse_timestamp(ts: str) -> float:
    """Convert [MM:SS.xx] format to seconds"""
    minutes, seconds = ts[1:-1].split(":")  # Remove [] and split
    return float(minutes) * 60 + float(seconds)


def extract_adlibs(lrc_file: Path) -> list[AdLib]:
    """Extract ad-libs from LRC file"""
    # Common 21 Savage ad-libs
    adlib_patterns = [
        r"\(21\)",
        r"\(pussy\)",
        r"\(on God\)",
        r"\(straight up\)",
    ]

    # First pass: collect all lines with timestamps
    lines: list[LrcLine] = []
    with open(lrc_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (
                not line
                or line.startswith("[id:")
                or line.startswith("[ar:")
                or line.startswith("[al:")
                or line.startswith("[ti:")
                or line.startswith("[length:")
            ):
                continue

            # Extract timestamp
            timestamp_match = re.match(r"\[(\d{2}:\d{2}\.\d{2})\]", line)
            if not timestamp_match:
                continue

            timestamp = parse_timestamp(timestamp_match.group(0))
            text = line[timestamp_match.end() :].strip()
            lines.append(LrcLine(timestamp, text))

    # Second pass: find adlibs and use next line's timestamp
    adlibs = []
    for i, line in enumerate(lines):
        for pattern in adlib_patterns:
            matches = re.findall(pattern, line.text)
            if matches:
                for match in matches:
                    # Get timestamp from next line if available, otherwise use current line
                    next_timestamp = (
                        lines[i + 1].timestamp if i + 1 < len(lines) else line.timestamp
                    )
                    # Subtract a small offset to account for the delay
                    timestamp = max(0, next_timestamp - 0.5)

                    # Remove parentheses
                    adlib_text = match[1:-1]
                    adlibs.append(AdLib(timestamp, adlib_text))

    return sorted(adlibs, key=lambda x: x.timestamp)


def main():
    lrc_file = Path("data/21 Savage - Runnin.lrc")
    adlibs = extract_adlibs(lrc_file)

    # Save as JSON for web player
    output = [{"time": a.timestamp, "text": a.text} for a in adlibs]
    with open("data/adlibs.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print for verification
    for adlib in adlibs:
        print(f"{adlib.timestamp:.2f}s: {adlib.text}")


if __name__ == "__main__":
    main()
