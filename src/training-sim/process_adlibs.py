import json
import re
from dataclasses import dataclass
from pathlib import Path


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

    adlibs = []

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

            # Look for ad-libs in parentheses
            for pattern in adlib_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    for match in matches:
                        # Remove parentheses
                        adlib_text = match[1:-1]
                        adlibs.append(AdLib(timestamp, adlib_text))

    return adlibs


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
