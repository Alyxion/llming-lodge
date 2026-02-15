"""Generate README.pypi.md with absolute GitHub URLs for PyPI rendering."""

import re
from pathlib import Path

REPO_RAW = "https://raw.githubusercontent.com/Alyxion/llming-lodge/main"
REPO_URL = "https://github.com/Alyxion/llming-lodge/blob/main"

ROOT = Path(__file__).resolve().parent.parent


def main():
    readme = (ROOT / "README.md").read_text()

    # Convert relative image src in HTML tags to absolute raw URLs
    readme = re.sub(
        r'src="(?!https?://)([^"]+)"',
        lambda m: f'src="{REPO_RAW}/{m.group(1)}"',
        readme,
    )

    # Convert relative markdown images to absolute raw URLs
    readme = re.sub(
        r'!\[([^\]]*)\]\((?!https?://)([^)]+)\)',
        lambda m: f'![{m.group(1)}]({REPO_RAW}/{m.group(2)})',
        readme,
    )

    (ROOT / "README.pypi.md").write_text(readme)
    print("Generated README.pypi.md")


if __name__ == "__main__":
    main()
