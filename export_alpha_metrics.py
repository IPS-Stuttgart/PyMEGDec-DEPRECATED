"""Backward-compatible wrapper for ``pymegdec alpha metrics``."""

from script_bootstrap import add_src_to_path

add_src_to_path(__file__)

from pymegdec.alpha_cli import alpha_metrics  # noqa: E402


def main() -> int:
    return alpha_metrics()


if __name__ == "__main__":
    raise SystemExit(main())
