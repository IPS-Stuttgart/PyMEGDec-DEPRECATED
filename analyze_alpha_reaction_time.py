"""Backward-compatible wrapper for ``pymegdec alpha reaction-time``."""

from script_bootstrap import add_src_to_path

add_src_to_path(__file__)

from pymegdec.alpha_cli import alpha_reaction_time  # noqa: E402


def main() -> int:
    return alpha_reaction_time()


if __name__ == "__main__":
    raise SystemExit(main())
