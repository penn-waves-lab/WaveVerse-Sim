"""JSON configuration shared by the spatial and temporal examples."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_config(parser, workflow, argv=None):
    """Load file defaults, then apply explicit command-line overrides."""
    argv = sys.argv[1:] if argv is None else list(argv)
    default = ROOT / "configs" / f"{workflow}.json"
    parser.add_argument(
        "--config", type=Path, default=default, help="JSON configuration file"
    )
    parser.add_argument(
        "--specular",
        dest="specular_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include pure specular returns (fresh solve per position/chirp)",
    )
    parser.add_argument(
        "--scattering",
        dest="scattering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include scattering paths and their reflection prefixes",
    )
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path, default=default)
    path = probe.parse_known_args(argv)[0].config.resolve()
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.pop("workflow", None) != workflow:
            raise ValueError(f"Expected a {workflow!r} configuration object")
        actions = {
            action.dest: action
            for action in parser._actions
            if action.dest not in ("help", "config")
        }
        extra = {"materials", "material_assignments"}
        unknown = data.keys() - actions.keys() - extra
        if unknown:
            raise ValueError(
                f"Unknown configuration keys: {', '.join(sorted(unknown))}"
            )
        defaults = {}
        for key, value in data.items():
            if key in extra:
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be an object")
                defaults[key] = value
                continue
            action = actions[key]

            def scalar(item):
                if action.type in (int, float):
                    if isinstance(item, bool) or not isinstance(item, (int, float)):
                        raise ValueError(f"{key} must contain numbers")
                    if not math.isfinite(item) or (
                        action.type is int and not isinstance(item, int)
                    ):
                        raise ValueError(f"Invalid numeric value for {key}")
                elif action.type is Path and not isinstance(item, str):
                    raise ValueError(f"{key} must be a path string")
                converted = action.type(item) if action.type else item
                if isinstance(converted, Path) and not converted.is_absolute():
                    converted = (path.parent / converted).resolve()
                if action.choices and converted not in action.choices:
                    raise ValueError(f"Invalid {key}: {converted!r}")
                return converted

            if isinstance(
                action,
                (
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                    argparse.BooleanOptionalAction,
                ),
            ):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be true or false")
                defaults[key] = value
            elif isinstance(action, argparse._AppendAction):
                if not isinstance(value, list) or not value:
                    raise ValueError(f"{key} must be a nonempty list")
                if any(
                    not isinstance(row, list) or len(row) != action.nargs
                    for row in value
                ):
                    raise ValueError(
                        f"{key} must contain lists of {action.nargs} values"
                    )
                defaults[key] = [[scalar(item) for item in row] for row in value]
                if any(
                    token.split("=", 1)[0] in action.option_strings for token in argv
                ):
                    defaults[key] = None
            elif isinstance(action.nargs, int):
                if not isinstance(value, list) or len(value) != action.nargs:
                    raise ValueError(f"{key} must contain {action.nargs} values")
                defaults[key] = [scalar(item) for item in value]
            else:
                if value is None or isinstance(value, (dict, list, bool)):
                    raise ValueError(f"Invalid value for {key}")
                defaults[key] = scalar(value)
        parser.set_defaults(
            materials={},
            material_assignments={},
            **{k: v for k, v in defaults.items() if k not in extra},
        )
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in extra})
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"{path}: {error}")
    args = parser.parse_args(argv)
    args.config = path
    if not args.specular_only and not args.scattering:
        parser.error("Enable at least one of specular or scattering paths")
    return args


def resolved_config(args):
    """JSON-serializable settings recorded alongside simulation results."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
