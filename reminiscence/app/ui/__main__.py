"""Launch the native Reminiscence desktop UI."""
from ..services.engine import ReminiscenceEngine
from .window import launch

def main() -> int:
    return launch(ReminiscenceEngine())

if __name__ == "__main__":
    raise SystemExit(main())
