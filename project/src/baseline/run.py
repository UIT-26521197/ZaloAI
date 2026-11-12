from ..config import load_config
from ..inference.run import run
from ..utils.logging import configure_logging


def main(argv=None):
    configure_logging()
    return run(load_config("baseline", argv))


if __name__ == "__main__":
    main()
