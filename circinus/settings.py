from pathlib import Path

import dynaconf


settings = dynaconf.Dynaconf(environments=False)


def load_config(config_file: Path | str) -> dynaconf.Dynaconf:
    settings.load_file(path=str((Path(config_file).resolve(strict=True))))
    return settings
