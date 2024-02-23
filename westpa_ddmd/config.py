import json
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import yaml
from pydantic_settings import BaseSettings as _BaseSettings
from pydantic import validator

_T = TypeVar("_T")

PathLike = Union[str, Path]


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


def _resolve_mkdir(value: Path) -> Path:
    p = value.resolve()
    p.mkdir(exist_ok=True, parents=True)
    return p


def mkdir_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_mkdir)
    return _validator


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, filename: PathLike) -> None:
        with open(filename, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)
