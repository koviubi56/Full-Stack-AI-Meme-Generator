import os
import sys
from typing import Any, Callable, Mapping, TypeVar

import dotenv
import nox

# By default (unless explicit set otherwise) running now should only
# run `test`, because `test_coverage` also uploads to CodeCov
nox.options.sessions = ["test"]

T = TypeVar("T")
PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]


def get_or_else(
    mapping: Mapping[T, Any], key: T, or_else: Callable[[], T]
) -> T:
    try:
        return mapping[key]
    except KeyError:
        return or_else()


@nox.session(python=PYTHON_VERSIONS)
def test_coverage(session: nox.Session) -> None:
    dotenv.load_dotenv()
    session.install("-U", "pip", "setuptools", "wheel")
    session.install(
        "-U",
        "-r",
        "requirements.txt",
        "pytest-randomly",
        "pytest-codecov[git]",
    )
    env = {
        "CODECOV_TOKEN": os.environ.get("CODECOV_TOKEN", sys.argv[-1]),
    }
    session.run(
        "pytest",
        "--codecov",
        "--ff",
        "-vv",
        "-r",
        "A",
        "-l",
        "--color=yes",
        "--code-highlight=yes",
        "--continue-on-collection-errors",
        env=env,
    )


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    session.install("-U", "pip", "setuptools", "wheel")
    session.install(
        "-U", "-r", "requirements.txt", "pytest-xdist", "pytest-randomly"
    )
    session.run(
        "pytest",
        "-n",
        "auto",
        "--ff",
        "-vv",
        "-r",
        "A",
        "-l",
        "--color=yes",
        "--code-highlight=yes",
        "--continue-on-collection-errors",
    )
