import os
import sys

import nox

# By default (unless explicit set otherwise) running now should only
# run `test`, because `test_coverage` also uploads to CodeCov
nox.options.sessions = ["test"]

PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def test_coverage(session: nox.Session) -> None:
    session.install("-U", "pip", "setuptools", "wheel")
    session.install(
        "-U", "-r", "requirements.txt", "pytest-randomly", "coverage"
    )
    codecov_token = os.environ.get("CODECOV_TOKEN", sys.argv[-1])
    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "--ff",
        "-vv",
        "-r",
        "A",
        # "-l",
        "--color=yes",
        "--code-highlight=yes",
        "--continue-on-collection-errors",
    )
    if os.name == "nt":
        raise RuntimeError("cannot upload coverage results on Windows")
    session.run(
        "curl",
        "-O",
        "-s",
        "https://uploader.codecov.io/latest/linux/codecov",
    )
    session.run("chmod", "+x", "codecov")
    session.run("./codecov", env={"CODECOV_TOKEN": codecov_token})


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
        # "-l",
        "--color=yes",
        "--code-highlight=yes",
        "--continue-on-collection-errors",
    )
