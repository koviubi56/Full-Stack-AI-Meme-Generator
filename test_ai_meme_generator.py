"""
Test the AI meme generator.

Copyright (C) 2023  Koviubi56

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import base64
import io
import pathlib
import sys
import warnings
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock

import pytest

import AIMemeGenerator

MEME_TEXT = "When the H"
IMAGE_PROMPT = "The letter H"
AI_RESPONSE = f"Meme Text: {MEME_TEXT}\nImage Prompt: {IMAGE_PROMPT}"
TEST_IMAGE = pathlib.Path(__file__, "..", "test_image.png").resolve()
TEST_FULL_MEME = "test_full_meme_{}.png"


def mock_set_file_path(
    base_name: str, output_directory: pathlib.Path
) -> pathlib.Path:
    path = pathlib.Path(output_directory, f"{base_name}.png").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_font() -> Optional[pathlib.Path]:
    try:
        return AIMemeGenerator.check_font("arial.ttf").resolve()
    except Exception:
        path = pathlib.Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf")
        if path.exists():
            return path.resolve()
        warnings.warn(  # noqa: B028
            "!*! COULD NOT FIND A USABLE FONT !*! Skipping image generation"
            " and checking."
        )


def get_full_meme(font: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
    if font:
        full_meme = pathlib.Path(
            __file__, "..", TEST_FULL_MEME.format(font.stem)
        ).resolve()
        if full_meme.exists():
            return full_meme
        warnings.warn(  # noqa: B028
            f"the font {font} does not have a corresponding full meme file"
            f" {full_meme}! Skipping image testing..."
        )
    return None


def do_test_generate(
    *,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    api_keys: AIMemeGenerator.APIKeys,
    text_platform: str,
    image_platform: str,
    use_config: bool,
    use_api_keys_file: bool,
    use_cli_arguments: bool,
    no_user_input: bool,
) -> None:
    font = get_font()
    text_model = (
        "gpt-4"
        if text_platform == "openai"
        else "ggml-model-gpt4all-falcon-q4_0.bin"
    )
    monkeypatch.chdir(tmp_path)
    if use_api_keys_file:
        pathlib.Path(tmp_path, AIMemeGenerator.API_KEYS_FILE_NAME).write_text(
            f"""
[keys]
openai = {repr(api_keys.openai_key) if api_keys.openai_key else '""'}
clipdrop = {repr(api_keys.clipdrop_key) if api_keys.clipdrop_key else '""'}
stabilityai =\
 {repr(api_keys.stability_key) if api_keys.stability_key else '""'}
""",
            encoding="utf-8",
        )
        kwargs = {}
    elif not use_cli_arguments:
        kwargs = {
            "openai_key": api_keys.openai_key,
            "stability_key": api_keys.stability_key,
            "clipdrop_key": api_keys.clipdrop_key,
        }
    else:
        kwargs = {}

    with monkeypatch.context() as monkey:
        # // mock_set_api = Mock(return_value=None)
        # // monkey.setattr(
        # //     AIMemeGenerator.OpenAIText, "_set_api_key", mock_set_api
        # // )
        # // monkey.setattr(
        # //     AIMemeGenerator.OpenAIImage, "_set_api_key", mock_set_api
        # // )

        mock_message = MagicMock()
        mock_message.choices[0].message.content = AI_RESPONSE
        mock_chat_create = Mock(return_value=mock_message)

        mock_image = {
            "data": [{"b64_json": base64.b64encode(TEST_IMAGE.read_bytes())}]
        }
        mock_openai_image_create = Mock(return_value=mock_image)

        mock_openai = Mock()
        mock_openai.ChatCompletion.create = mock_chat_create
        mock_openai.Image.create = mock_openai_image_create
        monkey.setattr(
            AIMemeGenerator.OpenAIText,
            "_chat_completion_create",
            mock_chat_create,
        )
        monkey.setattr(
            AIMemeGenerator.OpenAIImage,
            "_image_create",
            mock_openai_image_create,
        )

        # ---

        class _GPT4All:
            def __init__(
                *_,  # noqa: ANN002
                **__,  # noqa: ANN003
            ) -> None:
                pass

            def generate(
                *_,  # noqa: ANN002
                **__,  # noqa: ANN003
            ) -> str:
                return AI_RESPONSE

        monkey.setattr(
            AIMemeGenerator.GPT4AllText,
            "_get_model",
            Mock(side_effect=_GPT4All),
        )

        # ---

        mock_artifact = Mock()
        mock_artifact.type = 1  # ARTIFACT_IMAGE
        mock_artifact.binary = TEST_IMAGE.read_bytes()
        mock_response = Mock()
        mock_response.artifacts = [mock_artifact]

        class Stabilityapi:
            @staticmethod
            def generate(
                prompt: str,
                steps: int,
                cfg_scale: float,
                width: int,
                height: int,
                samples: int,
                sampler: int,
            ) -> List[object]:
                assert prompt == IMAGE_PROMPT
                assert steps == 30
                assert cfg_scale == 7.0
                assert width == 1024
                assert height == 1024
                assert samples == 1
                assert sampler == 9  ## SAMPLER_K_DPMPP_2M
                return [mock_response]

        monkey.setattr(
            AIMemeGenerator.StabilityImage,
            "_get_interface",
            Mock(side_effect=Stabilityapi),
        )

        # ---

        class HTTPResponse:
            content = TEST_IMAGE.read_bytes()

            @staticmethod
            def raise_for_status() -> None:
                pass

        monkey.setattr(
            AIMemeGenerator.ClipdropImage,
            "_request",
            Mock(side_effect=HTTPResponse),
        )

        # ---

        mock_file = Mock(side_effect=mock_set_file_path)
        monkey.setattr(AIMemeGenerator, "set_file_path", mock_file)
        monkey.setattr(sys, "argv", [])
        if not font:
            monkey.setattr(
                AIMemeGenerator, "check_font", Mock(return_value=None)
            )
            monkey.setattr(
                AIMemeGenerator,
                "create_meme",
                Mock(return_value=io.BytesIO(b"")),
            )

        # ---

        if not no_user_input:
            mock_input = Mock(
                side_effect=lambda prompt: "" if ">" in prompt else "n"
            )
            monkey.setattr(AIMemeGenerator, "input", mock_input, raising=False)

        if use_cli_arguments:

            class Parser:
                @staticmethod
                def parse_args(
                    *_, **__  # noqa: ANN002, ANN003
                ) -> argparse.Namespace:
                    return argparse.Namespace(
                        text_generation_service=text_platform,
                        text_model=text_model,
                        image_platform=image_platform,
                        openai_key=api_keys.openai_key,
                        clipdrop_key=api_keys.clipdrop_key,
                        stability_key=api_keys.stability_key,
                        temperature=1.0,
                        basic_instructions="You will create funny memes that"
                        " are clever and original, and not cliche or lame.",
                        image_special_instructions="The images should be"
                        " photographic.",
                        no_file_save=False,
                        no_user_input=no_user_input,
                        user_prompt="anything",
                        meme_count=1,
                    )

            monkey.setattr(AIMemeGenerator, "parser", Parser)
        else:
            kwargs["no_user_input"] = no_user_input
        if use_config:
            image_platform_string = (
                ""
                if use_cli_arguments
                else f"image_platform = {image_platform!r}"
            )
            text_platform_string = (
                ""
                if use_cli_arguments
                else f"text_generation_service = {text_platform!r}"
            )
            text_model_string = (
                "" if use_cli_arguments else f"text_model = {text_model!r}"
            )
            font_line = f"font_file = '{font.resolve()}'" if font else ""
            pathlib.Path(AIMemeGenerator.SETTINGS_FILE_NAME).write_text(
                f"""
[ai_settings]
{text_platform_string}
{text_model_string}
{image_platform_string}
[advanced]
{font_line}
output_directory = '{pathlib.Path().cwd().resolve()}'
use_this_config = true
""",
                encoding="utf-8",
            )
            print(
                "cfg file",
                pathlib.Path(AIMemeGenerator.SETTINGS_FILE_NAME).read_text(),
            )
            if use_api_keys_file:
                memes = AIMemeGenerator.generate(no_user_input=no_user_input)
            else:
                memes = AIMemeGenerator.generate(
                    **kwargs,
                )
        else:
            if not use_cli_arguments:
                kwargs["text_generation_service"] = text_platform
                kwargs["text_model"] = text_model
                kwargs["image_platform"] = image_platform
            memes = AIMemeGenerator.generate(
                output_directory=pathlib.Path.cwd().resolve(),
                font_file_name=str(font),
                **kwargs,
            )

    # Check output
    assert len(memes) == 1
    assert memes[0].meme_text == MEME_TEXT
    assert memes[0].image_prompt == IMAGE_PROMPT
    full_meme = get_full_meme(font)
    if full_meme:
        assert memes[0].virtual_meme_file.getvalue() == full_meme.read_bytes()
    # Check mocks
    if text_platform == "openai":
        mock_chat_create.assert_called_once()
    if image_platform == "openai":
        mock_openai_image_create.assert_called_once_with()
    else:
        mock_openai_image_create.assert_not_called()
    mock_file.assert_called_once_with("meme", tmp_path.resolve())


BIG_LIST = [
    (False, False, False, False),
    (False, False, False, True),
    (False, False, True, False),
    (False, False, True, True),
    (False, True, False, False),
    (False, True, False, True),
    (False, True, True, False),
    (False, True, True, True),
    (True, False, False, False),
    (True, False, False, True),
    (True, False, True, False),
    (True, False, True, True),
    (True, True, False, False),
    (True, True, False, True),
    (True, True, True, False),
    (True, True, True, True),
]

TEXT_PLATFORMS = ("openai", "gpt4all")
IMAGE_PLATFORMS = ("openai", "clipdrop", "stability")
TEXT_IMAGE_COMBINATIONS = [
    (text, image) for image in IMAGE_PLATFORMS for text in TEXT_PLATFORMS
]
assert len(TEXT_IMAGE_COMBINATIONS) == len(TEXT_PLATFORMS) * len(
    IMAGE_PLATFORMS
)


@pytest.mark.parametrize(("config", "keys", "cli", "no_input"), BIG_LIST)
def test_generate(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    config: bool,
    keys: bool,
    cli: bool,
    no_input: bool,
) -> None:
    for text, image in TEXT_IMAGE_COMBINATIONS:
        do_test_generate(
            text_platform=text,
            tmp_path=tmp_path_factory.mktemp("tmp"),
            monkeypatch=monkeypatch,
            api_keys=AIMemeGenerator.APIKeys(
                "openai", "clipdrop", "stability"
            ),
            image_platform=image,
            use_config=config,
            use_api_keys_file=keys,
            use_cli_arguments=cli,
            no_user_input=no_input,
        )


def test_get_api_keys(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AIMemeGenerator.MissingAPIKeyError):
        AIMemeGenerator.get_api_keys(no_user_input=True)

    with monkeypatch.context() as monkey:
        input_monkey = Mock(return_value="")
        monkey.setattr(AIMemeGenerator, "input", input_monkey, raising=False)
        with pytest.raises(SystemExit):
            AIMemeGenerator.get_api_keys(no_user_input=False)
        assert (
            pathlib.Path(
                tmp_path, AIMemeGenerator.API_KEYS_FILE_NAME
            ).read_bytes()
            == AIMemeGenerator.get_assets_file(
                AIMemeGenerator.DEFAULT_API_KEYS_FILE_NAME
            ).read_bytes()
        )

    assert AIMemeGenerator.get_api_keys(
        args=argparse.Namespace(
            openai_key="openai",
            stability_key="stability",
            clipdrop_key="clipdrop",
        ),
        no_user_input=True,
    ) == AIMemeGenerator.APIKeys("openai", "clipdrop", "stability")


def test_check_font() -> None:
    with pytest.raises(AIMemeGenerator.NoFontFileError):
        AIMemeGenerator.check_font("_doesnt_exist")


def test_get_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.chdir(tmp_path)

    assert AIMemeGenerator.get_settings(True) == {}

    # ---

    with monkeypatch.context() as monkey:
        input_mock = Mock(return_value="")
        monkey.setattr(AIMemeGenerator, "input", input_mock, raising=False)

        assert AIMemeGenerator.get_settings(
            False
        ) == AIMemeGenerator.tomllib.loads(
            AIMemeGenerator.get_assets_file(
                AIMemeGenerator.DEFAULT_SETTINGS_FILE_NAME
            ).read_text(encoding="utf-8")
        )

    # ---

    pathlib.Path(AIMemeGenerator.SETTINGS_FILE_NAME).write_text(
        "invalid syntax :(", encoding="utf-8"
    )

    assert AIMemeGenerator.get_settings(True) == {}
    assert AIMemeGenerator.get_settings(False) == {}


def test_image_generation_request() -> None:
    with pytest.raises(AIMemeGenerator.InvalidImagePlatformError):
        AIMemeGenerator.image_generation_request(
            AIMemeGenerator.APIKeys("", None, None), "", "_doesnt_exist"
        )
