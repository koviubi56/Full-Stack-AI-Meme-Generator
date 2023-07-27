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
from typing import Any, Optional
from unittest.mock import MagicMock, Mock

import pytest

import AIMemeGenerator

MEME_TEXT = "When the H"
IMAGE_PROMPT = "The letter H"
AI_RESPONSE = f"Meme Text: {MEME_TEXT}\nImage Prompt: {IMAGE_PROMPT}"
TEST_IMAGE = pathlib.Path(__file__, "..", "test_image.png").resolve()
TEST_FULL_MEME = "test_full_meme_{}.png"


def mock_initialize_api_clients(
    api_keys: AIMemeGenerator.APIKeys, image_platform: str
) -> Optional[Any]:
    if api_keys.stability_key and image_platform.lower() == "stability":
        return object()
    return None


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
    image_platform: str,
    use_config: bool,
    use_api_keys_file: bool,
) -> None:
    font = get_font()
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
    else:
        kwargs = {
            "openai_key": api_keys.openai_key,
            "stability_key": api_keys.stability_key,
            "clipdrop_key": api_keys.clipdrop_key,
        }
    with monkeypatch.context() as monkey:
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
        monkey.setattr(AIMemeGenerator, "openai", mock_openai)

        # ---

        mock_artifact = Mock()
        mock_artifact.type = AIMemeGenerator.generation.ARTIFACT_IMAGE
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
            ) -> list[object]:
                assert prompt == IMAGE_PROMPT
                assert steps == 30
                assert cfg_scale == 7.0
                assert width == 1024
                assert height == 1024
                assert samples == 1
                assert sampler == AIMemeGenerator.generation.SAMPLER_K_DPMPP_2M
                return [mock_response]

        class Client:
            @staticmethod
            def StabilityInference(  # noqa: N802
                *_, **__  # noqa: ANN003, ANN002
            ) -> Stabilityapi:
                return Stabilityapi()

        monkey.setattr(
            AIMemeGenerator,
            "client",
            Client,
            raising=False,
        )

        # ---

        class HTTPResponse:
            content = TEST_IMAGE.read_bytes()

            @staticmethod
            def raise_for_status() -> None:
                pass

        class Requests:
            @staticmethod
            def post(
                url: str,
                files: dict[str, tuple[None, str, str]],
                headers: dict[str, str],
                timeout: int,
            ) -> HTTPResponse:
                assert url == "https://clipdrop-api.co/text-to-image/v1"
                assert files == {"prompt": (None, IMAGE_PROMPT, "text/plain")}
                assert headers == {"x-api-key": api_keys.clipdrop_key}
                assert timeout == 60
                return HTTPResponse()

        monkey.setattr(AIMemeGenerator, "requests", Requests)

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

        if use_config:
            font_line = f"font_file = '{font.resolve()}'" if font else ""
            pathlib.Path(AIMemeGenerator.SETTINGS_FILE_NAME).write_text(
                f"""
[ai_settings]
image_platform = {image_platform!r}
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
                memes = AIMemeGenerator.generate(no_user_input=True)
            else:
                memes = AIMemeGenerator.generate(
                    no_user_input=True,
                    **kwargs,
                )
        else:
            memes = AIMemeGenerator.generate(
                output_directory=pathlib.Path.cwd().resolve(),
                image_platform=image_platform,
                no_user_input=True,
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
    assert memes[0].file == pathlib.Path(tmp_path, "meme.png").resolve()
    assert (
        pathlib.Path(tmp_path, "log.txt").read_text(encoding="utf-8")
        == f"""
Meme File Name: meme.png
AI Basic Instructions: You will create funny memes that are clever and\
 original, and not cliche or lame.
AI Special Image Instructions: The images should be photographic.
User Prompt: 'anything'
Chat Bot Meme Text: When the H
Chat Bot Image Prompt: The letter H
Image Generation Platform: {image_platform}

"""
    )
    if not use_config:
        assert pathlib.Path(
            tmp_path, AIMemeGenerator.SETTINGS_FILE_NAME
        ).read_text(encoding="utf-8") == AIMemeGenerator.get_assets_file(
            AIMemeGenerator.DEFAULT_SETTINGS_FILE_NAME
        ).read_text(
            encoding="utf-8"
        )

    # Check mocks
    mock_chat_create.assert_called_once_with(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": AIMemeGenerator.construct_system_prompt(
                    "You will create funny memes that are clever and original,"
                    " and not cliche or lame.",
                    "The images should be photographic.",
                ),
            },
            {"role": "user", "content": "anything"},
        ],
        temperature=1.0,
    )
    if image_platform == "openai":
        mock_openai_image_create.assert_called_once_with(
            prompt=IMAGE_PROMPT,
            n=1,
            size="512x512",
            response_format="b64_json",
        )
    else:
        mock_openai_image_create.assert_not_called()
    mock_file.assert_called_once_with("meme", tmp_path.resolve())


def do_test_generate_four_times(
    *,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    api_keys: AIMemeGenerator.APIKeys,
    image_platform: str,
) -> None:
    for config, keys in [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]:
        do_test_generate(
            tmp_path=tmp_path_factory.mktemp("tmp"),
            monkeypatch=monkeypatch,
            api_keys=api_keys,
            image_platform=image_platform,
            use_config=config,
            use_api_keys_file=keys,
        )


def test_generate_only_openai(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    do_test_generate_four_times(
        tmp_path_factory=tmp_path_factory,
        monkeypatch=monkeypatch,
        image_platform="openai",
        api_keys=AIMemeGenerator.APIKeys("openai", None, None),
    )


def test_generate_openai_plus_clipdrop(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    do_test_generate_four_times(
        tmp_path_factory=tmp_path_factory,
        monkeypatch=monkeypatch,
        image_platform="clipdrop",
        api_keys=AIMemeGenerator.APIKeys("openai", "clipdrop", None),
    )


def test_generate_openai_plus_stability(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    do_test_generate_four_times(
        tmp_path_factory=tmp_path_factory,
        monkeypatch=monkeypatch,
        image_platform="stability",
        api_keys=AIMemeGenerator.APIKeys("openai", None, "stability"),
    )


def test_get_api_keys(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as monkey:
        monkey.chdir(tmp_path_factory.mktemp("tmp"))
        with pytest.raises(SystemExit):
            AIMemeGenerator.get_api_keys(no_user_input=True)
        assert pathlib.Path(AIMemeGenerator.API_KEYS_FILE_NAME).exists()
        assert AIMemeGenerator.get_api_keys(
            no_user_input=True
        ) == AIMemeGenerator.APIKeys(None, None, None)

        monkey.chdir(tmp_path_factory.mktemp("tmp"))
        assert AIMemeGenerator.get_api_keys(
            args=argparse.Namespace(
                openai_key="openai",
                stability_key="stability",
                clipdrop_key="clipdrop",
            ),
            no_user_input=True,
        ) == AIMemeGenerator.APIKeys("openai", "clipdrop", "stability")


def test_validate_api_keys() -> None:
    with pytest.raises(AIMemeGenerator.InvalidImagePlatformError):
        AIMemeGenerator.validate_api_keys(
            AIMemeGenerator.APIKeys(
                "openai",
                None,
                None,
            ),
            "asdf",
        )

    # ---

    with pytest.raises(AIMemeGenerator.MissingAPIKeyError):
        AIMemeGenerator.validate_api_keys(
            AIMemeGenerator.APIKeys(
                None,  # type: ignore  # noqa: PGH003
                None,
                None,
            ),
            "openai",
        )
    with pytest.raises(AIMemeGenerator.MissingAPIKeyError):
        AIMemeGenerator.validate_api_keys(
            AIMemeGenerator.APIKeys(
                "",
                None,
                None,
            ),
            "openai",
        )
    AIMemeGenerator.validate_api_keys(
        AIMemeGenerator.APIKeys(
            "openai",
            None,
            None,
        ),
        "openai",
    )
    with pytest.raises(AIMemeGenerator.MissingAPIKeyError):
        AIMemeGenerator.validate_api_keys(
            AIMemeGenerator.APIKeys(
                "openai",
                None,
                None,
            ),
            "clipdrop",
        )
    with pytest.raises(AIMemeGenerator.MissingAPIKeyError):
        AIMemeGenerator.validate_api_keys(
            AIMemeGenerator.APIKeys(
                "openai",
                None,
                None,
            ),
            "stability",
        )
    with pytest.raises(AIMemeGenerator.MissingAPIKeyError):
        AIMemeGenerator.validate_api_keys(
            AIMemeGenerator.APIKeys(
                "openai",
                None,
                "stability",
            ),
            "clipdrop",
        )
    with pytest.raises(AIMemeGenerator.MissingAPIKeyError):
        AIMemeGenerator.validate_api_keys(
            AIMemeGenerator.APIKeys(
                "openai",
                "clipdrop",
                None,
            ),
            "stability",
        )
    AIMemeGenerator.validate_api_keys(
        AIMemeGenerator.APIKeys(
            "openai",
            "clipdrop",
            None,
        ),
        "clipdrop",
    )
    AIMemeGenerator.validate_api_keys(
        AIMemeGenerator.APIKeys(
            "openai",
            None,
            "stability",
        ),
        "stability",
    )
    AIMemeGenerator.validate_api_keys(
        AIMemeGenerator.APIKeys(
            "openai",
            "clipdrop",
            "stability",
        ),
        "clipdrop",
    )
    AIMemeGenerator.validate_api_keys(
        AIMemeGenerator.APIKeys(
            "openai",
            "clipdrop",
            "stability",
        ),
        "stability",
    )
