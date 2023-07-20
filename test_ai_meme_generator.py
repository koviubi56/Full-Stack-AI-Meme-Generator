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
import io
import pathlib
import sys
import warnings
from typing import Any, Optional
from unittest.mock import Mock

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
    return pathlib.Path(output_directory, f"{base_name}.png").resolve()


def mock_get_assets_file(file_name: str) -> pathlib.Path:
    return pathlib.Path(__file__, "..", "assets", file_name).resolve()


def get_font() -> Optional[pathlib.Path]:
    try:
        return AIMemeGenerator.check_font("arial.ttf", True).resolve()
    except BaseException:  # may raise SystemExit
        path = pathlib.Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf")
        if path.exists():
            return path.resolve()
        warnings.warn(  # noqa: B028
            "!*! COULD NOT FIND A USABLE FONT !*! Skipping image generation"
            " and checking."
        )


def do_test_generate(
    *,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    api_keys: AIMemeGenerator.APIKeys,
    image_platform: str,
) -> None:
    font = get_font()
    monkeypatch.chdir(tmp_path)
    pathlib.Path(tmp_path, AIMemeGenerator.API_KEYS_FILE_NAME).write_text(
        f"""
[keys]
openai = "{api_keys.openai_key if api_keys.openai_key else 'null'}"
clipdrop = "{api_keys.clipdrop_key if api_keys.clipdrop_key else 'null'}"
stabilityai = "{api_keys.stability_key if api_keys.stability_key else 'null'}"
""",
        encoding="utf-8",
    )
    with monkeypatch.context() as monkey:
        mock_init = Mock(side_effect=mock_initialize_api_clients)
        monkey.setattr(AIMemeGenerator, "initialize_api_clients", mock_init)
        mock_message = Mock(return_value=AI_RESPONSE)
        monkey.setattr(
            AIMemeGenerator, "send_and_receive_message", mock_message
        )
        mock_image = Mock(return_value=io.BytesIO(TEST_IMAGE.read_bytes()))
        monkey.setattr(AIMemeGenerator, "image_generation_request", mock_image)
        mock_file = Mock(side_effect=mock_set_file_path)
        monkey.setattr(AIMemeGenerator, "set_file_path", mock_file)
        mock_asset = Mock(side_effect=mock_get_assets_file)
        monkey.setattr(AIMemeGenerator, "get_assets_file", mock_asset)
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

        memes = AIMemeGenerator.generate(
            output_folder=pathlib.Path.cwd().resolve(),
            image_platform=image_platform,
            no_user_input=True,
            font_file_name=str(font),
        )

    # Check output
    assert len(memes) == 1
    assert memes[0].meme_text == MEME_TEXT
    assert memes[0].image_prompt == IMAGE_PROMPT
    if font:
        full_meme = pathlib.Path(
            __file__, "..", TEST_FULL_MEME.format(font.stem)
        ).resolve()
        if full_meme.exists():
            assert (
                memes[0].virtual_meme_file.getvalue() == full_meme.read_bytes()
            )
        else:
            warnings.warn(  # noqa: B028
                f"the font {font} does not have a corresponding full meme file"
                f" {full_meme}! Skipping image testing..."
            )
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
    assert pathlib.Path(
        tmp_path, AIMemeGenerator.SETTINGS_FILE_NAME
    ).read_text(encoding="utf-8") == mock_get_assets_file(
        AIMemeGenerator.DEFAULT_SETTINGS_FILE_NAME
    ).read_text(
        encoding="utf-8"
    )

    # Check mocks
    mock_init.assert_called_once_with(api_keys, image_platform)
    mock_message.assert_called_once_with(
        "gpt-4",
        "anything",
        [
            {
                "role": "system",
                "content": AIMemeGenerator.construct_system_prompt(
                    "You will create funny memes that are clever and original,"
                    " and not cliche or lame.",
                    "The images should be photographic.",
                ),
            }
        ],
        1.0,
    )
    if image_platform == "stability":
        mock_image.assert_called_once()
        assert len(mock_image.call_args_list[0].args) == 4  # noqa: PLR2004
        assert mock_image.call_args_list[0].args[0] == api_keys
        assert mock_image.call_args_list[0].args[1] == IMAGE_PROMPT
        assert type(mock_image.call_args_list[0].args[3]) is object
    else:
        mock_image.assert_called_once_with(
            api_keys, IMAGE_PROMPT, image_platform, None
        )
    mock_file.assert_called_once_with("meme", tmp_path.resolve())
    mock_asset.assert_called_once_with(
        AIMemeGenerator.DEFAULT_SETTINGS_FILE_NAME
    )


def test_generate_only_openai(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    do_test_generate(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        image_platform="openai",
        api_keys=AIMemeGenerator.APIKeys("openai", None, None),
    )


def test_generate_openai_plus_clipdrop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    do_test_generate(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        image_platform="clipdrop",
        api_keys=AIMemeGenerator.APIKeys("openai", "clipdrop", None),
    )


def test_generate_openai_plus_stability(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    do_test_generate(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        image_platform="stability",
        api_keys=AIMemeGenerator.APIKeys("openai", None, "stability"),
    )
