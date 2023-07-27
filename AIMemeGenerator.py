#!/usr/bin/env python3
"""
AI Meme Generator.

Creates start-to-finish memes using various AI service APIs. OpenAI's chatGPT
to generate the meme text and image prompt, and several optional image
generators for the meme picture. Then combines the meme text and image into a
meme using Pillow.

Originally created by ThioJoe <github.com/ThioJoe/Full-Stack-AI-Meme-Generator>
Modified by Koviubi56 in 2023.

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
__version__ = "1.0.1"

import argparse
import base64
import dataclasses
import datetime
import io
import os
import pathlib
import platform
import re
import shutil
import sys
import textwrap
import traceback
from typing import Any, Dict, Iterable, List, Literal, Optional

import colorama
import openai
import requests
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import termcolor
from openai.error import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from stability_sdk import client

try:
    import tomllib  # novermin
except Exception:
    import tomli as tomllib

colorama.init()

SETTINGS_FILE_NAME = "settings.toml"
DEFAULT_SETTINGS_FILE_NAME = "settings_default.toml"
API_KEYS_FILE_NAME = "api_keys.toml"
DEFAULT_API_KEYS_FILE_NAME = "api_keys_empty.toml"


# =============================================================================


class MemeGeneratorError(RuntimeError):
    """Base class for all AIMemeGenerator exceptions."""


@dataclasses.dataclass(frozen=True)
class NoFontFileError(MemeGeneratorError):
    """Could not find the font file."""

    font_file: str

    def __str__(self) -> str:
        return (
            f"Font file {self.font_file!r} not found. Please add the font file"
            " to the same folder as this script. Or set the variable above to"
            " the name of a font file in the system font folder."
        )


@dataclasses.dataclass(frozen=True)
class MissingAPIKeyError(MemeGeneratorError):
    """A required API key is missing."""

    api: Literal["openai", "stability", "clipdrop"]

    def __str__(self) -> str:
        return (
            f"{self.api} is expecting an API key, but no {self.api} API key"
            " was found in the api_keys.toml file."
        )


@dataclasses.dataclass(frozen=True)
class InvalidImagePlatformError(MemeGeneratorError):
    """Invalid image platform."""

    image_platform: str

    def __str__(self) -> str:
        return f"Invalid image platform {self.image_platform!r}."


@dataclasses.dataclass(frozen=True)
class APIKeys:
    """
    The API keys.

    Args:
        openai_key (str): OpenAI API key.
        clipdrop_key (Optional[str]): ClipDrop API key.
        stability_key (Optional[str]): Stability API key.
    """

    openai_key: str
    clipdrop_key: Optional[str]
    stability_key: Optional[str]


@dataclasses.dataclass(frozen=True)
class Meme:
    """
    A dictionary containing the meme's text and image prompt.

    Args:
        meme_text (str): The meme's text.
        image_prompt (str): The image prompt.
    """

    meme_text: str
    image_prompt: str


@dataclasses.dataclass(frozen=True)
class FullMeme:
    """
    A full meme.

    Args:
        meme_text (str): The meme's text.
        image_prompt (str): The image's prompt.
        virtual_meme_file (io.BytesIO): The virtual meme image file.
        file (pathlib.Path): The meme image file.
    """

    meme_text: str
    image_prompt: str
    virtual_meme_file: io.BytesIO
    file: pathlib.Path


# ============================= Argument Parser ===============================
# Parse the arguments at the start of the script
parser = argparse.ArgumentParser()
parser.add_argument("--openai-key", help="OpenAI API key")
parser.add_argument("--clipdrop-key", help="ClipDrop API key")
parser.add_argument("--stability-key", help="Stability AI API key")
parser.add_argument(
    "--user-prompt",
    help="A meme subject or concept to send to the chat bot. If not specified,"
    " the user will be prompted to enter a subject or concept.",
)
parser.add_argument(
    "--meme-count",
    help="The number of memes to create. If using arguments and not specified,"
    " the default is 1.",
)
parser.add_argument(
    "--image-platform",
    help="The image platform to use. If using arguments and not specified, the"
    " default is 'clipdrop'. Possible options: 'openai', 'stability',"
    " 'clipdrop'",
)
parser.add_argument(
    "--temperature",
    help="The temperature to use for the chat bot. If using arguments and not"
    " specified, the default is 1.0",
)
parser.add_argument(
    "--basic-instructions",
    help="The basic instructions to use for the chat bot. If using arguments"
    " and not specified, default will be used.",
)
parser.add_argument(
    "--image-special-instructions",
    help="The image special instructions to use for the chat bot. If using"
    " arguments and not specified, default will be used",
)
# These don't need to be specified as true/false, just specifying them will
# set them to true
parser.add_argument(
    "--no-user-input",
    action="store_true",
    help="Will prevent any user input prompts, and will instead use default"
    " values or other arguments.",
)
parser.add_argument(
    "--no-file-save",
    action="store_true",
    help="If specified, the meme will not be saved to a file, and only"
    " returned as virtual file part of memeResultsDictsList.",
)


# ==================== Run Checks and Import Configs  =========================


def search_for_file(
    directory: pathlib.Path, file_name: str
) -> Optional[pathlib.Path]:
    """
    Search for the file `file_name` within `directory`.

    Args:
        directory (pathlib.Path): The directory to search in.
        file_name (str): The file's name to look for.

    Returns:
        Optional[pathlib.Path]: The first file that matched `file_name` or
        None.
    """
    try:
        return next(directory.rglob(file_name))
    except StopIteration:
        return None


def search_for_file_in_directories(
    directories: Iterable[pathlib.Path], file_name: str
) -> Optional[pathlib.Path]:
    """
    Search for the file `file_name` within `directories`.

    Args:
        directories (Iterable[pathlib.Path]): The directories to search in.
        file_name (str): The file's name to look for.

    Returns:
        Optional[pathlib.Path]: The first file that matched `file_name` or
        None.
    """
    for directory in directories:
        file = search_for_file(directory, file_name)
        if file:
            return file
    return None


def check_font(font_file_name: str) -> pathlib.Path:
    """
    Check for font file in current directory, then check for font file in Fonts
    folder, warn user and exit if not found.

    Args:
        font_file_name (str): The font file's name.
        no_user_input (bool): Don't ask for user input.

    Returns:
        pathlib.Path: The font file.
    """
    # Check for font file in current directory
    termcolor.cprint("Checking the font...", "black")
    path = pathlib.Path(font_file_name)
    if path.exists():
        return path

    if platform.system() == "Windows":
        # Check for font file in Fonts folder (Windows)
        file = pathlib.Path(os.environ["WINDIR"], "Fonts", font_file_name)
    elif platform.system() == "Linux":
        # Check for font file in font directories (Linux)
        font_directories = [
            pathlib.Path("/usr/share/fonts"),
            pathlib.Path("~/.fonts").expanduser(),
            pathlib.Path("~/.local/share/fonts").expanduser(),
            pathlib.Path("/usr/local/share/fonts"),
        ]
        file = search_for_file_in_directories(font_directories, font_file_name)
    elif (
        platform.system() == "Darwin"
    ):  # Darwin is the underlying system for macOS
        # Check for font file in font directories (macOS)
        font_directories = [
            pathlib.Path("/Library/Fonts"),
            pathlib.Path("~/Library/Fonts").expanduser(),
        ]
        file = search_for_file_in_directories(font_directories, font_file_name)
    else:
        file = None

    # Warn user and exit if not found
    if (not file) or (not file.exists()):
        raise NoFontFileError(font_file_name)
    # Return the font file path
    return file


def get_config(
    config_file: pathlib.Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of the config file.

    Args:
        config_file (pathlib.Path): The config file.

    Returns:
        Dict[str, Dict[str, Any]]: The settings read from the file.
    """
    with config_file.open("rb") as file:
        return tomllib.load(file)


def get_assets_file(file_name: str) -> pathlib.Path:
    """
    Get `assets/file_name`

    Args:
        file_name (str): The file's name.

    Returns:
        pathlib.Path: The asset file.
    """
    if hasattr(sys, "_MEIPASS"):  # If running as a pyinstaller bundle
        return pathlib.Path(sys._MEIPASS, file_name).resolve()  # noqa: SLF001
    return pathlib.Path(__file__, "..", "assets", file_name).resolve()


def get_settings(no_user_input: bool) -> Dict[str, Dict[str, Any]]:
    """
    Get the settings. Create the file if it doesn't exist.

    Args:
        no_user_input (bool): Don't ask for user input

    Returns:
        Dict[str, Dict[str, Any]]: The settings.
    """
    termcolor.cprint("Getting settings...", "black")
    file = pathlib.Path(SETTINGS_FILE_NAME).resolve()

    if not file.exists():
        file_to_copy_path = get_assets_file(DEFAULT_SETTINGS_FILE_NAME)
        shutil.copyfile(file_to_copy_path, SETTINGS_FILE_NAME)
        termcolor.cprint(
            "Settings file not found, so default"
            f" '{SETTINGS_FILE_NAME}' file created. You can use it going"
            " forward to change more advanced settings if you want.",
            "cyan",
        )
        if not no_user_input:
            input("Press Enter to continue...")

    # Try to get settings file, if fails, use default settings
    try:
        settings = get_config(file)
    except Exception:
        termcolor.cprint(traceback.format_exc(), "yellow")
        termcolor.cprint(
            "ERROR: Could not read settings file. Using default settings"
            " instead.",
            "red",
        )
        settings = get_config(get_assets_file(DEFAULT_SETTINGS_FILE_NAME))

    # If something went wrong and empty settings, will use default settings
    if settings == {}:
        termcolor.cprint(
            "ERROR: Something went wrong reading the settings file. Using"
            " default settings instead.",
            "red",
        )
        settings = get_config(get_assets_file(DEFAULT_SETTINGS_FILE_NAME))

    termcolor.cprint(
        f"Config file {file} exists, using that ({settings})", "black"
    )
    return settings


def get_api_keys(
    args: Optional[argparse.Namespace] = None, no_user_input: bool = False
) -> APIKeys:
    """
    Get API key constants from config file or command line arguments.

    Args:
        args (Optional[argparse.Namespace], optional): The command line
        namespace. Defaults to None.
        no_user_input (bool, optional): Don't ask for user input. Defaults to
        False.

    Returns:
        APIKeys: The API keys.
    """
    # Checks if api_keys.toml file exists, if not create empty one from default
    file = pathlib.Path(API_KEYS_FILE_NAME)

    # Default values
    openai_key, clipdrop_key, stability_key = None, None, None

    # Checks if any arguments are not None, and uses those values if so
    if args and any((args.openai_key, args.clipdrop_key, args.stability_key)):
        openai_key = args.openai_key if args.openai_key else openai_key
        clipdrop_key = args.clipdrop_key if args.clipdrop_key else clipdrop_key
        stability_key = (
            args.stability_key if args.stability_key else stability_key
        )
        return APIKeys(openai_key, clipdrop_key, stability_key)

    # Try to read keys from config file. Default value of '' will be used if
    # not found
    if file.exists():
        keys_dict = get_config(file).get("keys", {})
        openai_key = keys_dict.get("openai", None) or None
        clipdrop_key = keys_dict.get("clipdrop", None) or None
        stability_key = keys_dict.get("stabilityai", None) or None
        return APIKeys(openai_key, clipdrop_key, stability_key)

    # Copy default empty keys file from assets folder.
    file_to_copy_path = get_assets_file(DEFAULT_API_KEYS_FILE_NAME)
    shutil.copyfile(file_to_copy_path, API_KEYS_FILE_NAME)
    termcolor.cprint(
        "Because running for the first time,"
        f' "{API_KEYS_FILE_NAME}" was created. Please add your API keys to'
        " the API Keys file.",
        "cyan",
    )
    if not no_user_input:
        input("Press Enter to exit...")
    sys.exit(1)


# ------------ VALIDATION ------------


def validate_api_keys(
    api_keys: APIKeys,
    image_platform: str,
) -> None:
    """
    Validate `api_keys`.

    Args:
        api_keys (APIKeys): The API keys.
        image_platform (str): The image platform to use.
        no_user_input (bool): Don't ask for user input
    """
    termcolor.cprint("Validating the API keys...", "black")
    if not api_keys.openai_key:
        raise MissingAPIKeyError("openai")

    valid_image_platforms = ["openai", "stability", "clipdrop"]
    image_platform = image_platform.lower()

    if image_platform not in valid_image_platforms:
        raise InvalidImagePlatformError(image_platform)
    if image_platform == "stability" and not api_keys.stability_key:
        raise MissingAPIKeyError("stability")
    if image_platform == "clipdrop" and not api_keys.clipdrop_key:
        raise MissingAPIKeyError("clipdrop")


def initialize_api_clients(
    api_keys: APIKeys, image_platform: str
) -> Optional[client.StabilityInference]:
    """
    Initialize the API clients.

    Args:
        api_keys (APIKeys): The API keys.
        image_platform (str): The image platform to use.

    Returns:
        Optional[client.StabilityInference]: If the stability API key is
        provided and the image platform is stability, return the stability
        interface, otherwise None.
    """
    termcolor.cprint("Initializing API clients...", "black")
    if api_keys.openai_key:
        openai.api_key = api_keys.openai_key

    if api_keys.stability_key and image_platform.lower() == "stability":
        return client.StabilityInference(
            key=api_keys.stability_key,  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-1024-v0-9",
        )
    return None


# ================================== Functions ================================


def set_file_path(
    base_name: str, output_directory: pathlib.Path
) -> pathlib.Path:
    """
    Sets the name and path of the file to be used.

    Args:
        base_name (str): The base name for the file.
        output_directory (pathlib.Path): The directory to put the file in.

    Returns:
        pathlib.Path: The new file.
    """
    # Generate a timestamp string to append to the file name
    timestamp = datetime.datetime.now().strftime("%f")  # noqa: DTZ005

    # If the output folder does not exist, create it
    output_directory.mkdir(parents=True, exist_ok=True)

    return pathlib.Path(output_directory, f"{base_name}_{timestamp}.png")


def write_log_file(
    user_prompt: str,
    ai_meme: Meme,
    file: pathlib.Path,
    log_directory: pathlib.Path,
    basic: str,
    special: str,
    platform: str,
) -> None:
    """
    Write or append log file containing the user user message, chat bot meme
    text, and chat bot image prompt for each meme.

    Args:
        user_prompt (str): The user prompt.
        ai_meme (Meme): The AI meme.
        file (pathlib.Path): The meme file.
        log_directory (pathlib.Path): The log directory.
        basic (str): The basic AI instruction.
        special (str): The special AI instruction.
        platform (str): The image generation platform.
    """
    # Get file name from path
    meme_file_name = file.name
    with log_directory.joinpath("log.txt").open(
        "a", encoding="utf-8"
    ) as log_file:
        log_file.write(
            textwrap.dedent(
                f"""
                Meme File Name: {meme_file_name}
                AI Basic Instructions: {basic}
                AI Special Image Instructions: {special}
                User Prompt: '{user_prompt}'
                Chat Bot Meme Text: {ai_meme.meme_text}
                Chat Bot Image Prompt: {ai_meme.image_prompt}
                Image Generation Platform: {platform}
                \n"""
            )
        )


def construct_system_prompt(
    basic_instructions: str, image_special_instructions: str
) -> str:
    """
    Construct the system prompt for the chat bot.

    Args:
        basic_instructions (str): The basic AI instructions.
        image_special_instructions (str): The special AI instructions.

    Returns:
        str: The system prompt.
    """
    format_instructions = (
        "You are a meme generator with the following formatting instructions."
        " Each meme will consist of text that will appear at the top, and an"
        " image to go along with it. The user will send you a message with a"
        " general theme or concept on which you will base the meme. The user"
        ' may choose to send you a text saying something like "anything" or'
        ' "whatever you want", or even no text at all, which you should not'
        " take literally, but take to mean they wish for you to come up with"
        " something yourself.  The memes don't necessarily need to start with"
        ' "when", but they can. In any case, you will respond with two things:'
        " First, the text of the meme that will be displayed in the final"
        " meme. Second, some text that will be used as an image prompt for an"
        " AI image generator to generate an image to also be used as part of"
        " the meme. You must respond only in the format as described next,"
        " because your response will be parsed, so it is important it conforms"
        ' to the format. The first line of your response should be: "Meme'
        ' Text: " followed by the meme text. The second line of your response'
        ' should be: "Image Prompt: " followed by the image prompt text.  ---'
        " Now here are additional instructions... "
    )
    basic_instruction_append = (
        "Next are instructions for the overall approach you should take to"
        " creating the memes. Interpret as best as possible:"
        f" {basic_instructions} | "
    )
    special_instructions_append = (
        "Next are any special instructions for the image prompt. For example,"
        ' if the instructions are "the images should be photographic style",'
        ' your prompt may append ", photograph" at the end, or begin with'
        ' "photograph of". It does not have to literally match the instruction'
        f" but interpret as best as possible: {image_special_instructions}"
    )

    return (
        format_instructions
        + basic_instruction_append
        + special_instructions_append
    )


def parse_meme(message: str) -> Optional[Meme]:
    """
    Gets the meme text and image prompt from the message sent by the chat
    bot.

    Args:
        message (str): The AI message.

    Returns:
        Optional[MemeDict]: The meme dictionary or None.
    """
    # The regex pattern to match
    pattern = r"Meme Text: (\"(.*?)\"|(.*?))\n*\s*Image Prompt: (.*?)$"

    match = re.search(pattern, message, re.DOTALL)

    if match:
        # If meme text is enclosed in quotes it will be in group 2, otherwise,
        # it will be in group 3.
        meme_text = (
            match.group(2) if match.group(2) is not None else match.group(3)
        )

        return Meme(meme_text=meme_text, image_prompt=match.group(4))
    return None


def send_and_receive_message(
    text_model: str,
    user_message: str,
    conversation_temp: List[Dict[str, str]],
    temperature: float = 1.0,
) -> str:
    """
    Sends the user message to the chat bot and returns the chat bot's response.

    Args:
        text_model (str): The text model to use.
        user_message (str): The user message.
        conversation_temp (List[Dict[str, str]]): Messages.
        temperature (float, optional): The temperature (randomness). Defaults
        to 1.0.

    Returns:
        str: The AI's response
    """
    # Prepare to send request along with context by appending user message to
    # previous conversation
    conversation_temp.append({"role": "user", "content": user_message})

    termcolor.cprint("  Sending request to write meme...", "cyan")
    try:
        chat_response = openai.ChatCompletion.create(
            model=text_model,
            messages=conversation_temp,
            temperature=temperature,
        )
    except RateLimitError:
        termcolor.cprint(
            "hint: Did you setup payment? See <https://openai.com/pricing>",
            "cyan",
        )
        raise

    return chat_response.choices[0].message.content


def create_meme(
    image_path: io.BytesIO,
    top_text: str,
    file_path: pathlib.Path,
    font_file: pathlib.Path,
    no_file_save: bool = False,
    min_scale: float = 0.05,
    buffer_scale: float = 0.03,
    font_scale: float = 1,
) -> io.BytesIO:
    """
    Create the meme image.

    Args:
        image_path (io.BytesIO): The virtual image file.
        top_text (str): Top text.
        file_path (pathlib.Path): The file to write the image to.
        font_file (pathlib.Path): The font file.
        no_file_save (bool, optional): Don't save the file to `file_path`.
        Defaults to False.
        min_scale (float, optional): Minimum scale. Defaults to 0.05.
        buffer_scale (float, optional): Buffer scale. Defaults to 0.03.
        font_scale (float, optional): Font scale. Defaults to 1.

    Returns:
        io.BytesIO: The virtual image file.
    """
    termcolor.cprint("  Creating meme image...", "cyan")

    # Load the image. Can be a path or a file-like object such as IO.BytesIO
    # virtual file
    image = Image.open(image_path)

    # Calculate buffer size based on buffer_scale
    buffer_size = int(buffer_scale * image.width)

    # Get a drawing context
    d = ImageDraw.Draw(image)

    # Split the text into words
    words = top_text.split()

    # Initialize the font size and wrapped text
    font_size = int(font_scale * image.width)
    fnt = ImageFont.truetype(str(font_file), font_size)
    wrapped_text = top_text

    # Try to fit the text on a single line by reducing the font size
    while (
        d.textbbox((0, 0), wrapped_text, font=fnt)[2]
        > image.width - 2 * buffer_size
    ):
        font_size *= 0.9  # Reduce the font size by 10%
        if font_size < min_scale * image.width:
            # If the font size is less than the minimum scale, wrap the text
            lines = [words[0]]
            for word in words[1:]:
                new_line = (lines[-1] + " " + word).rstrip()
                if (
                    d.textbbox((0, 0), new_line, font=fnt)[2]
                    > image.width - 2 * buffer_size
                ):
                    lines.append(word)
                else:
                    lines[-1] = new_line
            wrapped_text = "\n".join(lines)
            break
        fnt = ImageFont.truetype(str(font_file), int(font_size))

    # Calculate the bounding box of the text
    textbbox_val = d.multiline_textbbox((0, 0), wrapped_text, font=fnt)

    # Create a white band for the top text, with a buffer equal to 10% of the
    # font size
    band_height = (
        textbbox_val[3]
        - textbbox_val[1]
        + int(font_size * 0.1)
        + 2 * buffer_size
    )
    band = Image.new("RGBA", (image.width, band_height), (255, 255, 255, 255))

    # Draw the text on the white band
    d = ImageDraw.Draw(band)

    # The midpoint of the width and height of the bounding box
    text_x = band.width // 2
    text_y = band.height // 2

    d.multiline_text(
        (text_x, text_y),
        wrapped_text,
        font=fnt,
        fill=(0, 0, 0, 255),
        anchor="mm",
        align="center",
    )

    # Create a new image and paste the band and original image onto it
    new_img = Image.new("RGBA", (image.width, image.height + band_height))
    new_img.paste(band, (0, 0))
    new_img.paste(image, (0, band_height))

    if not no_file_save:
        # Save the result to a file
        new_img.save(file_path)

    # Return image as virtual file
    virtual_meme_file = io.BytesIO()
    new_img.save(virtual_meme_file, format="PNG")

    return virtual_meme_file


def image_generation_request(
    api_keys: APIKeys,
    image_prompt: str,
    platform: str,
    stability_api: Optional[client.StabilityInference] = None,
) -> io.BytesIO:
    """
    Create the image.

    Args:
        api_keys (APIKeys): The API keys.
        image_prompt (str): The image platform to use.
        platform (str): The platform to use.
        stability_api (Optional[client.StabilityInference], optional): The
        stability interface. Defaults to None.

    Raises:
        ValueError: If `platform == stability` and `not stability_api`
        ValueError: If the request activated the API's safety filters
        ValueError: If `platform == clipdrop` and `not api_keys.clipdrop`.
        ValueError: If `platform` is invalid.
        Exception: If there was some unknown error.

    Returns:
        io.BytesIO: The virtual image file.
    """
    if platform == "openai":
        openai_response = openai.Image.create(
            prompt=image_prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
        )
        # Convert image data to virtual file
        image_data = base64.b64decode(openai_response["data"][0]["b64_json"])
        virtual_image_file = io.BytesIO()
        # Write the image data to the virtual file
        virtual_image_file.write(image_data)

    elif platform == "stability":
        # Set up our initial generation parameters.
        stability_response = stability_api.generate(
            prompt=image_prompt,
            steps=30,
            cfg_scale=7.0,
            width=1024,
            height=1024,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M,
        )

        # Set up our warning to print to the console if the adult content
        # classifier is tripped. If adult content classifier is not tripped,
        # save generated images.
        for resp in stability_response:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    raise ValueError(
                        "Your request activated the API's safety filters and"
                        " could not be processed. Please modify the prompt and"
                        " try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    virtual_image_file = io.BytesIO(artifact.binary)

    elif platform == "clipdrop":
        r = requests.post(
            "https://clipdrop-api.co/text-to-image/v1",
            files={"prompt": (None, image_prompt, "text/plain")},
            headers={"x-api-key": api_keys.clipdrop_key},
            timeout=60,
        )
        r.raise_for_status()
        virtual_image_file = io.BytesIO(
            r.content
        )  # r.content contains the bytes of the returned image

    else:
        raise InvalidImagePlatformError(platform)

    try:
        return virtual_image_file
    except NameError as error:
        raise RuntimeError(
            "Could not generate image due to above error"
        ) from error


# ==================== RUN ====================


def generate(
    text_model: str = "gpt-4",
    temperature: float = 1.0,
    basic_instructions: str = "You will create funny memes that are clever and"
    " original, and not cliche or lame.",
    image_special_instructions: str = "The images should be photographic.",
    user_entered_prompt: str = "anything",
    meme_count: int = 1,
    image_platform: str = "openai",
    font_file_name: str = "arial.ttf",
    base_file_name: str = "meme",
    output_directory: str | pathlib.Path = "Outputs",
    openai_key: Optional[str] = None,
    stability_key: Optional[str] = None,
    clipdrop_key: Optional[str] = None,
    no_user_input: bool = False,
    no_file_save: bool = False,
) -> List[FullMeme]:
    """
    Generate the memes.

    Args:
        text_model (str, optional): The text model to use. Defaults to "gpt-4".
        temperature (float, optional): The temperature (randomness). Defaults
        to 1.0.
        basic_instructions (str, optional): The basic instructions. Defaults
        to "You will create funny memes that are clever and"" original, and not
        cliche or lame.".
        image_special_instructions (str, optional): The image special
        instructions. Defaults to "The images should be photographic.".
        user_entered_prompt (str, optional): The user entered prompt. Defaults
        to "anything".
        meme_count (int, optional): The amount of memes to generate. Defaults
        to 1.
        image_platform (str, optional): The image platform to use. Must be one
        of "openai", "stability", and "clipdrop". Defaults to "openai".
        font_file_name (str, optional): The font file's name to use. Defaults
        to "arial.ttf".
        base_file_name (str, optional): The base file name for the images.
        Defaults to "meme".
        output_directory (pathlib.Path, optional): The directory to put the
        images into. Defaults to pathlib.Path("Outputs").
        openai_key (Optional[str], optional): The OpenAI API key. Defaults to
        None.
        stability_key (Optional[str], optional): The stability API key.
        Defaults to None.
        clipdrop_key (Optional[str], optional): The clipdrop API key. Defaults
        to None.
        no_user_input (bool, optional): Don't ask for user input. Defaults to
        False.
        no_file_save (bool, optional): Don't save the files. Defaults to False.

    Returns:
        List[FullMeme]: The list of the memes.
        Its length may be less than `meme_count` if some memes were skipped due
        to errors.
    """
    # Display Header
    termcolor.cprint(
        f" AI Meme Generator - {__version__} ".center(
            shutil.get_terminal_size().columns, "="
        ),
        "blue",
    )
    print(
        """
AI Meme Generator
Originally created by ThioJoe <github.com/ThioJoe/Full-Stack-AI-Meme-Generator>
Modified by Koviubi56 in 2023.

Copyright (C) 2023  Koviubi56
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.
"""
    )
    settings = get_settings(no_user_input)
    use_config = settings.get("advanced", {}).get(
        "use_this_config",
        False
        # ! This MUST be set to False!
        # This is because get_settings() will return the default config if the
        # config file doesn't exist, which WOULD OVERWRITE the arguments.
    )
    if use_config:
        termcolor.cprint("Getting settings from config file...", "cyan")
        text_model = settings.get("ai_settings", {}).get(
            "text_model", text_model
        )
        temperature = float(  # float() is not necessary, but why not
            settings.get("ai_settings", {}).get("temperature", temperature)
        )
        basic_instructions = settings.get("basic", {}).get(
            "basic_instructions", basic_instructions
        )
        image_special_instructions = settings.get("basic", {}).get(
            "image_special_instructions", image_special_instructions
        )
        image_platform = settings.get("ai_settings", {}).get(
            "image_platform", image_platform
        )
        font_file_name = settings.get("advanced", {}).get(
            "font_file", font_file_name
        )
        base_file_name = settings.get("advanced", {}).get(
            "base_file_name", base_file_name
        )
        output_directory = settings.get("advanced", {}).get(
            "output_directory", output_directory
        )

    output_directory = pathlib.Path(output_directory).resolve()

    # Parse the arguments
    args = parser.parse_args()

    # If API Keys not provided as parameters, get them from config file or
    # command line arguments
    if openai_key:
        termcolor.cprint("Getting API keys from arguments...", "cyan")
        api_keys = APIKeys(openai_key, clipdrop_key, stability_key)
    else:
        termcolor.cprint("Getting API keys from file...", "cyan")
        api_keys = get_api_keys(args, no_user_input)

    # Validate api keys
    validate_api_keys(api_keys, image_platform)
    # Initialize api clients. Only get stability_api object back because
    # openai.api_key has global scope
    stability_api = initialize_api_clients(api_keys, image_platform)

    # Check if any settings arguments, and replace the default values with the
    # args if so. To run automated from command line, specify at least 1
    # argument.
    if args.image_platform:
        image_platform = args.image_platform
        termcolor.cprint(
            f"Image platform will be {image_platform} from cli arguments",
            "cyan",
        )
    if args.temperature:
        temperature = float(args.temperature)
        termcolor.cprint(
            f"Temperature will be {temperature} from cli arguments",
            "cyan",
        )
    if args.basic_instructions:
        basic_instructions = args.basic_instructions
        termcolor.cprint(
            f"Basic instructions will be {basic_instructions} from cli"
            " arguments",
            "cyan",
        )
    if args.image_special_instructions:
        image_special_instructions = args.image_special_instructions
        termcolor.cprint(
            f"Image special instructions will be {image_special_instructions}"
            " from cli arguments",
            "cyan",
        )
    if args.no_file_save:
        no_file_save = True
        termcolor.cprint(
            "Won't save file according to cli arguments",
            "cyan",
        )
    if args.no_user_input:
        no_user_input = True
        termcolor.cprint(
            "Won't ask for user input according to cli arguments",
            "cyan",
        )

    system_prompt = construct_system_prompt(
        basic_instructions, image_special_instructions
    )
    conversation = [{"role": "system", "content": system_prompt}]

    if not no_user_input:
        # If no user prompt argument set, get user input for prompt
        if args.user_prompt:
            user_entered_prompt = args.user_prompt
        else:
            print(
                "Enter a meme subject or concept (Or just hit enter to let"
                " the AI decide)"
            )
            user_entered_prompt = input(" >  ")
            if (
                not user_entered_prompt
            ):  # If user puts in nothing, set to "anything"
                user_entered_prompt = "anything"

        # If no meme count argument set, get user input for meme count
        if args.meme_count:
            meme_count = int(args.meme_count)
        else:
            # Set the number of memes to create
            meme_count = 1  # Default will be none if nothing entered
            print(
                "Enter the number of memes to create (Or just hit Enter for"
                " 1): "
            )
            user_entered_count = input(" >  ")
            if user_entered_count:
                meme_count = int(user_entered_count)

    # Get full path of font file from font file name
    font_file = check_font(font_file_name)

    def single_meme_generation_loop() -> Optional[FullMeme]:
        # Send request to chat bot to generate meme text and image prompt
        chat_response = send_and_receive_message(
            text_model, user_entered_prompt, conversation, temperature
        )

        # Take chat message and convert to dictionary with meme_text and
        # image_prompt
        meme = parse_meme(chat_response)
        if not meme:
            termcolor.cprint(
                "  Could not interpret response! Skipping", "yellow"
            )
            return None
        image_prompt = meme.image_prompt
        meme_text = meme.meme_text

        # Print the meme text and image prompt
        termcolor.cprint("  Meme Text:\t" + meme_text, "cyan")
        termcolor.cprint("  Image Prompt:\t" + image_prompt, "cyan")

        # Send image prompt to image generator and get image back
        # (Using DALL·E API)
        termcolor.cprint("  Sending image creation request...", "cyan")
        virtual_image_file = image_generation_request(
            api_keys, image_prompt, image_platform, stability_api
        )

        # Combine the meme text and image into a meme
        file = set_file_path(base_file_name, output_directory)
        termcolor.cprint("  Creating full meme...", "cyan")
        virtual_meme_file = create_meme(
            virtual_image_file,
            meme_text,
            file,
            no_file_save=no_file_save,
            font_file=font_file,
        )
        if not no_file_save:
            # Write the user message, meme text, and image prompt to a log file
            write_log_file(
                user_entered_prompt,
                meme,
                file,
                output_directory,
                basic_instructions,
                image_special_instructions,
                image_platform,
            )

        termcolor.cprint("  Done!", "cyan")
        return FullMeme(meme_text, image_prompt, virtual_meme_file, file)

    # Create list of dictionaries to hold the results of each meme so that they
    # can be returned by main() if called from command line
    meme_results_dicts_list: List[FullMeme] = []

    for number in range(1, meme_count + 1):
        while True:
            termcolor.cprint("-" * shutil.get_terminal_size().columns, "blue")
            termcolor.cprint(
                f"Generating meme {number} of {meme_count}...", "cyan"
            )
            try:
                meme_info_dict = single_meme_generation_loop()
            except Exception:  # noqa: PERF203
                termcolor.cprint("Error while generating the meme:", "red")
                if no_user_input:
                    raise
                termcolor.cprint(traceback.format_exc(), "red")
                task = input("Abort, retry, skip? [A/r/s] ").lower()
                if task == "r":
                    continue
                if task == "s":
                    break
                sys.exit(1)

            # Add meme info dict to list of meme results
            if meme_info_dict:
                meme_results_dicts_list.append(meme_info_dict)
            break

    # If called from command line, will return the list of meme results
    return meme_results_dicts_list


if __name__ == "__main__":
    try:
        generate()
    except Exception:
        termcolor.cprint(traceback.format_exc(), "red")
        termcolor.cprint(
            "Please read the above error message! If you think this is a bug,"
            " please report it on GitHub including the **entire** above"
            " traceback.",
            "yellow",
        )
        input("Press Enter to exit...")
