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
from typing import Iterable, Literal, NamedTuple, TypedDict

import openai
import requests
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from openai.error import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pkg_resources import parse_version
from stability_sdk import client

try:
    import tomllib  # novermin
except Exception:
    import tomli as tomllib

SETTINGS_FILE_NAME = "settings.toml"
DEFAULT_SETTINGS_FILE_NAME = "settings_default.toml"
API_KEYS_FILE_NAME = "api_keys.toml"
DEFAULT_API_KEYS_FILE_NAME = "api_keys_empty.toml"


# =============================================================================


class MemeDict(TypedDict):
    meme_text: str
    image_prompt: str


def construct_system_prompt(
    basic_instructions: str, image_special_instructions: str
) -> str:
    """Construct the system prompt for the chat bot."""
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
    special_instructions_ippend = (
        "Next are any special instructions for the image prompt. For example,"
        ' if the instructions are "the images should be photographic style",'
        ' your prompt may append ", photograph" at the end, or begin with'
        ' "photograph of". It does not have to literally match the instruction'
        f" but interpret as best as possible: {image_special_instructions}"
    )

    return (
        format_instructions
        + basic_instruction_append
        + special_instructions_ippend
    )


# ============================= Argument Parser ===============================
# Parse the arguments at the start of the script
parser = argparse.ArgumentParser()
parser.add_argument("--openaikey", help="OpenAI API key")
parser.add_argument("--clipdropkey", help="ClipDrop API key")
parser.add_argument("--stabilitykey", help="Stability AI API key")
parser.add_argument(
    "--userprompt",
    help="A meme subject or concept to send to the chat bot. If not specified,"
    " the user will be prompted to enter a subject or concept.",
)
parser.add_argument(
    "--memecount",
    help="The number of memes to create. If using arguments and not specified,"
    " the default is 1.",
)
parser.add_argument(
    "--imageplatform",
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
    "--basicinstructions",
    help="The basic instructions to use for the chat bot. If using arguments"
    " and not specified, default will be used.",
)
parser.add_argument(
    "--imagespecialinstructions",
    help="The image special instructions to use for the chat bot. If using"
    " arguments and not specified, default will be used",
)
# These don't need to be specified as true/false, just specifying them will
# set them to true
parser.add_argument(
    "--nouserinput",
    action="store_true",
    help="Will prevent any user input prompts, and will instead use default"
    " values or other arguments.",
)
parser.add_argument(
    "--nofilesave",
    action="store_true",
    help="If specified, the meme will not be saved to a file, and only"
    " returned as virtual file part of memeResultsDictsList.",
)
args = parser.parse_args()


# Create a namedtuple class
class ApiKeys(NamedTuple):
    openai_key: str
    clipdrop_key: str | None
    stability_key: str | None


# ==================== Run Checks and Import Configs  =========================


def search_for_file(
    directory: pathlib.Path, file_name: str
) -> pathlib.Path | None:
    try:
        return next(directory.rglob(file_name))
    except StopIteration:
        return None


def search_for_file_in_directories(
    directories: Iterable[pathlib.Path], file_name: str
) -> pathlib.Path | None:
    for directory in directories:
        file = search_for_file(directory, file_name)
        if file:
            return file
    return None


def check_font(font_file: str) -> pathlib.Path:
    """
    Check for font file in current directory, then check for font file in
    Fonts folder, warn user and exit if not found
    """
    # Check for font file in current directory
    path = pathlib.Path(font_file)
    if path.exists():
        return path

    if platform.system() == "Windows":
        # Check for font file in Fonts folder (Windows)
        file = pathlib.Path(os.environ["WINDIR"], "Fonts", font_file)
    elif platform.system() == "Linux":
        # Check for font file in font directories (Linux)
        font_directories = [
            pathlib.Path("/usr/share/fonts"),
            pathlib.Path("~/.fonts").expanduser(),
            pathlib.Path("~/.local/share/fonts").expanduser(),
            pathlib.Path("/usr/local/share/fonts"),
        ]
        file = search_for_file_in_directories(font_directories, font_file)
    elif (
        platform.system() == "Darwin"
    ):  # Darwin is the underlying system for macOS
        # Check for font file in font directories (macOS)
        font_directories = [
            pathlib.Path("/Library/Fonts"),
            pathlib.Path("~/Library/Fonts").expanduser(),
        ]
        file = search_for_file_in_directories(font_directories, font_file)
    else:
        file = None

    # Warn user and exit if not found
    if (file is None) or (not file.exists()):
        print(
            f'\n  ERROR:  Font file "{font_file}" not found. Please add the'
            " font file to the same folder as this script. Or set the variable"
            " above to the name of a font file in the system font folder."
        )
        input("\nPress Enter to exit...")
        sys.exit()
    # Return the font file path
    return file


def get_config(config_file: pathlib.Path) -> dict[str, str | float | bool]:
    """Returns a dictionary of the config file"""
    with config_file.open("rb") as file:
        return tomllib.load(file)


def get_assets_file(file_name: str) -> pathlib.Path:
    if hasattr(sys, "_MEIPASS"):  # If running as a pyinstaller bundle
        return pathlib.Path(sys._MEIPASS, file_name)
    return pathlib.Path("assets", file_name)


def get_settings() -> dict[str, dict[str, str | float | bool]]:
    file = pathlib.Path(SETTINGS_FILE_NAME)

    if not file.exists():
        file_to_copy_path = get_assets_file(DEFAULT_SETTINGS_FILE_NAME)
        shutil.copyfile(file_to_copy_path, SETTINGS_FILE_NAME)
        print(
            "\nINFO: Settings file not found, so default"
            f" '{SETTINGS_FILE_NAME}' file created. You can use it going"
            " forward to change more advanced settings if you want."
        )
        input("\nPress Enter to continue...")

    # Try to get settings file, if fails, use default settings
    try:
        settings = get_config(file)
    except Exception:
        print(
            "\nERROR: Could not read settings file. Using default settings"
            " instead."
        )
        settings = get_config(get_assets_file(DEFAULT_SETTINGS_FILE_NAME))

    # If something went wrong and empty settings, will use default settings
    if settings == {}:
        print(
            "\nERROR: Something went wrong reading the settings file. Using"
            " default settings instead."
        )
        settings = get_config(get_assets_file(DEFAULT_SETTINGS_FILE_NAME))

    return settings


# Get API key constants from config file or command line arguments
def get_api_keys(args: argparse.Namespace | None = None) -> ApiKeys:
    # Checks if api_keys.toml file exists, if not create empty one from default
    file = pathlib.Path(API_KEYS_FILE_NAME)
    if not file.exists():
        file_to_copy_path = get_assets_file(DEFAULT_API_KEYS_FILE_NAME)
        # Copy default empty keys file from assets folder. Use absolute path
        shutil.copyfile(file_to_copy_path, API_KEYS_FILE_NAME)
        print(
            "\n  INFO:  Because running for the first time,"
            f' "{API_KEYS_FILE_NAME}" was created. Please add your API keys to'
            " the API Keys file."
        )
        input("\nPress Enter to exit...")
        sys.exit()

    # Default values
    openai_key, clipdrop_key, stability_key = "", "", ""

    # Try to read keys from config file. Default value of '' will be used if
    # not found
    try:
        keys_dict = get_config(file)
        openai_key = keys_dict.get("openai", "")
        clipdrop_key = keys_dict.get("clipdrop", "")
        stability_key = keys_dict.get("stabilityai", "")
    except FileNotFoundError:
        print(
            "Config not found, checking for command line arguments."
        )  # Could not read from config file, will try cli arguments next

    # Checks if any arguments are not None, and uses those values if so
    if args and any(vars(args).values()):
        openai_key = args.openaikey if args.openaikey else openai_key
        clipdrop_key = args.clipdropkey if args.clipdropkey else clipdrop_key
        stability_key = (
            args.stabilitykey if args.stabilitykey else stability_key
        )

    return ApiKeys(openai_key, clipdrop_key, stability_key)


# ------------ VALIDATION ------------


def validate_api_keys(
    api_keys: ApiKeys,
    image_platform: str,
) -> None:
    if not api_keys.openai_key:
        print(
            "\n  ERROR:  No OpenAI API key found. OpenAI API key is required"
            " - In order to generate text for the meme text and image prompt."
            f" Please add your OpenAI API key to the {API_KEYS_FILE_NAME}"
            " file."
        )
        input("\nPress Enter to exit...")
        sys.exit()

    valid_image_platforms = ["openai", "stability", "clipdrop"]
    image_platform = image_platform.lower()

    if image_platform not in valid_image_platforms:
        print(
            f'\n  ERROR:  Invalid image platform "{image_platform}". Valid'
            f" image platforms are: {valid_image_platforms}"
        )
        input("\nPress Enter to exit...")
        sys.exit()
    if image_platform == "stability" and not api_keys.stability_key:
        print(
            "\n  ERROR:  Stability AI was set as the image platform, but no"
            f" Stability AI API key was found in the {API_KEYS_FILE_NAME}"
            " file."
        )
        input("\nPress Enter to exit...")
        sys.exit()
    if image_platform == "clipdrop" and not api_keys.clipdrop_key:
        print(
            "\n  ERROR:  ClipDrop was set as the image platform, but no"
            f" ClipDrop API key was found in the {API_KEYS_FILE_NAME} file."
        )
        input("\nPress Enter to exit...")
        sys.exit()


def initialize_api_clients(
    api_keys: ApiKeys, image_platform: str
) -> client.StabilityInference | None:
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


# Sets the name and path of the file to be used
def set_file_path(
    base_name: str, output_directory: pathlib.Path
) -> pathlib.Path:
    # Generate a timestamp string to append to the file name
    timestamp = datetime.datetime.now().strftime("%f")  # noqa: DTZ005

    # If the output folder does not exist, create it
    output_directory.mkdir(parents=True, exist_ok=True)

    return pathlib.Path(output_directory, f"{base_name}_{timestamp}.png")


# Write or append log file containing the user user message, chat bot meme
# text, and chat bot image prompt for each meme
def write_log_file(
    user_prompt: str,
    ai_meme_dict: MemeDict,
    file: pathlib.Path,
    log_directory: pathlib.Path,
    basic: str,
    special: str,
    platform: str,
) -> None:
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
                Chat Bot Meme Text: {ai_meme_dict['meme_text']}
                Chat Bot Image Prompt: {ai_meme_dict['image_prompt']}
                Image Generation Platform: {platform}
                \n"""
            )
        )


def check_for_update(
    update_release_channel: str,
) -> bool | Literal["beta"]:
    if update_release_channel.lower() == "none":
        return False

    print("\nGetting info about latest updates...\n")

    try:
        if update_release_channel.lower() == "stable":
            response = requests.get(
                "https://api.github.com/repos/ThioJoe/Full-Stack-AI-Meme-Generator/releases/latest",
                timeout=30,
            )
        elif update_release_channel.lower() == "all":
            response = requests.get(
                "https://api.github.com/repos/ThioJoe/Full-Stack-AI-Meme-Generator/releases",
                timeout=30,
            )
        else:
            raise ValueError(
                f"Invalid release channel {update_release_channel!r}"
            )

        if response.status_code == 403:  # noqa: PLR2004
            print(
                "\nError [U-4]: Got an 403 (ratelimit_reached) when attempting"
                " to check for update."
            )
            print(
                "This means you have been rate limited by github.com. Please"
                " try again in a while.\n"
            )
            return False

        if response.status_code != 200:  # noqa: PLR2004
            print(
                "Error [U-3]: Got non 200 status code (got:"
                f" {response.status_code}) when attempting to check for"
                " update.\n"
            )
            print(
                "If this keeps happening, you may want to report the issue"
                " here: https://github.com/ThioJoe/Full-Stack-AI-Meme-Generator/issues"
            )
            return False

        # ??? no idea what this code block does (it probably doesn't work)
        # assume 200 response (good)
        if update_release_channel.lower() == "stable":
            latest_version = response.json()["name"]
            is_beta = False
        elif update_release_channel.lower() == "all":
            latest_version = response.json()[0]["name"]
            # check if latest version is a beta.
            # if it is continue, else check for another beta with a higher
            # version in the 10 newest releases
            is_beta = response.json()[0]["prerelease"]
            if is_beta is False:
                for i in range(9):
                    # add a "+ 1" to index to not count the first
                    # release (already checked)
                    latest_version_2 = response.json()[i + 1]["name"]
                    # make sure the version is higher than the current version
                    if parse_version(latest_version_2) > parse_version(
                        latest_version
                    ):
                        # update original latest version to the new version
                        latest_version = latest_version_2
                        is_beta = response.json()[i + 1]["prerelease"]
                        # exit loop
                        break

    except OSError as error:
        if "WinError 10013" in str(error):
            print(
                "WinError 10013: The OS blocked the connection to GitHub."
                " Check your firewall settings.\n"
            )
        else:
            traceback.print_exc()
            print(
                "Unknown OSError Error occurred while checking for updates\n"
            )
        return False
    except Exception:
        traceback.print_exc()
        print(
            "Error [Code U-1]: Problem while checking for updates. See above"
            " error for more details.\n"
        )
        print(
            "If this keeps happening, you may want to report the issue here: https://github.com/ThioJoe/Full-Stack-AI-Meme-Generator/issues"
        )
        return False

    if parse_version(latest_version) > parse_version(__version__):
        print(
            "------------------------ UPDATE AVAILABLE -----------------------"
        )
        if is_beta:
            print(
                " A new beta version is available! To see what's new visit:"
                " https://github.com/ThioJoe/Full-Stack-AI-Meme-Generator"
                "/releases "
            )
        else:
            print(
                " A new version is available! To see what's new visit:"
                " https://github.com/ThioJoe/Full-Stack-AI-Meme-Generator"
                "/releases "
            )
        print(f"     > Current Version: {__version__}")
        print(f"     > Latest Version: {latest_version}")
        if is_beta:
            print(
                "(To stop receiving beta releases, change the"
                " 'release_channel' setting in the config file)"
            )
        print(
            "------------------------------------------------------------------------------------------"
        )

        return "beta" if is_beta else True

    print(f"\nYou have the latest version: {__version__}")
    return False


def parse_meme(message: str) -> MemeDict | None:
    """
    Gets the meme text and image prompt from the message sent by the chat
    bot
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

        return MemeDict(meme_text=meme_text, image_prompt=match.group(4))
    return None


# Sends the user message to the chat bot and returns the chat bot's response
def send_and_receive_message(
    text_model: str,
    user_message: str,
    conversation_temp: list[dict[str, str]],
    temperature: float = 0.5,
) -> str:
    # Prepare to send request along with context by appending user message to
    # previous conversation
    conversation_temp.append({"role": "user", "content": user_message})

    print("Sending request to write meme...")
    try:
        chat_response = openai.ChatCompletion.create(
            model=text_model,
            messages=conversation_temp,
            temperature=temperature,
        )
    except RateLimitError:
        traceback.print_exc()
        print("\nDid you setup payment? See <https://openai.com/pricing>")
        # We don't re-raise the exception, because we want the hint^ to be
        # below the traceback
        sys.exit(1)

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
    print("Creating meme image...")

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
    api_keys: ApiKeys,
    image_prompt: str,
    platform: str,
    stability_api: client.StabilityInference | None = None,
) -> io.BytesIO:
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
        if not stability_api:
            raise ValueError(
                "Could not initialize the Stability API! Is the API key missing?"
            )
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
        raise ValueError(f"Invalid platform {platform!r}")

    try:
        return virtual_image_file
    except NameError as error:
        raise Exception(
            "Could not generate image due to above error"
        ) from error


# ==================== RUN ====================


# Set default values for parameters to those at top of script, but can be
# overridden by command line arguments or by being set when called from another
# script
def generate(
    text_model: str = "gpt-4",
    temperature: float = 1.0,
    basic_instructions: str = "You will create funny memes that are clever and"
    " original, and not cliche or lame.",
    image_special_instructions: str = "The images should be photographic.",
    user_entered_prompt: str = "anything",
    meme_count: int = 1,
    image_platform: str = "openai",
    font_file: str = "arial.ttf",
    base_file_name: str = "meme",
    output_folder: pathlib.Path = pathlib.Path("Outputs"),
    openai_key: str | None = None,
    stability_key: str | None = None,
    clipdrop_key: str | None = None,
    no_user_input: bool = False,
    no_file_save: bool = False,
    release_channel: str = "all",
) -> list[dict[str, str | pathlib.Path]]:
    # Load default settings from settings.toml file. Will be overridden by
    # command line arguments, or ignored if Use_This_Config is set to False
    settings = get_settings()
    use_config = settings.get(
        "use_this_config", False
    )  # If set to False, will ignore the settings.toml file
    if use_config:
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
        font_file = settings.get("advanced", {}).get("font_file", font_file)
        base_file_name = settings.get("advanced", {}).get(
            "base_file_name", base_file_name
        )
        output_folder = settings.get("advanced", {}).get(
            "output_folder", output_folder
        )
        release_channel = settings.get("advanced", {}).get(
            "release_channel", release_channel
        )

    # Parse the arguments
    args = parser.parse_args()

    # If API Keys not provided as parameters, get them from config file or
    # command line arguments
    if openai_key:
        api_keys = ApiKeys(openai_key, clipdrop_key, stability_key)
    else:
        api_keys = get_api_keys(args=args)

    # Validate api keys
    validate_api_keys(api_keys, image_platform)
    # Initialize api clients. Only get stability_api object back because
    # openai.api_key has global scope
    stability_api = initialize_api_clients(api_keys, image_platform)

    # Check if any settings arguments, and replace the default values with the
    # args if so. To run automated from command line, specify at least 1
    # argument.
    if args.imageplatform:
        image_platform = args.imageplatform
    if args.temperature:
        temperature = float(args.temperature)
    if args.basicinstructions:
        basic_instructions = args.basicinstructions
    if args.imagespecialinstructions:
        image_special_instructions = args.imagespecialinstructions
    if args.nofilesave:
        no_file_save = True
    if args.nouserinput:
        no_user_input = True

    system_prompt = construct_system_prompt(
        basic_instructions, image_special_instructions
    )
    conversation = [{"role": "system", "content": system_prompt}]

    # Get full path of font file from font file name
    font_file = check_font(font_file)

    if (
        (not no_user_input)
        and (
            release_channel.lower() == "all"
            or release_channel.lower() == "stable"
        )
        and check_for_update(release_channel)
    ):
        input("\nPress Enter to continue...")
    # Clear console
    os.system("cls" if os.name == "nt" else "clear")  # noqa: S605

    # ---------- Start User Input -----------
    # Display Header
    print(
        f"\n=============== AI Meme Generator - {__version__} ==============="
    )

    if not no_user_input:
        # If no user prompt argument set, get user input for prompt
        if args.userprompt:
            user_entered_prompt = args.userprompt
        else:
            print(
                "\nEnter a meme subject or concept (Or just hit enter to let"
                " the AI decide)"
            )
            user_entered_prompt = input(" >  ")
            if (
                not user_entered_prompt
            ):  # If user puts in nothing, set to "anything"
                user_entered_prompt = "anything"

        # If no meme count argument set, get user input for meme count
        if args.memecount:
            meme_count = int(args.memecount)
        else:
            # Set the number of memes to create
            meme_count = 1  # Default will be none if nothing entered
            print(
                "\nEnter the number of memes to create (Or just hit Enter for"
                " 1): "
            )
            user_entered_count = input(" >  ")
            if user_entered_count:
                meme_count = int(user_entered_count)

    def single_meme_generation_loop() -> (
        dict[str, str | pathlib.Path | io.BytesIO] | None
    ):
        # Send request to chat bot to generate meme text and image prompt
        chat_response = send_and_receive_message(
            text_model, user_entered_prompt, conversation, temperature
        )

        # Take chat message and convert to dictionary with meme_text and
        # image_prompt
        meme_dict = parse_meme(chat_response)
        if meme_dict is None:
            print("Could not interpret response! Skipping")
            return None
        image_prompt = meme_dict["image_prompt"]
        meme_text = meme_dict["meme_text"]

        # Print the meme text and image prompt
        print("\n   Meme Text:  " + meme_text)
        print("   Image Prompt:  " + image_prompt)

        # Send image prompt to image generator and get image back
        # (Using DALL·E API)
        print("\nSending image creation request...")
        virtual_image_file = image_generation_request(
            api_keys, image_prompt, image_platform, stability_api
        )

        # Combine the meme text and image into a meme
        file = set_file_path(base_file_name, output_folder)
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
                meme_dict,
                file,
                output_folder,
                basic_instructions,
                image_special_instructions,
                image_platform,
            )

        return {
            "meme_text": meme_text,
            "image_prompt": image_prompt,
            "virtual_meme_file": virtual_meme_file,
            "file": file,
        }

    # Create list of dictionaries to hold the results of each meme so that they
    # can be returned by main() if called from command line
    meme_results_dicts_list = []

    for number in range(1, meme_count + 1):
        print(
            "\n----------------------------------------------------------------------------------------------------"
        )
        print(f"Generating meme {number} of {meme_count}...")
        meme_info_dict = single_meme_generation_loop()

        # Add meme info dict to list of meme results
        if meme_info_dict:
            meme_results_dicts_list.append(meme_info_dict)

    # If called from command line, will return the list of meme results
    return meme_results_dicts_list


if __name__ == "__main__":
    try:
        generate()
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to exit...")
