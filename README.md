# Full Stack AI Meme Generator - Not maintained

Allows you to automatically generate meme images from start to finish using AI. It will generate the text for the meme (optionally based on a user-provided concept), create a related image, and combine the two into a final image file.

!["When you've been up all night coding and your AI meme generator only creates dad jokes." An image of a man sitting in front of a computer.](https://github.com/ThioJoe/Full-Stack-AI-Meme-Generator/assets/12518330/2d8ee7cc-a7d3-40ca-a894-64e10085db14)

## Features

- Uses OpenAI's GPT-4 to generate the text and image prompt for the meme.
- Automatically sends image prompt request to an AI image generator of choice, then combines the text and image
- Allows customization of the meme generation process through various settings.
- Generates memes with a user-provided subject or concept, or you can let the AI decide.

## Usage

1. For Python Version Only: Clone the repository & Install the necessary packages.
2. Obtain at least an OpenAI API key, but it is recommended to also use APIs from Clipdrop or Stability AI (DreamStudio) for the image generation stage.
3. Edit the settings variables in the settings.toml file.
4. Run the script and enter a meme subject or concept when prompted (optional).

## Settings

Various settings for the meme generation process can be customized:

- OpenAI API settings: Choose the text model and temperature for generating the meme text and image prompt.
- Image platform settings: Choose the platform for generating the meme image. Options include OpenAI's DALLE2, StabilityAI's DreamStudio, and ClipDrop.
- Basic Meme Instructions: You can tell the AI about the general style or qualities to apply to all memes, such as using dark humor, surreal humor, wholesome, etc.
- Special Image Instructions: You can tell the AI how to generate the image itself (more specifically,  how to write the image prompt). You can specify a style such as being a photograph, drawing, etc, or something more specific such as always using cats in the pictures.

## Optional Arguments

You can also pass options into the program via command-line arguments whether using the python version or exe version.

### API Key Arguments: Not necessary if the keys are already in api_keys.toml

- `--openai-key`: OpenAI API key.

- `--clip-dropkey`: ClipDrop API key.

- `--stability-key`: Stability AI API key.

### Basic Meme Arguments

- `--user-prompt`: A meme subject or concept to send to the chat bot. If not specified, the user will be prompted to enter a subject or concept.

- `--meme-count`: The number of memes to create. If using arguments and not specified, the default is 1.

### Advanced Meme Settings Arguments

- `--image-platform`: The image platform to use. If using arguments and not specified, the default is 'clipdrop'. Possible options: 'openai', 'stability', 'clipdrop'.

- `--temperature`: The temperature to use for the chat bot. If using arguments and not specified, the default is 1.0.

- `--basic-instructions`: The basic instructions to use for the chat bot. If using arguments and not specified, the default is "You will create funny memes that are clever and original, and not cliche or lame.".

- `--image-special-instructions`: The image special instructions to use for the chat bot. The default is "The images should be photographic.".

### Binary arguments: Just adding them activates them, no text needs to accompany them

- `--no-user-input`: If specified, this will prevent any user input prompts, and will instead use default values or other arguments.

- `--no-file-save`: If specified, the meme will not be saved to a file, and only returned as virtual file.
