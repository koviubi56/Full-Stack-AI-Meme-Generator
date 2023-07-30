# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 **! Added GPT4All as a text generation service.**
- 7cc72522d7f8c895cc5a2ce9cef2810e8c56fb84 **Catch errors and ask for Abort/retry/skip**
- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 Added new config key: `ai_settings.text_generation_service`
- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 Added new command line argument: `--text-generation-service`
- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 Added new classes: `InvalidTextPlatformError`, `GPT4AllText`
- 34f6313537f506349aee7cba6a9c55ee0542eb1e Added new classes: `TextABC`, `OpenAIText`, `ImageABC`, `OpenAIImage`, `StabilityImage`, `ClipdropImage`
- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 Added argument for `generate`: `text_generation_service`
- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 Added new function: `text_generation_request`
- 802bdec67a8c91e49aa1cbdb8b2f04a0842e15ec Added new classes: `APIKeys`, `Meme`, `FullMeme`
- 0b79a1b0a1c3bdc45551a74b5dd08fe160586ed7 Added new exceptions: `MemeGeneratorError`, `NoFontFileError`, `MissingAPIKeyError`, `InvalidImagePlatformError`
- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 Added new method to `TextABC`: `create_initialize_and_generate`
- 678605f3eeac8c92b6d7b6acfb19515f72ffdbd7 Added new dependency: `gpt4all`
- b46601a18148833ebad766a5e9bef727de3b29aa Added colors and new log messages
- e7e81eee2e4b98deb61f7f884dfe71add1385375 Added a new `MissingAPIKeyError.api` value: `"ALL"`

### Changed

- 513feadd17be71b25559d61cc531471eade60451 **Default text model is now `"gpt-3.5-turbo"`**
- 802bdec67a8c91e49aa1cbdb8b2f04a0842e15ec **Changed the command line arguments to `--openai-key`, `--clipdrop-key`, `--stability-key`, `--user-prompt`, `--meme-count`, `--image-platform`, `--basic-instructions`, `--image-special-instructions`, `--no-user-input`, `--no-file-save`**
- 14ff56fcc63938ebb4c8d65002074d50f62a2822 Removed update mechanism
- 7e8ddc537d65fc2bdc02f76fb482edae9734a576 Minimum supported python version is 3.8
- e7e81eee2e4b98deb61f7f884dfe71add1385375 Made creating default settings and api keys files optional
- 34f6313537f506349aee7cba6a9c55ee0542eb1e Moved text and image generators into their own classes

### Deprecated

### Removed

- 34f6313537f506349aee7cba6a9c55ee0542eb1e Removed validate_api_keys()
- 34f6313537f506349aee7cba6a9c55ee0542eb1e Removed send_and_receive_message()
- 0b79a1b0a1c3bdc45551a74b5dd08fe160586ed7 Removed image_generation_request()'s `stability_api` parameter
- 0b79a1b0a1c3bdc45551a74b5dd08fe160586ed7 Removed check_font()'s `no_user_input` parameter
- c4ecdf5db4dbed8eaba51eb949b1d38145b3fc9f Removed leftover global variables

### Fixed

- e7e81eee2e4b98deb61f7f884dfe71add1385375 Setting `use_this_config` explicitly to true is now not necessary
- e7e81eee2e4b98deb61f7f884dfe71add1385375 In `no_user_input` memes are now skipped when an error is raised
- e7e81eee2e4b98deb61f7f884dfe71add1385375 Fixed the order of configuration getting (it's now `config value > command line argument > function argument`)

### Security

## [1.0.2] - 2023-07-19

_(This version was "inherited" from ThioJoe's original version of Full-Stack-AI-Meme-Generator)_

### Added

- Added clarification about OpenAI option using DALLE2 in settings file

### Fixed

- Fixed incorrect passing of variables and API key validation because of leftover global variables, which caused issues when using Stability AI as image platform (#5)

## [1.0.1] - 2023-07-19

_(This version was "inherited" from ThioJoe's original version of Full-Stack-AI-Meme-Generator)_
