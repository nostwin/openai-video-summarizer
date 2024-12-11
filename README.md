# OpenAI Video Summarizer

## Overview

This project class is a tool designed to process video files through a pipeline that involves extracting audio, transcribing it, and generating a summary using OpenAI's API. It supports handling multiple videos and organizes outputs systematically in directories.

## Directory Structure
- **input**: Stores the mp4 videos that needs to be summarized.
- **output**: Stores audio, transcript, and related files for each video.
- **summary**: Stores Markdown files containing video summaries.

## Example Usage
1. Set the **.mp4** videos inside the **input** folder.

2. Set the **OPENAI_API_KEY** environment variable in .env file.

3. Set up global variables variables:
```
OPENAI_API_KEY="your_api_key from .env"
AUDIO_MODEL="whisper-1"
CHAT_MODEL="gpt-4o-mini"
LANGUAGE="en"
SYSTEM_PROMPT="You are a helpful assistant."
USER_PROMPT="Summarize the transcription."
INPUT_PATH="input"
```

4. Run the script:
```
./video_summarizer.py
```

## Notes
- Ensure all required dependencies are installed (pip install -r requirements.txt).
- Configure input_path to the directory containing .mp4 files.
- Summaries are saved as .md files.