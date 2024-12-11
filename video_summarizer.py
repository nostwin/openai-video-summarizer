import logging
import sys
import os
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import math

# Global variables

AUDIO_MODEL = "whisper-1"
CHAT_MODEL = "gpt-4o-mini"
LANGUAGE = "en"
INPUT_PATH = "input"
SYSTEM_PROMPT = "You are an expert at summarizing transcripts. The summary output must be always in markdown."
USER_PROMPT = (
    "Summarize the following video transcript."
)


class VideoSummarizer:
    def __init__(self, openai: OpenAI, audio_model: str, chat_model: str,
                 language: str, system_prompt: str, user_prompt: str, input_path: str):
        self.openai = openai
        self.base_path = Path(__file__).resolve().parent
        self.input_path = Path(input_path)
        self.output_path = self.base_path / 'output'
        self.summary_path = self.base_path / 'summary'
        self.audio_model = audio_model
        self.chat_model = chat_model
        self.language = language
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        # Initialize paths
        self.audio_path = None
        self.transcript_path = None
        self.video_path = None

        # Set up directories
        self._setup_directories()

    def _setup_directories(self):
        """Ensure necessary directories exist."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.summary_path.mkdir(parents=True, exist_ok=True)

    def get_videos(self) -> List[Path]:
        """Get video files from input path."""
        try:
            logger.info(f'Getting videos from {self.input_path}')
            mp4_files = list(self.input_path.glob("*.mp4"))
            if not mp4_files:
                logger.warning("No mp4 files found.")
                exit(0)
            return mp4_files
        except Exception as e:
            logger.error(f"Error getting videos: {repr(e)}")
            return []

    def create_video_directories(self, video_name: str):
        """Create directories for each video."""
        self.video_path = self.output_path / video_name
        self.audio_path = self.video_path / 'audio'
        self.transcript_path = self.video_path / 'transcript'

        self.video_path.mkdir(parents=True, exist_ok=True)
        self.audio_path.mkdir(parents=True, exist_ok=True)
        self.transcript_path.mkdir(parents=True, exist_ok=True)

    def mp4_to_mp3(self, mp4_file_path: Path, video_name: str) -> Path:
        """Convert MP4 video to MP3 audio."""
        try:
            logger.info(f'Converting mp4 to mp3: {mp4_file_path}')
            audio_file = self.audio_path / f"{video_name}.mp3"
            if audio_file.exists():
                logger.info("SKIP - Audio file already exists. Skipping conversion.")
                return audio_file
            video = VideoFileClip(str(mp4_file_path))
            video.audio.write_audiofile(str(audio_file))
            video.close()
            return audio_file
        except Exception as e:
            logger.error(f"Error converting MP4 to MP3: {repr(e)}")
            return Path()

    def split_audio(self, input_file: Path, chunk_duration_min: int = 20) -> List[Path]:
        """Split audio into chunks."""
        try:
            logger.info("Splitting audio into chunks.")

            existing_chunks = sorted(self.audio_path.glob("chunk_*.mp3"))
            if existing_chunks:
                logger.info("SKIP - Chunks already exist, skipping split.")
                return existing_chunks

            audio = AudioSegment.from_mp3(input_file)
            chunk_duration_ms = chunk_duration_min * 60 * 1000  # convert to milliseconds
            num_chunks = math.ceil(len(audio) / chunk_duration_ms)
            chunk_paths = []

            for i in range(num_chunks):
                start_time = i * chunk_duration_ms
                end_time = min((i + 1) * chunk_duration_ms, len(audio))
                chunk = audio[start_time:end_time]
                chunk_path = self.audio_path / f"chunk_{i}.mp3"
                chunk.export(chunk_path, format="mp3")
                logger.info(f"Chunk {i} done.")
                chunk_paths.append(chunk_path)

            return chunk_paths
        except Exception as e:
            logger.error(f"Error splitting audio: {repr(e)}")
            return []

    def transcribe(self, chunk_paths: List[Path]) -> str:
        """Transcribe audio chunks to text."""
        try:
            logger.info(f"Transcribing audio chunks: {len(chunk_paths)}")

            transcription_file = self.transcript_path / "transcription.txt"

            if transcription_file.exists():
                logger.info("SKIP - Transcription file already exists, skipping transcription.")
                with open(transcription_file, "r", encoding='utf-8') as r:
                    return r.read()

            for chunk_path in chunk_paths:
                transcription = self.openai.audio.transcriptions.create(
                    model=self.audio_model,
                    file=chunk_path,
                    language=self.language
                )

                with open(transcription_file, "a", encoding='utf-8') as a:
                    a.write(f"{transcription.text}\n")

            with open(transcription_file, "r", encoding='utf-8') as r:
                return r.read()
        except Exception as e:
            logger.error(f"Error transcribing audio: {repr(e)}")
            return ""

    def summarize(self, content: str) -> str:
        """Summarize the transcribed text using GPT model."""
        try:
            logger.info("Summarizing transcription.")
            response = self.openai.chat.completions.create(
                model=self.chat_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt + f"\nTranscripcion:\n{content}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error summarizing transcription: {repr(e)}")
            return ""

    def save_summary(self, summary: str, video_name: str):
        """Save the summary to a markdown file."""
        try:
            summary_file = f"{video_name}_summary.md"
            with open(self.video_path / summary_file, "w", encoding='utf-8') as f:
                f.write(summary)
            with open(self.summary_path / summary_file, "w", encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"SUCCESS - Summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving summary: {repr(e)}")

    def process_video(self, video: Path) -> str:
        """Process a single video through the pipeline."""
        video_name = video.stem
        self.create_video_directories(video_name)

        audio_file = self.mp4_to_mp3(video, video_name)
        if not audio_file.exists():
            return "No audio file found."

        chunk_paths = self.split_audio(audio_file)
        if not chunk_paths:
            return "No audio chunks found."

        transcription = self.transcribe(chunk_paths)
        if not transcription:
            return "No transcription found."

        summary = self.summarize(transcription)
        if not summary:
            return "No summary found."

        self.save_summary(summary, video_name)
        return "OK"


def setup_logging():
    """Set up logging configuration."""
    global logger

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    video_summarizer = VideoSummarizer(
        openai=openai,
        audio_model=AUDIO_MODEL,
        chat_model=CHAT_MODEL,
        language=LANGUAGE,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        input_path=INPUT_PATH
    )

    videos = video_summarizer.get_videos()
    for video in videos:
        video_summarizer.process_video(video)


if __name__ == "__main__":
    setup_logging()
    main()
