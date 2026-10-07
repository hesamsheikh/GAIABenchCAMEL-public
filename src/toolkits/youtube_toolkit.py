"""YouTube toolkit for extracting video transcripts."""

import logging
import re
from typing import List, Optional
from camel.toolkits import FunctionTool

logger = logging.getLogger(__name__)


class YouTubeToolkit:
    """A toolkit for extracting YouTube video transcripts."""

    def get_youtube_transcript(
        self,
        url_or_video_id: str,
        languages: Optional[List[str]] = None,
    ) -> str:
        """Get the transcript/captions of a YouTube video.

        Use this tool when you need to know what is said in a YouTube video,
        or to answer questions about video content based on dialogue/narration.

        Args:
            url_or_video_id (str): YouTube URL or video ID.
                Examples:
                - "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                - "https://youtu.be/dQw4w9WgXcQ"
                - "dQw4w9WgXcQ"
            languages (List[str], optional): Preferred languages for transcript.
                Defaults to ["en", "en-US", "en-GB"].

        Returns:
            str: The video transcript, or an error message.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import (
                TranscriptsDisabled,
                NoTranscriptFound,
                VideoUnavailable,
            )
        except ImportError:
            return (
                "Error: youtube-transcript-api is not installed. "
                "Install it with: pip install youtube-transcript-api"
            )

        # Extract video ID from URL
        video_id = self._extract_video_id(url_or_video_id)
        if not video_id:
            return f"Error: Could not extract video ID from '{url_or_video_id}'"

        if languages is None:
            languages = ["en", "en-US", "en-GB"]

        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=languages)

            # Extract text from transcript
            text_parts = [snippet.text for snippet in transcript]
            return " ".join(text_parts)

        except TranscriptsDisabled:
            return f"Error: Transcripts are disabled for video {video_id}"
        except VideoUnavailable:
            return f"Error: Video {video_id} is unavailable"
        except NoTranscriptFound:
            return f"Error: No transcript found for video {video_id}"
        except Exception as e:
            logger.error(f"Error fetching transcript for {video_id}: {e}")
            return f"Error fetching transcript: {str(e)}"

    def get_youtube_video_info(self, url_or_video_id: str) -> str:
        """Get basic information about a YouTube video.

        Use this to get video title, channel, description without watching.

        Args:
            url_or_video_id (str): YouTube URL or video ID.

        Returns:
            str: Video information or error message.
        """
        import urllib.request
        import json

        video_id = self._extract_video_id(url_or_video_id)
        if not video_id:
            return f"Error: Could not extract video ID from '{url_or_video_id}'"

        try:
            # Use YouTube's oEmbed endpoint (no API key needed)
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            with urllib.request.urlopen(oembed_url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return (
                    f"Title: {data.get('title', 'Unknown')}\n"
                    f"Author: {data.get('author_name', 'Unknown')}\n"
                    f"Channel URL: {data.get('author_url', 'Unknown')}"
                )
        except Exception as e:
            logger.error(f"Error fetching video info for {video_id}: {e}")
            return f"Error fetching video info: {str(e)}"

    def _extract_video_id(self, url_or_id: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        url_or_id = url_or_id.strip()

        # Already a video ID (11 characters, alphanumeric with - and _)
        if re.match(r'^[\w-]{11}$', url_or_id):
            return url_or_id

        # Standard YouTube URL: youtube.com/watch?v=VIDEO_ID
        match = re.search(r'[?&]v=([^&]+)', url_or_id)
        if match:
            return match.group(1)

        # Short URL: youtu.be/VIDEO_ID
        match = re.search(r'youtu\.be/([^?&]+)', url_or_id)
        if match:
            return match.group(1)

        # Embed URL: youtube.com/embed/VIDEO_ID
        match = re.search(r'youtube\.com/embed/([^?&]+)', url_or_id)
        if match:
            return match.group(1)

        return None

    def get_tools(self) -> List[FunctionTool]:
        """Get the list of tools provided by this toolkit."""
        return [
            FunctionTool(self.get_youtube_transcript),
            FunctionTool(self.get_youtube_video_info),
        ]
