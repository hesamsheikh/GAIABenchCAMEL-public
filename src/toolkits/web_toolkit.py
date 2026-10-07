"""Enhanced Wikipedia toolkit for GAIA benchmark."""

import asyncio
import logging
from typing import List
from camel.toolkits import FunctionTool, Crawl4AIToolkit

logger = logging.getLogger(__name__)


class SyncCrawl4AIToolkit:
    """Synchronous wrapper for Crawl4AIToolkit with retry logic.

    The original Crawl4AIToolkit.scrape is async, which causes
    'cannot pickle coroutine' errors when used in sync contexts.
    This wrapper runs the async method in a new event loop and adds
    retry logic for resilience.
    """

    def __init__(
        self,
        timeout: int = 60,
        download_dir: str = "execution/downloads",
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        self._toolkit = Crawl4AIToolkit(timeout=timeout)
        self._download_dir = download_dir
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def scrape_url(self, url: str) -> str:
        """Scrape a webpage and return its content.

        Use this tool ONLY when you need to access a specific URL that is
        NOT Wikipedia. For Wikipedia content, prefer wikipedia_full_page
        or wikipedia_search_content tools instead.

        For general web searches, prefer search_tavily first.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            str: The scraped content of the webpage as text.
        """
        import time

        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                # Run the async scrape in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self._toolkit.scrape(url))
                    # Check if result is empty or error-like
                    if result and len(result.strip()) > 50:
                        return result
                    elif attempt < self._max_retries:
                        logger.warning(
                            f"Empty/short result from {url}, retrying ({attempt + 1}/{self._max_retries})..."
                        )
                        time.sleep(self._retry_delay)
                        continue
                    return result if result else f"Error scraping {url}: Empty response"
                finally:
                    loop.close()
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning(
                        f"Error scraping {url}: {e}, retrying ({attempt + 1}/{self._max_retries})..."
                    )
                    time.sleep(self._retry_delay)
                else:
                    logger.error(f"Error scraping {url} after {self._max_retries + 1} attempts: {e}")

        return f"Error scraping {url}: {last_error}"

    def download_file(self, url: str, filename: str = "") -> str:
        """Download a file from a URL to read locally.

        Use this for PDFs, documents, or files that scrape_url can't handle.
        After downloading, use read_file with the returned path to extract content.

        Args:
            url (str): The URL of the file to download.
            filename (str): Optional filename. If empty, extracted from URL.

        Returns:
            str: The local file path, or an error message.
        """
        import os
        import requests
        from urllib.parse import urlparse, unquote

        try:
            os.makedirs(self._download_dir, exist_ok=True)

            if not filename:
                parsed = urlparse(url)
                filename = os.path.basename(unquote(parsed.path)) or "downloaded_file"

            filepath = os.path.join(self._download_dir, filename)

            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return os.path.abspath(filepath)

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return f"Error downloading {url}: {e}"

    def get_tools(self) -> List[FunctionTool]:
        """Get the list of tools provided by this toolkit."""
        return [FunctionTool(self.scrape_url), FunctionTool(self.download_file)]


class WebToolkit:
    """A toolkit for enhanced Wikipedia access.

    This toolkit provides tools that are better suited for the GAIA benchmark
    than the default search_wiki (which only returns 5 sentences).
    """

    def wikipedia_full_page(self, entity: str, max_chars: int = 50000) -> str:
        """Get the full content of a Wikipedia page for an entity.

        Unlike search_wiki which only returns 5 sentences, this function
        returns the full Wikipedia article content (up to max_chars).
        Use this when you need detailed information like statistics,
        dates, measurements, or specific facts that may not appear in
        a summary.

        Args:
            entity (str): The Wikipedia article title to fetch
                (e.g., "Moon", "Eliud Kipchoge", "Orbit of the Moon").
            max_chars (int): Maximum characters to return. Default 50000.

        Returns:
            str: The full article content, or an error message if not found.
        """
        try:
            import wikipedia
            page = wikipedia.page(entity, auto_suggest=True)
            content = page.content
            if len(content) > max_chars:
                return content[:max_chars] + "\n\n[Content truncated...]"
            return content
        except Exception as e:
            return f"Error fetching Wikipedia page '{entity}': {str(e)}"

    def wikipedia_search_content(
        self,
        entity: str,
        search_terms: List[str],
        context_chars: int = 300
    ) -> str:
        """Search for specific terms within a Wikipedia article.

        This is useful when you need to find specific information
        (like "perigee", "world record", specific dates, measurements, etc.)
        within a Wikipedia article. It searches the FULL article content,
        not just the summary.

        Args:
            entity (str): The Wikipedia article title (e.g., "Moon",
                "Eliud Kipchoge").
            search_terms (List[str]): Terms to search for in the article.
            context_chars (int): Characters of context around each match.
                Default 300.

        Returns:
            str: Matching excerpts from the article with context.
        """
        try:
            import wikipedia
            import re

            page = wikipedia.page(entity, auto_suggest=True)
            content = page.content

            results = []
            for term in search_terms:
                pattern = f'.{{0,{context_chars}}}{re.escape(term)}.{{0,{context_chars}}}'
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    results.append(f"\n=== Matches for '{term}' ===")
                    for i, match in enumerate(matches[:3], 1):
                        results.append(f"{i}. ...{match}...")

            if not results:
                return f"No matches found for {search_terms} in '{entity}' article."

            return "\n".join(results)

        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

    def get_tools(self) -> List[FunctionTool]:
        """Get the list of tools provided by this toolkit.

        Returns:
            List[FunctionTool]: The function tools.
        """
        return [
            FunctionTool(self.wikipedia_full_page),
            FunctionTool(self.wikipedia_search_content),
        ]
