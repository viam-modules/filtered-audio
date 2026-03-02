import os
import ssl
import urllib.request

import certifi


def download_file(url: str, dest_path: str, logger=None) -> str:
    """
    Download a file from a URL with SSL verification and cleanup on failure.

    Args:
        url: URL to download from
        dest_path: Local path to save the file
        logger: Optional logger instance

    Returns:
        Path to the downloaded file
    """
    if logger:
        logger.info(f"Downloading {url}...")

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(url, context=ssl_context, timeout=120) as resp:
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

    return dest_path
