import os


def extract_dropped_file_paths(mime_data):
    """Local file paths from a drag and drop, skipping non-local URLs and directories."""
    paths = [url.toLocalFile() for url in mime_data.urls() if url.isLocalFile()]
    return [path for path in paths if path and not os.path.isdir(path)]
