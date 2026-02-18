import os

# Suppress Paddle/PaddleOCR initialization output before any imports
os.environ.setdefault("GLOG_minloglevel", "3")  # Suppress glog (C++ logging)
os.environ.setdefault("FLAGS_minloglevel", "3")  # Suppress Paddle flags logging
os.environ.setdefault("PADDLEOCR_QUIET", "1")  # PaddleOCR quiet mode

from .api import get_subtitles, save_subtitles_to_file
