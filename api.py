#!/usr/bin/env python3
"""
Simple HTTP API for audio classification.
Accepts a URL and returns audio tags in JSON format.

requirements.txt:
mediapipe
pydub
flask>=2.0.0
requests>=2.25.0
"""

import os
import sys
import tempfile
import requests
import argparse
from flask import Flask, request, jsonify
from urllib.parse import urlparse
import logging

# Add the path to import from the comfy-agent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'comfy-agent'))

# Import the existing audio classifier functions
from audio_classifier import load_audio_model, get_audio_tags

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier = None
models_directory = None

def download_audio_file(url, temp_dir):
    """Download audio file from URL to temporary directory."""
    try:
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # If no filename in URL, use a default with common audio extension
        if not filename or '.' not in filename:
            filename = 'audio_file.mp3'

        # Create temporary file path
        temp_file_path = os.path.join(temp_dir, filename)

        # Download the file
        logger.info(f"Downloading audio from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Write to temporary file
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded audio to: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.error(f"Error downloading audio file: {e}")
        raise

def initialize_classifier():
    """Initialize the audio classifier model."""
    global classifier, models_directory
    if classifier is None:
        try:
            if models_directory is None:
                raise ValueError("Models directory not specified. Please provide models directory as command line argument.")

            logger.info(f"Loading audio model from: {models_directory}")
            classifier = load_audio_model(models_directory)
            logger.info("Audio classifier loaded successfully")
        except Exception as e:
            logger.error(f"Error loading audio classifier: {e}")
            raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "audio-classifier-api"})

@app.route('/audio/tags', methods=['GET', 'POST'])
def audio_tags():
    """
    Classify audio from URL.

    For GET requests, expects URL as query parameter:
    GET /audio/tags?url=https://example.com/audio.mp3

    For POST requests, expects JSON payload:
    {
        "url": "https://example.com/audio.mp3"
    }

    Returns:
    {
        "music": 0.95,
        "speech": 0.05,
        ...
    }
    """
    try:
        # Get URL from query parameter or JSON body
        if request.method == 'GET':
            url = request.args.get('url')
            if not url:
                return jsonify({"error": "Missing 'url' query parameter"}), 400
        else:  # POST request
            data = request.get_json()
            if not data or 'url' not in data:
                return jsonify({"error": "Missing 'url' in request body"}), 400
            url = data['url']

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return jsonify({"error": "Invalid URL format"}), 400

        # Initialize classifier if needed
        initialize_classifier()

        # Create temporary directory for downloaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download audio file
            audio_file_path = download_audio_file(url, temp_dir)

            # Classify audio
            logger.info(f"Classifying audio: {audio_file_path}")
            tags = get_audio_tags(classifier, audio_file_path, debug=False)

            # Return results
            logger.info(f"Classification complete. Found {len(tags)} tags")
            return jsonify(tags)

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return jsonify({"error": f"Failed to download audio: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

@app.route('/classify-batch', methods=['POST'])
def classify_audio_batch():
    """
    Classify multiple audio files from URLs.

    Expected JSON payload:
    {
        "urls": ["https://example.com/audio1.mp3", "https://example.com/audio2.mp3"]
    }

    Returns:
    {
        "results": [
            {
                "tags": {...},
                "url": "https://example.com/audio1.mp3",
                "success": true
            },
            {
                "error": "Failed to download",
                "url": "https://example.com/audio2.mp3",
                "success": false
            }
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'urls' not in data:
            return jsonify({"error": "Missing 'urls' in request body"}), 400

        urls = data['urls']
        if not isinstance(urls, list):
            return jsonify({"error": "'urls' must be a list"}), 400

        # Initialize classifier if needed
        initialize_classifier()

        results = []

        # Process each URL
        for url in urls:
            try:
                # Validate URL
                if not url.startswith(('http://', 'https://')):
                    results.append({
                        "url": url,
                        "success": False,
                        "error": "Invalid URL format"
                    })
                    continue

                # Create temporary directory for this file
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download and classify
                    audio_file_path = download_audio_file(url, temp_dir)
                    tags = get_audio_tags(classifier, audio_file_path, debug=False)

                    results.append({
                        "url": url,
                        "success": True,
                        "tags": tags
                    })

            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })

        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        return jsonify({"error": f"Batch classification failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Audio Classification API Server')
    parser.add_argument('models_dir', help='Path to the models directory')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5005, help='Port to bind to (default: 5005)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Set the global models directory
    models_directory = os.path.abspath(args.models_dir)

    # Validate models directory exists
    if not os.path.exists(models_directory):
        logger.error(f"Models directory does not exist: {models_directory}")
        sys.exit(1)

    # Initialize classifier on startup
    try:
        initialize_classifier()
        logger.info("Starting audio classification API server...")
        app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
