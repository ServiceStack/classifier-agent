#!/usr/bin/env python

import os
import io
import sys
import time
import numpy as np
import requests
import json
import asyncio
import concurrent.futures


from collections import defaultdict, Counter

from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile
from pydub import AudioSegment

def classify_with_timeout(classifier, audio_clip, timeout_seconds=30):
    """Classify audio with a timeout mechanism using thread-based approach."""
    def _classify():
        return classifier.classify(audio_clip)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_classify)
        try:
            # Wait for the result with timeout
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Warning: Classification timed out after {timeout_seconds}s", flush=True)
            # Cancel the future (though it may not stop the underlying operation)
            future.cancel()
            return None
        except Exception as e:
            print(f"Warning: Classification failed: {e}", flush=True)
            return None


def classify_with_process_timeout(classifier, audio_clip, timeout_seconds=30):
    """Alternative timeout approach using multiprocessing (more aggressive)."""
    import multiprocessing
    import os

    def _classify_in_process(model_path, audio_data, sample_rate, result_queue):
        """Function to run classification in a separate process."""
        try:
            # Recreate classifier in the new process
            from mediapipe.tasks import python
            from mediapipe.tasks.python import audio
            from mediapipe.tasks.python.components import containers
            import numpy as np

            base_options = python.BaseOptions(model_asset_path=model_path)
            audio_options = audio.AudioClassifierOptions(
                base_options=base_options, max_results=10)
            new_classifier = audio.AudioClassifier.create_from_options(audio_options)

            # Recreate audio clip
            new_audio_clip = containers.AudioData.create_from_array(audio_data, sample_rate)

            # Perform classification
            result = new_classifier.classify(new_audio_clip)
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', str(e)))

    # Get model path and audio data for serialization
    try:
        # Extract model path from classifier (this is a bit hacky but necessary)
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models")
        classifiers_path = os.path.join(models_dir, 'classifiers')
        model_path = os.path.join(classifiers_path, 'classifier.tflite')

        # Extract audio data
        audio_data = audio_clip.buffer
        sample_rate = audio_clip.audio_format.sample_rate
    except Exception as e:
        raise Exception(f"Failed to extract classifier/audio data for multiprocessing: {e}")

    # Create a queue for results
    result_queue = multiprocessing.Queue()

    # Start the classification process
    process = multiprocessing.Process(
        target=_classify_in_process,
        args=(model_path, audio_data, sample_rate, result_queue)
    )
    process.start()

    # Wait for the process to complete or timeout
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        print(f"Warning: Classification process timed out after {timeout_seconds}s, terminating...", flush=True)
        process.terminate()
        process.join(timeout=5)  # Give it 5 seconds to terminate gracefully
        if process.is_alive():
            process.kill()  # Force kill if still alive
        raise TimeoutError(f"Classification timed out after {timeout_seconds} seconds")

    # Get the result if available
    try:
        if not result_queue.empty():
            status, result = result_queue.get_nowait()
            if status == 'success':
                return result
            else:
                raise Exception(f"Classification failed in process: {result}")
    except Exception as e:
        if "Classification failed in process" in str(e):
            raise e
        pass

    raise Exception("Classification process completed but no result was returned")


def classify_with_simple_timeout(classifier, audio_clip, timeout_seconds=10):
    """Simple timeout approach - just skip if it takes too long."""
    import threading
    import time

    result = [None]
    exception = [None]

    def _classify():
        try:
            result[0] = classifier.classify(audio_clip)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=_classify)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running, consider it timed out
        raise TimeoutError(f"Classification timed out after {timeout_seconds} seconds")

    if exception[0]:
        raise exception[0]

    if result[0] is None:
        raise Exception("Classification completed but returned no result")

    return result[0]


def convert_to_wav_data(audio_path, format):
    """Convert M4A AAC audio file to WAV format data for MediaPipe."""
    # Load the M4A file using pydub
    audio_segment = AudioSegment.from_file(audio_path, format=format)

    # Convert to mono if stereo
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)

    # Set sample rate to 16kHz (common for audio classification)
    audio_segment = audio_segment.set_frame_rate(16000)

    # Convert to 16-bit PCM
    audio_segment = audio_segment.set_sample_width(2)

    # Get raw audio data as numpy array
    raw_data = audio_segment.raw_data
    wav_data = np.frombuffer(raw_data, dtype=np.int16)

    return audio_segment.frame_rate, wav_data

def load_audio_file(audio_path):
    """Load audio file, converting M4A to WAV format if needed."""
    file_ext = os.path.splitext(audio_path)[1].lower()

    if file_ext != '.wav':
        format = file_ext[1:]
        return convert_to_wav_data(audio_path, format=format)
    else:
        # Use scipy for WAV files
        return wavfile.read(audio_path)

def get_audio_duration(sample_rate, wav_data):
    """Calculate audio duration in milliseconds."""
    return len(wav_data) * 1000 // sample_rate


def process_audio_segments(classifier, wav_data, sample_rate, segment_duration_ms=975, debug=False):
    """Process audio in segments and collect all classifications."""
    audio_duration_ms = get_audio_duration(sample_rate, wav_data)
    segment_size = int(sample_rate * segment_duration_ms / 1000)

    all_classifications = []
    category_scores = defaultdict(list)
    category_counts = Counter()

    if debug:
        print(f"[classifier-agent] Processing audio file ({audio_duration_ms/1000:.1f} seconds)...", flush=True)
        print(f"[classifier-agent] Segment duration: {segment_duration_ms}ms", flush=True)

    # Process audio in overlapping segments
    for start_ms in range(0, audio_duration_ms, segment_duration_ms):
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = min(start_sample + segment_size, len(wav_data))

        # Skip if segment is too short
        if end_sample - start_sample < segment_size // 2:
            break

        segment_data = wav_data[start_sample:end_sample]

        # Convert to float and normalize
        audio_clip = containers.AudioData.create_from_array(
            segment_data.astype(float) / np.iinfo(np.int16).max, sample_rate)

        try:
            started_at = time.time()
            if debug:
                print("[classifier-agent] Classifying segment...", flush=True)
            classification_results = classify_with_process_timeout(classifier, audio_clip, timeout_seconds=10)

            if classification_results is None:
                # Skip this segment if classification failed or timed out
                continue

            if debug:
                print(f"[classifier-agent] Classified in {time.time() - started_at:.2f}s", flush=True)

            # Process each classification result (MediaPipe may return multiple results per segment)
            for classification_result in classification_results:
                timestamp = start_ms
                classifications = classification_result.classifications[0].categories

                # Store all categories for this timestamp
                segment_info = {
                    'timestamp': timestamp,
                    'categories': []
                }

                for category in classifications:
                    category_name = category.category_name
                    score = category.score

                    segment_info['categories'].append({
                        'name': category_name,
                        'score': score
                    })

                    # Accumulate scores and counts
                    category_scores[category_name].append(score)
                    category_counts[category_name] += 1

                all_classifications.append(segment_info)

        except Exception as e:
            print(f"Warning: Failed to classify segment at {start_ms}ms: {e}", flush=True)
            continue

    return all_classifications, category_scores, category_counts

def load_audio_model(models_dir):
    classifiers_path = os.path.join(models_dir, 'classifiers')
    base_options = python.BaseOptions(model_asset_path=os.path.join(classifiers_path,'classifier.tflite'))
    audio_options = audio.AudioClassifierOptions(
        base_options=base_options, max_results=10)  # Increased to get more categories
    classifier = audio.AudioClassifier.create_from_options(audio_options)
    return classifier

def get_audio_tags(classifier, audio_path, debug=None):
    # Load audio file (supports both WAV and M4A formats)
    sample_rate, wav_data = load_audio_file(audio_path)
    return get_audio_tags_from_wav(classifier, sample_rate, wav_data, debug=debug)

def get_audio_tags_from_wav(classifier, sample_rate, wav_data, debug=None):
    # Process the entire audio file in segments
    all_classifications, category_scores, category_counts = process_audio_segments(
        classifier, wav_data, sample_rate, segment_duration_ms=975, debug=debug
    )
    max_count = max(category_counts.values())
    max_count = int(max_count * 1.1)
    sorted_by_count = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    tags = {}
    for i, (category, count) in enumerate(sorted_by_count[:20]):
        tags[category] = round(count/max_count, 5) # round to 5 decimals
    return tags

def download_and_convert_audio(audio_url):
    """Download audio from URL and convert to WAV format."""
    response = requests.get(audio_url, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    format = audio_url.split('.')[-1]
    sample_rate, wav_data = convert_to_wav_data(io.BytesIO(response.content), format=format)  # noqa: F821
    return sample_rate, wav_data

async def main():
    audio_url = sys.argv[1]

    # Run model loading and audio download+conversion concurrently
    loop = asyncio.get_event_loop()

    models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models")

    # Create tasks for concurrent execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit both I/O operations to run concurrently
        model_future = loop.run_in_executor(executor, load_audio_model, models_dir)
        audio_future = loop.run_in_executor(executor, download_and_convert_audio, audio_url)

        # Wait for both operations to complete
        classifier, (sample_rate, wav_data) = await asyncio.gather(model_future, audio_future)

    tags = get_audio_tags_from_wav(classifier, sample_rate, wav_data, debug=True)
    json.dump(tags, sys.stdout, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
