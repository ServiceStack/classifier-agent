import os
import sys
import numpy as np

from collections import defaultdict, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile
from pydub import AudioSegment

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
        print(f"Processing audio file ({audio_duration_ms/1000:.1f} seconds)...")
        print(f"Segment duration: {segment_duration_ms}ms")

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
            classification_results = classifier.classify(audio_clip)

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
            print(f"Warning: Failed to classify segment at {start_ms}ms: {e}")
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
        classifier, wav_data, sample_rate, segment_duration_ms=975
    )
    max_count = max(category_counts.values())
    max_count = int(max_count * 1.1)
    sorted_by_count = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    tags = {}
    for i, (category, count) in enumerate(sorted_by_count[:20]):
        tags[category] = round(count/max_count, 5) # round to 5 decimals
    return tags
