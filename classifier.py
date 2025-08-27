import time
import torch
import clip
from PIL import Image

import supervision as sv
from ultralytics import YOLO

import numpy as np
import os

# ONNX Nudenet
import _io
import cv2
import onnxruntime

# WD14 Tagger
import pandas as pd

# Global model cache to avoid reloading models
_model_cache = {}

def get_cached_model(model_type):
    """Get a cached model or None if not cached"""
    return _model_cache.get(model_type)

def cache_model(model_type, device, model, processor_or_preprocess):
    """Cache a loaded model"""
    _model_cache[model_type] = (device, model, processor_or_preprocess)

# Define rating categories with detailed descriptions
rating_descriptions = {
    "G": "Safe for work, family friendly, G-rated content, suitable for children, wholesome, educational, no violence, no suggestive content",
    "PG": "Mildly suggestive, PG-rated content, some action, brief language, parental guidance suggested",
    "PG-13": "Teen appropriate PG-13 content some intense scenes suggestive themes brief strong language",
    "M": "Mature content, strong language, suggestive content, sexual content, violence, restricted",
    "R": "R-rated adult themes, strong language, sexual content, partial nudity, violence, NSFW",
    # NSFW detection using erax and nudenet
    # "X": "NSFW, Explicit sexual X-rated adults only content, graphic nudity",
    # "XXX": "NSFW, Extreme explicit content XXX-rated hardcore pornography, graphic violence, adult themes",
}
ratings_scale = list(rating_descriptions.keys())


def get_device():
    device = "cpu"  # Default to CPU for compatibility
    # Try CUDA first, but fall back to CPU if there are compatibility issues
    if torch.cuda.is_available():
        try:
            # Test CUDA compatibility with a simple operation
            test_tensor = torch.tensor([1.0]).cuda()
            test_result = test_tensor + 1
            device = "cuda"
        except Exception as e:
            print(
                f"CUDA available but not compatible for CLIP, falling back to CPU: {e}"
            )
            device = "cpu"
    return device

def load_clip():
    return load_vit_b32_clip()

def load_vit_b32_clip():
    cached = get_cached_model("vit_b32")
    if cached:
        return cached

    device = get_device()
    model, preprocess = clip.load("ViT-B/32", device=device)

    cache_model("vit_b32", device, model, preprocess)
    return device, model, preprocess

def classify_image_rating_from_path(image_path):
    # Load and preprocess image
    device, model, preprocess = load_clip()
    with Image.open(image_path) as image:
        return classify_image_rating(device, model, preprocess, image)

def classify_image_rating(device, model, preprocess, image):
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Tokenize text descriptions
    text_inputs = [
        clip.tokenize(f"This image contains {desc}").to(device)
        for desc in rating_descriptions.values()
    ]

    # Calculate similarities
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(image_input)

        # Get text features for each rating
        similarities = []
        for text_input in text_inputs:
            text_features = model.encode_text(text_input)

            # Calculate cosine similarity
            similarity = torch.cosine_similarity(image_features, text_features)
            similarities.append(similarity.item())

    # Find the rating with highest similarity
    ratings = list(rating_descriptions.keys())
    max_idx = np.argmax(similarities)
    predicted_rating = ratings[max_idx]
    confidence = similarities[max_idx]

    # Create results dictionary
    results = {
        "predicted_rating": predicted_rating,
        "confidence": confidence,
        "all_scores": dict(zip(ratings, similarities)),
    }

    return results

def classify_image_categories(device, model, preprocess, image, categories):
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Prepare text descriptions for categories
    text_inputs = torch.cat(
        [clip.tokenize(f"an image of {category}") for category in categories]
    ).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Get top categories
    values, indices = similarity[0].topk(3)

    # Return top 5 categories with their confidence scores
    results = {}
    for value, index in zip(values, indices):
        results[categories[index]] = value.item()

    return results

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


__labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


def _postprocess(
    output,
    x_pad,
    y_pad,
    x_ratio,
    y_ratio,
    image_original_width,
    image_original_height,
    model_width,
    model_height,
):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0:4]

            # Convert from center coordinates to top-left corner coordinates
            x = x - w / 2
            y = y - h / 2

            # Scale coordinates to original image size
            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height

            # Remove padding
            x = x
            y = y

            # Clip coordinates to image boundaries
            x = max(0, min(x, image_original_width))
            y = max(0, min(y, image_original_height))
            w = min(w, image_original_width - x)
            h = min(h, image_original_height - y)

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        x, y, w, h = box
        detections.append(
            {
                "model": "640m",
                "class": __labels[class_id],
                "score": float(score),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )

    return detections

def load_yolo_erax(models_dir):
    yolo_erax = YOLO(os.path.join(models_dir, "erax_nsfw_yolo11m.pt"))  # 40.5M
    return yolo_erax

def classify_image_objects_erax(erax_yolo, image):
    results = erax_yolo(image, conf=0.2, iou=0.3, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    image_results = []
    if len(detections) > 0:
        for detection in detections:
            xyxy = detection[0].tolist()
            # round to integers
            box = [int(round(x)) for x in xyxy]
            image_results.append(
                {
                    "model": "erax",
                    "class": detection[5]["class_name"],
                    "score": detection[2].item(),
                    "box": box,
                }
            )
    return image_results

def load_nudenet(models_dir):
    nudenet = onnxruntime.InferenceSession(
        os.path.join(models_dir, "640m.onnx"), providers=None
    )
    return nudenet

# Nudenet
def _read_image(image_path, target_size=320):
    if isinstance(image_path, str):
        mat = cv2.imread(image_path)
    elif isinstance(image_path, Image.Image):
        mat = convert_from_image_to_cv2(image_path)
    elif isinstance(image_path, np.ndarray):
        mat = image_path
    elif isinstance(image_path, bytes):
        mat = cv2.imdecode(np.frombuffer(image_path, np.uint8), -1)
    elif isinstance(image_path, _io.BufferedReader):
        mat = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), -1)
    else:
        raise ValueError(
            "please make sure the image_path is str or np.ndarray or bytes"
        )

    image_original_width, image_original_height = mat.shape[1], mat.shape[0]

    mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

    max_size = max(mat_c3.shape[:2])  # get max size from width and height
    x_pad = max_size - mat_c3.shape[1]  # set xPadding
    x_ratio = max_size / mat_c3.shape[1]  # set xRatio
    y_pad = max_size - mat_c3.shape[0]  # set yPadding
    y_ratio = max_size / mat_c3.shape[0]  # set yRatio

    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    input_blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,  # normalize
        (target_size, target_size),  # resize to model input size
        (0, 0, 0),  # mean subtraction
        swapRB=True,  # swap red and blue channels
        crop=False,  # don't crop
    )

    return (
        input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


def classify_image_objects_nudenet(nudenet, image):
    inference_resolution = 640
    model_inputs = nudenet.get_inputs()

    input_width = inference_resolution
    input_height = inference_resolution
    input_name = model_inputs[0].name
    output_names = [output.name for output in nudenet.get_outputs()]
    (
        preprocessed_image,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    ) = _read_image(image, input_width)

    outputs = nudenet.run(output_names, {input_name: preprocessed_image})

    image_results = _postprocess(
        outputs,
        x_pad,
        y_pad,
        x_ratio,
        y_ratio,
        image_original_width,
        image_original_height,
        input_width,
        input_height,
    )
    return image_results

def classify_image_objects(erax_yolo, nudenet, image):
    image_results = classify_image_objects_erax(erax_yolo, image)
    image_results += classify_image_objects_nudenet(nudenet, image)
    return image_results

def load_wd14_model(models_dir):
    model_path = os.path.join(models_dir, "wd-v1-4-moat-tagger-v2.onnx")  # 326M
    csv_path = os.path.join(models_dir, "wd-v1-4-moat-tagger-v2.csv")  # 246K

    # Load the ONNX model with specific providers and session options
    providers = ["CPUExecutionProvider"]  # Force CPU to avoid GPU issues
    session_options = onnxruntime.SessionOptions()
    session_options.enable_cpu_mem_arena = (
        False  # Disable memory arena to avoid caching issues
    )
    session_options.enable_mem_pattern = False  # Disable memory pattern optimization

    model = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers
    )

    # Load the tags CSV
    tags_df = pd.read_csv(csv_path)
    tag_names = tags_df["name"].tolist()
    return model, tag_names


def pairs_to_dict(kvpairs):
    return {k: v for k, v in kvpairs}

def classify_image_tags(model, tag_names, image, threshold=0.6, debug=False):
    # Preprocess image

    # Get input details
    input = model.get_inputs()[0]
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    height = input.shape[1]

    # Reduce to max size and pad with white
    ratio = float(height) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    resized_image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(
        resized_image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2)
    )

    processed_image = np.array(square).astype(np.float32)
    processed_image = processed_image[:, :, ::-1]  # RGB -> BGR
    processed_image = np.expand_dims(processed_image, 0)

    if processed_image is None:
        return {}

    try:
        outputs = model.run([output_name], {input_name: processed_image})
        predictions = outputs[0][0]  # Remove batch dimension

        # Filter tags by threshold
        predicted_tags = []
        for i, score in enumerate(predictions):
            if score > threshold:
                tag_name = tag_names[i]
                predicted_tags.append((tag_name, float(score)))

        # Sort by confidence score (descending)
        predicted_tags.sort(key=lambda x: x[1], reverse=True)

        tags_dict = pairs_to_dict(predicted_tags)
        return tags_dict

    except Exception as e:
        print(f"Error predicting tags: {e}")
        return {}


def classify_images_tags(model, tag_names, image_paths, threshold=0.6, debug=False):
    ret = {}
    for image_path in image_paths:
        # Open and convert image
        with Image.open(image_path) as image:
            ret[image_path] = classify_image_tags(
                model, tag_names, image, threshold=threshold, debug=debug
            )

    return ret


def load_image_models(models_dir, debug=False):
    # Load the clip ratings and categories model
    start_time = time.time()
    device = get_device()
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    if debug:
        print(f"Loaded clip model in {time.time() - start_time:.2f}s")

    # Load the tags model
    start_time = time.time()
    tags_model, tag_names = load_wd14_model(models_dir=os.path.join(models_dir, "classifiers"))
    if debug:
        print(f"Loaded tags model in {time.time() - start_time:.2f}s")

    # Load the erax object detection model
    start_time = time.time()
    yolo_erax = load_yolo_erax(models_dir=os.path.join(models_dir, "classifiers"))
    if debug:
        print(f"Loaded erax object detection model in {time.time() - start_time:.2f}s")

    # Load the nudenet object detection model
    start_time = time.time()
    nudenet = load_nudenet(models_dir=os.path.join(models_dir, "classifiers"))
    if debug:
        print(f"Loaded nudenet object detection model in {time.time() - start_time:.2f}s")

    return {
        "device": device,
        "clip_model": clip_model,
        "preprocess": preprocess,
        "tags_model": tags_model,
        "tag_names": tag_names,
        "yolo_erax": yolo_erax,
        "nudenet": nudenet,
    }

def classify_image(models, categories, image, debug=False):
    device = models["device"]
    clip_model = models["clip_model"]
    preprocess = models["preprocess"]
    tags_model = models["tags_model"]
    tag_names = models["tag_names"]
    yolo_erax = models["yolo_erax"]
    nudenet = models["nudenet"]

    ret = {}
    start_time = start_time_ratings = time.time()
    ret["ratings"] = classify_image_rating(device, clip_model, preprocess, image)
    start_time_categories = time.time()
    ret["categories"] = classify_image_categories(device, clip_model, preprocess, image, categories)
    cat_len = len(ret["categories"])
    start_time_tags = time.time()
    ret["tags"] = classify_image_tags(tags_model, tag_names, image)
    tag_len = len(ret["tags"])
    start_time_objects = time.time()
    ret["objects"] = classify_image_objects(yolo_erax, nudenet, image)
    obj_len = len(ret["objects"])
    end_time = time.time()
    if debug:
        print(f"Classified Image in {end_time - start_time:.2f}s")
        print(f"     ratings    in {start_time_categories - start_time_ratings:.2f}s")
        print(f"  {cat_len: >2} categories in {start_time_tags - start_time_categories:.2f}s")
        print(f"  {tag_len: >2} tags       in {start_time_objects - start_time_tags:.2f}s")
        print(f"  {obj_len: >2} objects    in {end_time - start_time_objects:.2f}s")
    return ret
