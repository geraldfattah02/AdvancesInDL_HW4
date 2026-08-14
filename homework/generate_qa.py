import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .data import DATA_DIR

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the ego kart.
    Filters out karts that are out of sight (outside the image boundaries).

    The ego kart is identified first by track_id == 0 (the player's kart).
    If not found, falls back to the kart closest to the image center.

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)
        min_box_size: Minimum bounding box size to include a kart

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the ego kart
    """

    with open(info_path) as f:
        info = json.load(f)

    detections = info["detections"][view_index]

    # The original detection coordinates are 600x400.
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    karts = []

    for detection in detections:
        class_id, track_id, x1, y1, x2, y2 = detection

        class_id = int(class_id)
        track_id = int(track_id)

        # Only class 1 corresponds to karts.
        if class_id != 1:
            continue

        x1 = float(x1) * scale_x
        y1 = float(y1) * scale_y
        x2 = float(x2) * scale_x
        y2 = float(y2) * scale_y

        # Ignore boxes that are too small.
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue

        # Ignore boxes outside the image.
        if x2 < 0 or x1 > img_width or y2 < 0 or y1 > img_height:
            continue

        # Clamp coordinates to image boundaries.
        x1 = max(0, min(img_width, x1))
        x2 = max(0, min(img_width, x2))
        y1 = max(0, min(img_height, y1))
        y2 = max(0, min(img_height, y2))

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Look up kart name from the karts list using track_id.
        kart_name = None
        if "karts" in info:
            kart_info = info["karts"]

            if isinstance(kart_info, list) and 0 <= track_id < len(kart_info):
                entry = kart_info[track_id]
                kart_name = entry if isinstance(entry, str) else entry.get("name", entry.get("kart_name"))

            elif isinstance(kart_info, dict):
                kart_name = kart_info.get(str(track_id), kart_info.get(track_id))

        if kart_name is None and "objects" in info:
            objects = info["objects"]

            if isinstance(objects, dict):
                obj = objects.get(str(track_id), objects.get(track_id))
                if isinstance(obj, dict):
                    kart_name = obj.get("name", obj.get("kart_name"))

            elif isinstance(objects, list):
                for obj in objects:
                    if isinstance(obj, dict):
                        obj_id = obj.get("id", obj.get("track_id", obj.get("instance_id")))
                        if obj_id is not None and int(obj_id) == track_id:
                            kart_name = obj.get("name", obj.get("kart_name"))
                            break

        # Fall back to the track ID if no explicit name exists.
        if kart_name is None:
            kart_name = f"Kart {track_id}"

        karts.append(
            {
                "instance_id": track_id,
                "kart_name": kart_name,
                "center": (center_x, center_y),
                "is_center_kart": False,
            }
        )

    # Identify ego kart: prefer track_id == 0 (the player's kart),
    # fall back to closest kart to image center.
    if karts:
        ego_kart = next((k for k in karts if k["instance_id"] == 0), None)

        if ego_kart is not None:
            ego_kart["is_center_kart"] = True
        else:
            image_center = (img_width / 2, img_height / 2)
            closest = min(
                karts,
                key=lambda kart: (
                    (kart["center"][0] - image_center[0]) ** 2
                    + (kart["center"][1] - image_center[1]) ** 2
                ) ** 0.5,
            )
            closest["is_center_kart"] = True

    return karts


def count_total_karts(info_path: str) -> int:
    """
    Count the total number of karts in the scenario from the info JSON,
    regardless of visibility in any particular view.

    Args:
        info_path: Path to the info.json file

    Returns:
        Total number of karts in the scenario
    """
    with open(info_path) as f:
        info = json.load(f)

    if "karts" in info:
        kart_info = info["karts"]
        if isinstance(kart_info, (list, dict)):
            return len(kart_info)

    return 0


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)

    # The dataset uses track_name in the info JSON.
    if "track_name" in info:
        return str(info["track_name"])

    if "track" in info:
        track = info["track"]

        if isinstance(track, str):
            return track

        if isinstance(track, dict):
            return str(track.get("name", track.get("track_name", "Unknown")))

    return "Unknown"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """

    karts = extract_kart_objects(
        info_path,
        view_index,
        img_width,
        img_height,
    )

    track_name = extract_track_info(info_path)

    # Count total karts from the scenario metadata (not just visible ones).
    total_karts = count_total_karts(info_path)
    # Fall back to visible count if metadata is unavailable.
    if total_karts == 0:
        total_karts = len(karts)

    qa_pairs = []

    # ---------------------------------------------------------
    # 1. Ego car
    # ---------------------------------------------------------

    ego_kart = next(
        (kart for kart in karts if kart["is_center_kart"]),
        None,
    )

    if ego_kart is not None:
        qa_pairs.append(
            {
                "question": "What kart is the ego car?",
                "answer": ego_kart["kart_name"],
            }
        )

    # ---------------------------------------------------------
    # 2. Total number of karts (from scenario metadata)
    # ---------------------------------------------------------

    qa_pairs.append(
        {
            "question": "How many karts are there in the scenario?",
            "answer": str(total_karts),
        }
    )

    # ---------------------------------------------------------
    # 3. Track
    # ---------------------------------------------------------

    qa_pairs.append(
        {
            "question": "What track is this?",
            "answer": track_name,
        }
    )

    if ego_kart is None:
        return qa_pairs

    ego_x, ego_y = ego_kart["center"]

    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    # ---------------------------------------------------------
    # 4. Relative position of every other kart
    # ---------------------------------------------------------

    for kart in karts:
        if kart["is_center_kart"]:
            continue

        kart_x, kart_y = kart["center"]

        if kart_x < ego_x:
            horizontal = "left"
            left_count += 1
        else:
            horizontal = "right"
            right_count += 1

        if kart_y < ego_y:
            vertical = "front"
            front_count += 1
        else:
            vertical = "back"
            behind_count += 1

        kart_name = kart["kart_name"]

        qa_pairs.append(
            {
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": horizontal,
            }
        )

        qa_pairs.append(
            {
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": vertical,
            }
        )

        qa_pairs.append(
            {
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": f"{vertical} and {horizontal}",
            }
        )

    # ---------------------------------------------------------
    # 5. Counting questions
    # ---------------------------------------------------------

    qa_pairs.extend(
        [
            {
                "question": "How many karts are to the left of the ego car?",
                "answer": str(left_count),
            },
            {
                "question": "How many karts are to the right of the ego car?",
                "answer": str(right_count),
            },
            {
                "question": "How many karts are in front of the ego car?",
                "answer": str(front_count),
            },
            {
                "question": "How many karts are behind the ego car?",
                "answer": str(behind_count),
            },
        ]
    )

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all(split: str = "train", data_dir: str | None = None, output_file: str | None = None):
    """
    Generate QA pairs for all info files in a split and write them to JSON.

    Args:
        split: Dataset split subdirectory under data/ (e.g. "train")
        data_dir: Root data directory (default: project data/)
        output_file: Output JSON path (default: data/{split}/balanced_qa_pairs.json)
    """
    data_root = Path(data_dir) if data_dir else DATA_DIR
    split_dir = data_root / split
    output_path = Path(output_file) if output_file else split_dir / "balanced_qa_pairs.json"

    all_qa_pairs = []
    for info_file in sorted(split_dir.glob("*_info.json")):
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_file = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_file.exists():
                continue

            for qa_pair in generate_qa_pairs(str(info_file), view_index):
                all_qa_pairs.append(
                    {
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "image_file": f"{split}/{image_file.name}",
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"Wrote {len(all_qa_pairs)} QA pairs to {output_path}")


def collect_qa_pairs(split: str, data_dir: Path | None = None) -> dict[tuple[str, str], str]:
    """Generate QA pairs for a split and index them by (question, image_file)."""
    data_root = data_dir or DATA_DIR
    split_dir = data_root / split

    qa_index: dict[tuple[str, str], str] = {}
    for info_file in sorted(split_dir.glob("*_info.json")):
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_file = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_file.exists():
                continue

            image_key = f"{split}/{image_file.name}"
            for qa_pair in generate_qa_pairs(str(info_file), view_index):
                qa_index[(qa_pair["question"], image_key)] = qa_pair["answer"]

    return qa_index


def validate(split: str = "valid", data_dir: str | None = None, reference_file: str | None = None):
    """
    Compare generated QA pairs against the grader reference file.

    For every entry in balanced_qa_pairs.json that shares the same question and
    image_file with generated data, the answer must match.
    """
    data_root = Path(data_dir) if data_dir else DATA_DIR
    reference_path = Path(reference_file) if reference_file else data_root / "valid_grader" / "balanced_qa_pairs.json"

    with open(reference_path) as f:
        reference = json.load(f)

    generated = collect_qa_pairs(split, data_root)

    matched = 0
    missing = 0
    mismatched = []

    for entry in reference:
        key = (entry["question"], entry["image_file"])
        expected = entry["answer"]

        if key not in generated:
            missing += 1
            continue

        if generated[key] == expected:
            matched += 1
        else:
            mismatched.append(
                {
                    "image_file": entry["image_file"],
                    "question": entry["question"],
                    "expected": expected,
                    "got": generated[key],
                }
            )

    total = len(reference)
    print(f"Reference entries: {total}")
    print(f"Matched: {matched}/{total} ({100 * matched / total:.1f}%)")
    print(f"Missing from generated data: {missing}")
    print(f"Mismatched answers: {len(mismatched)}")

    if mismatched:
        print("\nFirst 10 mismatches:")
        for item in mismatched[:10]:
            print(f"  {item['image_file']} | {item['question']}")
            print(f"    expected: {item['expected']}")
            print(f"    got:      {item['got']}")

    return matched, total, mismatched


"""
Usage Examples:
   python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0
   python -m homework.generate_qa generate_all --split train
   python -m homework.generate_qa validate --split valid
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all, "validate": validate})


if __name__ == "__main__":
    main()