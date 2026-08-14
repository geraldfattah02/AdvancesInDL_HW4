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
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
}


# Original image dimensions for bounding-box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Example:
        00033_04_im.jpg
        -> frame_id = 0x33
        -> view_index = 4
    """
    filename = Path(image_path).name
    parts = filename.split("_")

    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index

    return 0, 0


def draw_detections(
    image_path: str,
    info_path: str,
    font_scale: float = 0.5,
    thickness: int = 1,
    min_box_size: int = 5,
) -> np.ndarray:
    """
    Draw kart detections on an image.

    The filtering here is ONLY for visualization.
    It is not used by extract_kart_objects().
    """

    pil_image = Image.open(image_path)

    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)

    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(
            f"Warning: View index {view_index} out of range "
            f"for detections"
        )
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection

        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (
            (x2_scaled - x1_scaled) < min_box_size
            or (y2_scaled - y1_scaled) < min_box_size
        ):
            continue

        if (
            x2_scaled < 0
            or x1_scaled > img_width
            or y2_scaled < 0
            or y1_scaled > img_height
        ):
            continue

        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        draw.rectangle(
            [(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)],
            outline=color,
            width=thickness,
        )

    return np.array(pil_image)


def _get_kart_name(info: dict, track_id: int) -> str:
    """
    Look up a kart name using its track ID.
    """

    if "karts" in info:
        kart_info = info["karts"]

        # Most likely format:
        # "karts": ["xue", "kiki", ...]
        if isinstance(kart_info, list):
            if 0 <= track_id < len(kart_info):
                entry = kart_info[track_id]

                if isinstance(entry, str):
                    return entry

                if isinstance(entry, dict):
                    name = entry.get(
                        "name",
                        entry.get("kart_name"),
                    )

                    if name is not None:
                        return str(name)

        # Alternative dictionary format
        elif isinstance(kart_info, dict):
            entry = kart_info.get(
                str(track_id),
                kart_info.get(track_id),
            )

            if isinstance(entry, str):
                return entry

            if isinstance(entry, dict):
                name = entry.get(
                    "name",
                    entry.get("kart_name"),
                )

                if name is not None:
                    return str(name)

    # Fallback
    if "objects" in info:
        objects = info["objects"]

        if isinstance(objects, dict):
            obj = objects.get(
                str(track_id),
                objects.get(track_id),
            )

            if isinstance(obj, dict):
                name = obj.get(
                    "name",
                    obj.get("kart_name"),
                )

                if name is not None:
                    return str(name)

        elif isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue

                obj_id = obj.get(
                    "id",
                    obj.get(
                        "track_id",
                        obj.get("instance_id"),
                    ),
                )

                if obj_id is None:
                    continue

                if int(obj_id) == track_id:
                    name = obj.get(
                        "name",
                        obj.get("kart_name"),
                    )

                    if name is not None:
                        return str(name)

    return f"Kart {track_id}"


def extract_kart_objects(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
) -> list:
    """
    Extract all kart detections from an info.json file.

    Important:
    Every detection with class_id == 1 is treated as a kart.

    We do NOT discard small or partially-visible boxes here because
    the assignment defines a kart by class_id == 1. The visualization
    function may filter boxes, but the QA/caption generation should
    not.

    The ego car is the kart whose bounding-box center is closest
    to the center of the image.
    """

    with open(info_path) as f:
        info = json.load(f)

    detections = info["detections"][view_index]

    # Original detection coordinates are 600x400.
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    karts = []

    for detection in detections:
        class_id, track_id, x1, y1, x2, y2 = detection

        class_id = int(class_id)
        track_id = int(track_id)

        # class_id == 1 means kart.
        if class_id != 1:
            continue

        # Convert coordinates to image dimensions used by the model.
        x1 = float(x1) * scale_x
        y1 = float(y1) * scale_y
        x2 = float(x2) * scale_x
        y2 = float(y2) * scale_y

        # Center of the bounding box.
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        kart_name = _get_kart_name(info, track_id)

        karts.append(
            {
                "instance_id": track_id,
                "kart_name": kart_name,
                "center": (center_x, center_y),
                "is_center_kart": False,
            }
        )

    # Find ego car.
    #
    # Euclidean distance:
    #
    # sqrt((x - cx)^2 + (y - cy)^2)
    #
    # We can omit sqrt because the square root is monotonic,
    # so minimizing squared Euclidean distance gives the same result.
    if karts:
        image_center_x = img_width / 2.0
        image_center_y = img_height / 2.0

        center_kart = min(
            karts,
            key=lambda kart: (
                (kart["center"][0] - image_center_x) ** 2
                + (kart["center"][1] - image_center_y) ** 2
            ),
        )

        center_kart["is_center_kart"] = True

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.
    """

    with open(info_path) as f:
        info = json.load(f)

    if "track_name" in info:
        return str(info["track_name"])

    if "track" in info:
        track = info["track"]

        if isinstance(track, str):
            return track

        if isinstance(track, dict):
            return str(
                track.get(
                    "name",
                    track.get("track_name", "Unknown"),
                )
            )

    return "Unknown"


def generate_qa_pairs(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
) -> list:
    """
    Generate QA pairs for a given view.
    """

    karts = extract_kart_objects(
        info_path,
        view_index,
        img_width,
        img_height,
    )

    track_name = extract_track_info(info_path)

    qa_pairs = []

    # ---------------------------------------------------------
    # 1. Ego car
    # ---------------------------------------------------------

    ego_kart = next(
        (
            kart
            for kart in karts
            if kart["is_center_kart"]
        ),
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
    # 2. Total number of karts
    # ---------------------------------------------------------

    qa_pairs.append(
        {
            "question": "How many karts are there in the scenario?",
            "answer": str(len(karts)),
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

    # No ego means there are no relative-position questions.
    if ego_kart is None:
        return qa_pairs

    ego_x, ego_y = ego_kart["center"]

    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    # ---------------------------------------------------------
    # 4. Relative positions
    # ---------------------------------------------------------

    for kart in karts:
        if kart["is_center_kart"]:
            continue

        kart_x, kart_y = kart["center"]
        kart_name = kart["kart_name"]

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

        qa_pairs.append(
            {
                "question": (
                    f"Is {kart_name} to the left or right "
                    "of the ego car?"
                ),
                "answer": horizontal,
            }
        )

        qa_pairs.append(
            {
                "question": (
                    f"Is {kart_name} in front of or behind "
                    "the ego car?"
                ),
                "answer": vertical,
            }
        )

        qa_pairs.append(
            {
                "question": (
                    f"Where is {kart_name} relative to "
                    "the ego car?"
                ),
                "answer": f"{vertical} and {horizontal}",
            }
        )

    # ---------------------------------------------------------
    # 5. Counting questions
    # ---------------------------------------------------------

    qa_pairs.extend(
        [
            {
                "question": (
                    "How many karts are to the left "
                    "of the ego car?"
                ),
                "answer": str(left_count),
            },
            {
                "question": (
                    "How many karts are to the right "
                    "of the ego car?"
                ),
                "answer": str(right_count),
            },
            {
                "question": (
                    "How many karts are in front "
                    "of the ego car?"
                ),
                "answer": str(front_count),
            },
            {
                "question": (
                    "How many karts are behind "
                    "the ego car?"
                ),
                "answer": str(behind_count),
            },
        ]
    )

    return qa_pairs


def check_qa_pairs(
    info_file: str,
    view_index: int,
):
    """
    Check QA pairs for a specific info file and view index.
    """

    info_path = Path(info_file)

    base_name = info_path.stem.replace(
        "_info",
        "",
    )

    image_files = list(
        info_path.parent.glob(
            f"{base_name}_{view_index:02d}_im.jpg"
        )
    )

    if not image_files:
        raise FileNotFoundError(
            f"Could not find image for {info_file}, "
            f"view {view_index}"
        )

    image_file = image_files[0]

    annotated_image = draw_detections(
        str(image_file),
        info_file,
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(
        f"Frame {extract_frame_info(str(image_file))[0]}, "
        f"View {view_index}"
    )
    plt.show()

    qa_pairs = generate_qa_pairs(
        info_file,
        view_index,
    )

    print("\nQuestion-Answer Pairs:")
    print("-" * 50)

    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all(
    split: str = "train",
    data_dir: str | None = None,
    output_file: str | None = None,
):
    """
    Generate QA pairs for all info files in a split.
    """

    data_root = (
        Path(data_dir)
        if data_dir
        else DATA_DIR
    )

    split_dir = data_root / split

    output_path = (
        Path(output_file)
        if output_file
        else split_dir / "balanced_qa_pairs.json"
    )

    all_qa_pairs = []

    for info_file in sorted(
        split_dir.glob("*_info.json")
    ):
        base_name = info_file.stem.replace(
            "_info",
            "",
        )

        for view_index in range(10):
            image_file = (
                split_dir
                / f"{base_name}_{view_index:02d}_im.jpg"
            )

            if not image_file.exists():
                continue

            qa_pairs = generate_qa_pairs(
                str(info_file),
                view_index,
            )

            for qa_pair in qa_pairs:
                all_qa_pairs.append(
                    {
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "image_file": (
                            f"{split}/{image_file.name}"
                        ),
                    }
                )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            all_qa_pairs,
            f,
            indent=2,
        )

    print(
        f"Wrote {len(all_qa_pairs)} QA pairs "
        f"to {output_path}"
    )


def collect_qa_pairs(
    split: str,
    data_dir: Path | None = None,
) -> dict[tuple[str, str], str]:
    """
    Generate QA pairs and index them by:
        (question, image_file)
    """

    data_root = data_dir or DATA_DIR
    split_dir = data_root / split

    qa_index = {}

    for info_file in sorted(
        split_dir.glob("*_info.json")
    ):
        base_name = info_file.stem.replace(
            "_info",
            "",
        )

        for view_index in range(10):
            image_file = (
                split_dir
                / f"{base_name}_{view_index:02d}_im.jpg"
            )

            if not image_file.exists():
                continue

            image_key = (
                f"{split}/{image_file.name}"
            )

            qa_pairs = generate_qa_pairs(
                str(info_file),
                view_index,
            )

            for qa_pair in qa_pairs:
                qa_index[
                    (
                        qa_pair["question"],
                        image_key,
                    )
                ] = qa_pair["answer"]

    return qa_index


def validate(
    split: str = "valid",
    data_dir: str | None = None,
    reference_file: str | None = None,
):
    """
    Compare generated QA pairs against the grader.
    """

    data_root = (
        Path(data_dir)
        if data_dir
        else DATA_DIR
    )

    reference_path = (
        Path(reference_file)
        if reference_file
        else (
            data_root
            / "valid_grader"
            / "balanced_qa_pairs.json"
        )
    )

    with open(reference_path) as f:
        reference = json.load(f)

    generated = collect_qa_pairs(
        split,
        data_root,
    )

    matched = 0
    missing = 0
    mismatched = []

    for entry in reference:
        key = (
            entry["question"],
            entry["image_file"],
        )

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
    print(
        f"Matched: {matched}/{total} "
        f"({100 * matched / total:.1f}%)"
    )
    print(f"Missing from generated data: {missing}")
    print(
        f"Mismatched answers: {len(mismatched)}"
    )

    if mismatched:
        print("\nFirst 10 mismatches:")

        for item in mismatched[:10]:
            print(
                f"  {item['image_file']} | "
                f"{item['question']}"
            )
            print(
                f"    expected: {item['expected']}"
            )
            print(
                f"    got:      {item['got']}"
            )

    return matched, total, mismatched


def main():
    fire.Fire(
        {
            "check": check_qa_pairs,
            "generate_all": generate_all,
            "validate": validate,
        }
    )


if __name__ == "__main__":
    main()