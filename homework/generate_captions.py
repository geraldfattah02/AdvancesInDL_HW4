import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .data import DATA_DIR
from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    from .generate_qa import (
        extract_kart_objects,
        extract_track_info,
    )

    karts = extract_kart_objects(
        info_path,
        view_index,
        img_width,
        img_height,
    )

    track_name = extract_track_info(info_path)

    captions = []

    # ---------------------------------------------------------
    # 1. Ego car
    # ---------------------------------------------------------

    ego_kart = next(
        (kart for kart in karts if kart["is_center_kart"]),
        None,
    )

    if ego_kart is not None:
        captions.append(
            f"{ego_kart['kart_name']} is the ego car."
        )

    # ---------------------------------------------------------
    # 2. Counting
    # ---------------------------------------------------------

    captions.append(
        f"There are {len(karts)} karts in the scene."
    )

    # ---------------------------------------------------------
    # 3. Track
    # ---------------------------------------------------------

    captions.append(
        f"The track is {track_name}."
    )

    # ---------------------------------------------------------
    # 4. Relative positions
    # ---------------------------------------------------------

    if ego_kart is not None:
        ego_x, ego_y = ego_kart["center"]

        for kart in karts:
            if kart["is_center_kart"]:
                continue

            kart_x, kart_y = kart["center"]
            kart_name = kart["kart_name"]

            if kart_y < ego_y:
                captions.append(f"{kart_name} is in front of the ego car.")
            else:
                captions.append(f"{kart_name} is behind the ego car.")

            if kart_x < ego_x:
                captions.append(f"{kart_name} is left of the ego car.")
            else:
                captions.append(f"{kart_name} is right of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all(split: str = "train", data_dir: str | None = None, output_file: str | None = None):
    """
    Generate caption pairs for all info files in a split and write them to JSON.

    Args:
        split: Dataset split subdirectory under data/ (e.g. "train")
        data_dir: Root data directory (default: project data/)
        output_file: Output JSON path (default: data/{split}/balanced_captions.json)
    """
    data_root = Path(data_dir) if data_dir else DATA_DIR
    split_dir = data_root / split
    output_path = Path(output_file) if output_file else split_dir / "balanced_captions.json"

    all_captions = []
    for info_file in sorted(split_dir.glob("*_info.json")):
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_file = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_file.exists():
                continue

            for caption in generate_caption(str(info_file), view_index):
                all_captions.append(
                    {
                        "image_file": f"{split}/{image_file.name}",
                        "caption": caption,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Wrote {len(all_captions)} captions to {output_path}")


    print(f"Wrote {len(all_captions)} captions to {output_path}")


def collect_captions(split: str, data_dir: Path | None = None) -> dict[str, set[str]]:
    """Generate captions for a split and index them by image_file."""
    data_root = data_dir or DATA_DIR
    split_dir = data_root / split

    captions_by_image: dict[str, set[str]] = {}
    for info_file in sorted(split_dir.glob("*_info.json")):
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_file = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_file.exists():
                continue

            image_key = f"{split}/{image_file.name}"
            captions_by_image.setdefault(image_key, set())
            for caption in generate_caption(str(info_file), view_index):
                captions_by_image[image_key].add(caption)

    return captions_by_image


def validate(split: str = "valid", data_dir: str | None = None, reference_file: str | None = None):
    """
    Check that each correct caption from all_mc_qas.json appears in generated data.
    """
    data_root = Path(data_dir) if data_dir else DATA_DIR
    reference_path = Path(reference_file) if reference_file else data_root / "valid_grader" / "all_mc_qas.json"

    with open(reference_path) as f:
        reference = json.load(f)

    generated = collect_captions(split, data_root)

    matched = 0
    missing = []

    for entry in reference:
        image_file = entry["image_file"]
        correct_caption = entry["candidates"][entry["correct_index"]]

        if image_file not in generated:
            missing.append({"image_file": image_file, "caption": correct_caption, "reason": "image not generated"})
            continue

        if correct_caption in generated[image_file]:
            matched += 1
        else:
            missing.append({"image_file": image_file, "caption": correct_caption, "reason": "caption not found"})

    total = len(reference)
    print(f"Reference entries: {total}")
    print(f"Matched: {matched}/{total} ({100 * matched / total:.1f}%)")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nFirst 10 missing:")
        for item in missing[:10]:
            print(f"  {item['image_file']} | {item['caption']} ({item['reason']})")

    return matched, total, missing


"""
Usage Examples:
   python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0
   python -m homework.generate_captions generate_all --split train
   python -m homework.generate_captions validate --split valid
"""


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all, "validate": validate})


if __name__ == "__main__":
    main()
