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
        f"There are {len(karts)} karts in the scenario."
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

            horizontal = (
                "to the left"
                if kart_x < ego_x
                else "to the right"
            )

            vertical = (
                "in front of"
                if kart_y < ego_y
                else "behind"
            )

            captions.append(
                f"{kart['kart_name']} is {horizontal} and {vertical} "
                "the ego car."
            )

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


"""
Usage Examples:
   python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0
   python -m homework.generate_captions generate_all --split train
"""


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all})


if __name__ == "__main__":
    main()
