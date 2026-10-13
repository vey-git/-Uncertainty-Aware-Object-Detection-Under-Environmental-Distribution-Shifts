"""
Turns the weapon datasets we downloaded into one dataset that YOLO can train on.

Each download stores its annotations differently. Some are already in YOLO format
but number their classes their own way, one mixes polygon outlines in with its
boxes, one is an XML export from the CVAT labelling tool, and one numbers its
classes so inconsistently that the numbers can't be trusted at all. YOLO needs
every image to follow the same rules, so this script reads each source in whatever
format it came in and writes it back out in one standard layout:

    data/weapons/<source>/images/<split>/<image>
    data/weapons/<source>/labels/<split>/<image>.txt

YOLO finds the label for an image by swapping "images" for "labels" in its path and
changing the extension to .txt, which is why the two folders mirror each other.

Every label file has one line per object:

    <class> <x centre> <y centre> <width> <height>

The four numbers are fractions of the image's width and height (0 to 1) rather than
pixels, so the labels stay correct however YOLO resizes the image. An empty label
file is allowed and means "there's nothing to find in this picture", which is how
the model learns not to fire on ordinary scenes.

Every source ends up with the same two classes: 0 = gun, 1 = knife.

The sources are deliberately kept in separate folders rather than poured into one.
The dissertation is about how detection holds up when the data shifts, so we want
to be able to train on one source and test on another, as well as train on all of
them together. The .yaml files written into data/ are how you tell YOLO which
combination to use, and data/weapons.yaml is everything combined.

Run it from the project folder:

    python prepare_datasets.py

It deletes and rebuilds data/weapons every time, so re-running it is always safe.
The original downloads are only ever read, never changed.
"""
import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "weapons"
YAML_DIR = ROOT / "data"

# The class list every source gets mapped onto. A class's position in this list
# is the number YOLO uses for it, so gun is 0 and knife is 1.
CLASS_NAMES = ["gun", "knife"]
GUN, KNIFE = 0, 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Where each download lives inside the project folder.
KAGGLE_DIR = ROOT / "dataset"
HARIS_DIR = ROOT / "HARIS_Weapon_Detection_Dataset_v2" / "HARIS_Weapon_Detection_Dataset"
WEAPON_TYPES_DIR = ROOT / "archive-3" / "weapon_detection"
PEOPLE_WITH_GUNS_DIR = ROOT / "archive"

# archive-2 (the 141 "Scene" CCTV images) isn't used, because every one of its
# images is already inside HARIS, byte for byte, as the files starting "ds_1__".
# Adding it a second time would just make the model count those images twice.


# ---------------------------------------------------------------------------
# Working with boxes
# ---------------------------------------------------------------------------

def clamp(value):
    """Keep a coordinate inside the image, i.e. between 0 and 1."""
    return max(0.0, min(value, 1.0))


def corners_to_yolo(cls, x1, y1, x2, y2):
    """
    Turn a box given by its top-left and bottom-right corners (as 0-1 fractions)
    into a YOLO label: (class, x centre, y centre, width, height).

    Hand-drawn boxes sometimes stick out slightly past the edge of the image, so the
    corners are pulled back inside it first. If that leaves a box with no width or
    no height, it isn't a real box and we return None so the caller can skip it.
    """
    x1, y1, x2, y2 = clamp(x1), clamp(y1), clamp(x2), clamp(y2)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    return cls, x1 + width / 2, y1 + height / 2, width, height


def read_yolo_labels(label_path):
    """
    Read one YOLO label file. Returns (boxes, skipped), where boxes is a list of
    (class, x centre, y centre, width, height) and skipped counts the lines we
    couldn't use, so the summary at the end can report them.

    A missing label file isn't an error; it just means the image has no objects.

    Most lines are the usual five numbers. A few of the Kaggle labels are
    segmentation outlines instead: the class followed by a list of x y points
    tracing round the object. YOLO would misread those as boxes, so each one is
    replaced by the smallest box that contains all of its points.
    """
    if not label_path.exists():
        return [], 0

    boxes, skipped = [], 0
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            skipped += 1
            continue

        if len(values) == 4:
            cx, cy, w, h = values
            box = corners_to_yolo(cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        elif len(values) >= 6 and len(values) % 2 == 0:
            # x and y alternate along the line, so every other value is an x.
            xs, ys = values[0::2], values[1::2]
            box = corners_to_yolo(cls, min(xs), min(ys), max(xs), max(ys))
        else:
            box = None

        if box is None:
            skipped += 1
        else:
            boxes.append(box)
    return boxes, skipped


def images_in(folder):
    """Every image file in a folder, sorted so that each run processes them in the same order."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# One reader per source
#
# Each reader deals with the quirks of its own download and hands back the same
# thing: (split, image path, boxes, number of label lines skipped), one image at
# a time. Everything after this point can then treat all the sources the same.
# ---------------------------------------------------------------------------

def read_kaggle():
    """
    Kaggle "Weapon and Knife Detection", about 137k images.

    Already in YOLO format with the same numbering as ours (0 = gun, 1 = knife)
    and its own train/val/test split, so the only real work is converting the
    polygon lines. Bear in mind that many of its training images are already
    augmented (rotated, flipped, with added noise or blacked-out squares), which
    is worth remembering when we degrade images on purpose for the experiments.
    """
    for split in ("train", "val", "test"):
        for image in images_in(KAGGLE_DIR / "images" / split):
            label = KAGGLE_DIR / "labels" / split / f"{image.stem}.txt"
            yield (split, image, *read_yolo_labels(label))


def read_haris():
    """
    HARIS Weapon Detection v2, about 11.5k images.

    This is itself a merge of several smaller datasets; the "ds_N__" at the start
    of each file name says which one an image came from. Same class numbering as
    ours. It includes 534 "hard negatives": frames of CCTV footage from the
    UCF-Crime dataset with no weapon in them and empty label files, which are
    useful for teaching the model what *not* to flag. It only comes with train
    and val splits.
    """
    for split in ("train", "val"):
        for image in images_in(HARIS_DIR / split / "images"):
            label = HARIS_DIR / split / "labels" / f"{image.stem}.txt"
            yield (split, image, *read_yolo_labels(label))


# Every image in the weapon-types set is named after the weapon in it, for example
# "Handgun_12.jpeg". These two types count as knives; everything else (rifles,
# shotguns, SMGs, sniper rifles, bazookas, grenade launchers) counts as a gun.
BLADE_TYPES = {"Knife", "Sword"}


def read_weapon_types():
    """
    Weapon-type dataset from archive-3, about 700 images across nine weapon types.

    The boxes themselves are drawn well, but the class numbers in its label files
    are all over the place: handguns show up as 0 and as 1, sniper rifles as 0, 2
    and 4, and none of it matches the numbering in the dataset's own metadata.csv.
    So the class number in the file is ignored and the class is taken from the
    image's file name instead, which is reliable. Every box in a "Knife_..." or
    "Sword_..." image becomes a knife and every box in any other image becomes a gun.

    (The archive-3/test folder is skipped because it has no labels.)
    """
    for split in ("train", "val"):
        for image in images_in(WEAPON_TYPES_DIR / split / "images"):
            weapon_type = image.stem.rsplit("_", 1)[0]
            cls = KNIFE if weapon_type in BLADE_TYPES else GUN

            label = WEAPON_TYPES_DIR / split / "labels" / f"{image.stem}.txt"
            boxes, skipped = read_yolo_labels(label)
            boxes = [(cls, *box[1:]) for box in boxes]
            yield split, image, boxes, skipped


def read_people_with_guns():
    """
    "People with guns" from archive, 11 images.

    Exported from the CVAT labelling tool as a single XML file. Each <image>
    element holds the picture's size and the shapes drawn on it: people as boxes
    and guns as polygons whose points are in pixels, written "x,y;x,y;...". We
    keep only the guns, turn each one into its bounding box and divide by the
    image size to get the 0-1 fractions YOLO wants.

    Eleven images is far too few to learn anything from, so all of them go into a
    test split. They're useful as a small set of completely separate photos to
    check a trained model against.
    """
    root = ET.parse(PEOPLE_WITH_GUNS_DIR / "annotations.xml").getroot()
    for image in root.iter("image"):
        width, height = float(image.get("width")), float(image.get("height"))
        boxes = []

        for shape in image:
            if shape.get("label") != "gun":
                continue
            if shape.tag == "box":
                x1, y1 = float(shape.get("xtl")), float(shape.get("ytl"))
                x2, y2 = float(shape.get("xbr")), float(shape.get("ybr"))
            elif shape.tag == "polygon":
                points = [tuple(map(float, p.split(","))) for p in shape.get("points").split(";")]
                xs, ys = zip(*points)
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                continue

            box = corners_to_yolo(GUN, x1 / width, y1 / height, x2 / width, y2 / height)
            if box:
                boxes.append(box)

        yield "test", PEOPLE_WITH_GUNS_DIR / image.get("name"), boxes, 0


# The order here matters: if the same image turns up in more than one source,
# the first source in this list keeps it (see build() for why).
SOURCES = {
    "kaggle": (KAGGLE_DIR, read_kaggle),
    "haris": (HARIS_DIR, read_haris),
    "weapon_types": (WEAPON_TYPES_DIR, read_weapon_types),
    "people_with_guns": (PEOPLE_WITH_GUNS_DIR, read_people_with_guns),
}

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------

def is_damaged_jpeg(data):
    """
    True if these bytes are a JPEG that doesn't end properly.

    Every JPEG starts with the bytes FF D8 and should finish with FF D9. A few of the
    downloaded images are cut short or have junk after the end marker. They still
    open fine, but when YOLO spots one it repairs it by saving over the file.
    """
    return data[:2] == b"\xff\xd8" and data[-2:] != b"\xff\xd9"


def place_image(src, dst, data):
    """
    Put an image into the new dataset, as a hard link where possible.

    A hard link is a second name for the same file on disk, so it takes up no extra
    space (the Kaggle set alone is over 6 GB) and it keeps working even if the
    original download is deleted later. The catch is that changing the file through
    one name changes it under the other name as well. Damaged JPEGs are therefore
    written as real copies, so that when YOLO repairs them it repairs our copy and
    leaves the original download alone. We also fall back to a copy if linking
    isn't possible, for example because the two folders are on different drives.
    """
    if not is_damaged_jpeg(data):
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    dst.write_bytes(data)


def build():
    """Read every source, remove duplicates and write out the new dataset. Returns per-split counts."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    samples = []
    for source, (folder, reader) in SOURCES.items():
        if not folder.exists():
            print(f"Skipping {source}: {folder.relative_to(ROOT)} not found")
            continue
        print(f"Reading {source} annotations...")
        samples.extend((source, *sample) for sample in reader())

    # Exact duplicates. The same picture often appears more than once, both within
    # a dataset and across datasets (several of these were put together from the
    # same online collections). The Kaggle set, for instance, has around 500 test
    # images that are identical copies of its training images. If a test image was
    # also in the training data, the model has effectively already seen the answer,
    # and its test score comes out better than it really is. For a project that's
    # all about measuring how well the detector really performs, that matters.
    #
    # The rule is simple: an image is only allowed to exist in one place. We go
    # through every training split first, then validation, then test, and the
    # first place an image is seen is where it stays; any later copy is dropped.
    # Training goes first so that nothing the model learns from can also turn up
    # in the data we use to judge it.
    #
    # Copies are spotted using an MD5 hash, a short fingerprint worked out from a
    # file's contents: two files with the same fingerprint are the same image,
    # whatever they happen to be called. (MD5 is fine for this; we only need to
    # spot copies, not stop anyone deliberately forging them.) It only catches
    # byte-for-byte copies though. The same photo saved again at a different
    # size or quality gets a different fingerprint and won't be caught.
    #
    # Python's sort is "stable", meaning items that tie keep their original order,
    # so within each split the sources stay in the order listed in SOURCES.
    samples.sort(key=lambda sample: SPLITS.index(sample[1]))

    print(f"Checking {len(samples):,} images for duplicates and writing them out...")
    seen_hashes = set()
    stats = {}
    for source, split, image, boxes, skipped in samples:
        # Thanks to the sort, all the images for one source and split arrive
        # together, so the folders are set up when the first of them turns up.
        if (source, split) not in stats:
            print(f"  {source}/{split}")
            stats[(source, split)] = Counter()
            used_names = set()
            image_dir = OUT_DIR / source / "images" / split
            label_dir = OUT_DIR / source / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
        counts = stats[(source, split)]

        data = image.read_bytes()
        digest = hashlib.md5(data).hexdigest()
        if digest in seen_hashes:
            counts["duplicates"] += 1
            continue
        seen_hashes.add(digest)

        # Anything that isn't one of our classes is dropped and counted as skipped.
        kept = [box for box in boxes if box[0] in (GUN, KNIFE)]
        counts["skipped"] += skipped + len(boxes) - len(kept)

        # Spaces in file names cause trouble in some tools, so swap them out. Two
        # images could also share a name once that's done (or differ only in their
        # extension, which would give them the same label file), so add a number
        # to the end if the name has already been used in this folder.
        stem = base = image.stem.replace(" ", "_")
        n = 1
        while stem in used_names:
            stem = f"{base}_{n}"
            n += 1
        used_names.add(stem)

        place_image(image, image_dir / f"{stem}{image.suffix}", data)
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in kept]
        (label_dir / f"{stem}.txt").write_text("".join(line + "\n" for line in lines))

        counts["images"] += 1
        counts["background"] += not kept
        for box in kept:
            counts[CLASS_NAMES[box[0]]] += 1

    return stats


# ---------------------------------------------------------------------------
# Config files and summary
# ---------------------------------------------------------------------------

def write_yaml(name, description, splits):
    """
    Write a dataset config file, data/<name>.yaml, that you pass to YOLO to say
    which images to train, validate and test on and what the classes are called.

    There's no "path:" line on purpose. Without one, YOLO reads the folders as
    relative to wherever the .yaml file itself is, so the configs keep working if
    the project folder is moved or renamed. A split can list several folders,
    and YOLO simply uses all of them together.
    """
    lines = [f"# {line}" for line in description.splitlines()]
    lines += ["# Generated by prepare_datasets.py. Re-run that script rather than editing this file.", ""]

    for split in SPLITS:
        folders = splits.get(split, [])
        if len(folders) == 1:
            lines.append(f"{split}: {folders[0]}")
        elif folders:
            lines.append(f"{split}:")
            lines += [f"  - {folder}" for folder in folders]

    lines += ["", "names:"]
    lines += [f"  {i}: {class_name}" for i, class_name in enumerate(CLASS_NAMES)]
    (YAML_DIR / f"{name}.yaml").write_text("\n".join(lines) + "\n")


def write_yamls(stats):
    """One config per source, plus data/weapons.yaml that combines all of them."""
    combined = {split: [] for split in SPLITS}

    for source in SOURCES:
        splits = {split: [f"weapons/{source}/images/{split}"]
                  for split in SPLITS if stats.get((source, split), {}).get("images")}
        if not splits:
            continue
        for split, folders in splits.items():
            combined[split] += folders

        if "train" in splits:
            write_yaml(source, f"Weapon images from the {source} source only.", splits)
        else:
            # YOLO refuses to load a config without train and val entries, even if
            # you only want to test with it. Pointing them at the test images keeps
            # it happy; the comment at the top warns against training on it.
            test_only = {"train": splits["test"], "val": splits["test"], "test": splits["test"]}
            write_yaml(source, f"Weapon images from the {source} source only.\n"
                               "Evaluation only: train and val just point at the test images\n"
                               "because YOLO insists they are set. Don't train with this file.", test_only)

    write_yaml("weapons", "All weapon sources combined. This is the one to train on by default.", combined)


def format_summary(stats):
    """A table of what ended up where, handy for the methodology chapter."""
    columns = ["images", "background", *CLASS_NAMES, "duplicates", "skipped"]
    header = f"{'source':<18}{'split':<7}" + "".join(f"{c:>12}" for c in columns)
    rows = [header, "-" * len(header)]
    for (source, split), counts in stats.items():
        rows.append(f"{source:<18}{split:<7}" + "".join(f"{counts[c]:>12,}" for c in columns))
    rows += [
        "",
        "images      images written to the new dataset",
        "background  images with no weapon in them (empty label file)",
        "gun/knife   number of boxes of each class",
        "duplicates  images dropped because the same image was already used elsewhere",
        "skipped     label lines that couldn't be read or weren't a gun or knife",
    ]
    return "\n".join(rows)


def main():
    stats = build()
    write_yamls(stats)

    summary = format_summary(stats)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)
    print(f"\nDataset written to {OUT_DIR.relative_to(ROOT)}/, configs to {YAML_DIR.relative_to(ROOT)}/*.yaml")


if __name__ == "__main__":
    main()
