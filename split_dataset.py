from pathlib import Path
import random
import shutil

root = Path("dataset/hydra")

train_images = root / "train/images"
train_labels = root / "train/labels"
val_images = root / "valid/images"
val_labels = root / "valid/labels"

val_images.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

images = [
    p for p in train_images.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

random.seed(67)
random.shuffle(images)

selected = images[:int(len(images) * 0.1)]

for image in selected:
    label = train_labels / f"{image.stem}.txt"

    shutil.move(str(image), str(val_images / image.name))

    if label.exists():
        shutil.move(str(label), str(val_labels / label.name))

print(f"Moved {len(selected)} images into validation")