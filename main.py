import moondream as md
from PIL import Image
import time

model = md.photon("moondream3.1-9B-A2B")

OBJECT_NAME: str = "cardboard box"

def check_frame(path: str) -> bool:
    image = Image.open(path)

    res = model.query(image, f"Is there anything other than a {OBJECT_NAME} in the image?")

    return res["answer"].strip().lower().startswith("no")

def main():
    while True:
        triggered = check_frame("frame.jpg")
        if triggered:
            print(f"{OBJECT_NAME.upper()} DETECTED")

        time.sleep(0.5)


if __name__ == "__main__":
    main()
