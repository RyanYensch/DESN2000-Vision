import cv2
import threading
import time
import argparse
from PIL import Image
from transformers import pipeline


classifier = pipeline(task="zero-shot-image-classification", model="openai/clip-vit-base-patch32", device="cpu")


TARGET_CLASSES = [
    "cardboard box",
    "flattened corrugated cardboard",
    "plastic drink bottle",
]


DEFAULT_CLASSES = [
    "cardboard box",
    "flattened corrugated cardboard",
    "plastic drink bottle",
    "glass bottle",
    "plastic wrapper",
    "empty tabletop",
    "unrelated household objects"
]

latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# Show result
last_result = None
last_analysis_frame = None
result_lock = threading.Lock()

def bg_loop(class_names: list[str]):
    global last_result, last_analysis_frame, latest_frame


    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            latest_frame = None

        if frame is None:
            stop_event.wait(0.01)
            continue

        start = time.perf_counter()

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        predictions = classifier(image, candidate_labels=class_names)

        best = predictions[0]

        elapsed_ms = (time.perf_counter() - start) * 1000


        annotated = frame.copy()

        label = best["label"]
        score = best["score"]

        is_target = label in TARGET_CLASSES

        status = "TARGET" if is_target > 0 else "OTHER"
        colour = (0, 255, 0) if is_target > 0 else (0, 165, 255)
        text = f"{status}: {label} - {score:.1%}"

        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

        cv2.putText(annotated, f"Timing: {elapsed_ms:.0f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with result_lock:
            last_result = label
            last_analysis_frame = annotated


def main():
    global latest_frame

    parser = argparse.ArgumentParser()
    parser.add_argument("--list-classes", action="store_true", help="List the names of the classes")
    parser.add_argument("--add-class", default=None)
    args = parser.parse_args()


    #  Display all of the options
    if args.list_classes is True:
        for name in DEFAULT_CLASSES:
            print(name)

        return

    class_names = DEFAULT_CLASSES.copy()

    if args.add_class is not None:
        class_names.append(args.add_class)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")


    t = threading.Thread(target=bg_loop, daemon=True, args=(class_names,))
    t.start()

    print(f"Watching for {', '.join(class_names)}...")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            with frame_lock:
                latest_frame = frame

            ts = time.strftime("%H:%M:%S")
            cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Webcam", frame)

            with result_lock:
                analysis = last_analysis_frame

            if analysis is not None:
                cv2.imshow("Detection", analysis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
