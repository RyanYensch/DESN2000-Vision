import cv2
import threading
import time
import argparse
from ultralytics import YOLO


model = YOLO("yolo26n.pt")

DEFAULT_CLASSES = ["bottle", "cell phone"]
target_ids: list[int] = []

latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# Show result
last_result = None
last_boxes = []
last_analysis_frame = None
result_lock = threading.Lock()

def bg_loop():
    global last_result, last_analysis_frame, latest_frame

    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            latest_frame = None

        if frame is None:
            stop_event.wait(0.01)
            continue

        start = time.perf_counter()

        res = model.predict(source=frame, device="cpu", imgsz=416, conf=0.35, classes=target_ids, verbose=False)[0]

        elapsed_ms = (time.perf_counter() - start) * 1000


        annotated = res.plot()

        detected_count = len(res.boxes) if res.boxes is not None else 0

        status = "DETECTED" if detected_count > 0 else "NOT DETECTED"
        colour = (0, 255, 0) if detected_count > 0 else (0, 0, 255)
        text = f"{status} ({detected_count}): {elapsed_ms:.0f}ms"

        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

        ts = time.strftime("%H:%M:%S")
        cv2.putText(annotated, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        with result_lock:
            last_result = status
            last_analysis_frame = annotated


def main():
    global latest_frame, target_ids

    parser = argparse.ArgumentParser()
    parser.add_argument("--add-class", default=None)
    args = parser.parse_args()

    class_names = DEFAULT_CLASSES.copy()

    if args.add_class is not None:
        class_names.append(args.add_class)


    unknown = [name for name in class_names if name not in model.names.values()]

    if unknown:
        raise ValueError(f"Unsupported YOLO classes: {unknown}")

    target_ids = [class_id for class_id, name in model.names.items() if name in class_names]


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")


    t = threading.Thread(target=bg_loop, daemon=True)
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
