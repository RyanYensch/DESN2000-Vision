import torch
import cv2
import threading
from PIL import Image
from transformers import AutoModelForCausalLM
import time

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=torch.float32, device_map={"": "cpu"})

model.eval()

OBJECT_NAME: str = "smartphone"

latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# Show result
last_result = None
last_boxes = []
last_analysis_frame = None
result_lock = threading.Lock()

def bg_loop():
    global last_result, last_boxes, last_analysis_frame

    while not stop_event.is_set():
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = model.detect(image, OBJECT_NAME)

        objects = res["objects"]

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if objects:
            for o in objects:
                x1, y1 = int(o["x_min"] * w), int(o["y_min"] * h)
                x2, y2 = int(o["x_max"] * w), int(o["y_max"] * h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            status = "DETECTED"
            colour = (0, 255, 0)
            text = f"{OBJECT_NAME.upper()} DETECTED"
            last_boxes = [(o["x_min"], o["y_min"], o["x_max"], o["y_max"]) for o in objects]
        else:
            status = "NOT DETECTED"
            colour = (0, 0, 255)
            text = "NOT DETECTED"


        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        
        ts = time.strftime("%H:%M:%S")
        cv2.putText(annotated, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        with result_lock:
            last_result = status
            last_analysis_frame = annotated


def main():
    global latest_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")


    t = threading.Thread(target=bg_loop, daemon=True)
    t.start()

    print(f"Watching for {OBJECT_NAME}...")

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
