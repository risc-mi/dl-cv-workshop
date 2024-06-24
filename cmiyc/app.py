from typing import Tuple, Optional

from pathlib import Path
import cv2
import PIL
import numpy as np


def load_detector():
    net = cv2.dnn.readNet(
        (Path(__file__).parent / 'vzdb_v2-tiny.weights').as_posix(),
        (Path(__file__).parent / 'vzdb_v2-tiny.cfg').as_posix()
    )
    # ignored if no CUDA available
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    detector = cv2.dnn_DetectionModel(net)
    detector.setInputParams(size=(416, 416), scale=1 / 255, swapRB=False)
    return detector


class CMIYCApp:

    def __init__(self, classifier=None, detector=None, detect: bool = True):
        self.classifier = classifier
        self.detector = detector
        self.detect = detect

        self._class_imgs = {}
        for file in (Path(__file__).parent / 'class_imgs').iterdir():
            self._class_imgs[int(file.stem)] = np.asarray(PIL.Image.open(file).resize((100, 100)))

        self._frame = None
        self._bounding_box = (0.5, 0.5, 1., 1.)
        self._sign_category = 0
        self._prediction = 0
        self._clear_last_frame()

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        return self._bounding_box

    @property
    def sign_category(self) -> int:
        return self._sign_category

    @property
    def prediction(self) -> int:
        return self._prediction

    def normalize_for_detection(self, x: np.ndarray):
        if np.all(self._detection_mean_std[0] == 0) and np.all(self._detection_mean_std[1] == 1):
            return x
        return (x - self._detection_mean_std[0]) / self._detection_mean_std[1]

    def get_frame(self) -> Optional[np.ndarray]:
        if self._frame is None or self._prediction < 0:
            return self._frame
        else:
            h, w = self._frame.shape[:2]
            x, y, w0, h0 = self._bounding_box
            x = max(int(round(x * w)), 0)
            y = max(int(round(y * h)), 0)
            w0 = min(int(round(w0 * w * 0.5)), w - x)       # half width
            h0 = min(int(round(h0 * h * 0.5)), h - y)       # half height

            frame = self._frame.copy()

            # bounding box
            frame[y - h0:y + h0, max(x - 2 - w0, 0):x - w0] = (0, 255, 0)
            frame[y - h0:y + h0, x + w0:x + w0 + 2] = (0, 255, 0)
            frame[max(y - 2 - h0, 0):y - h0, x - w0:x + w0] = (0, 255, 0)
            frame[y + h0:y + h0 + 2, x - w0:x + w0] = (0, 255, 0)

            # class image
            class_img = self._class_imgs.get(self._prediction, None)
            if class_img is not None:
                frame[-100:, :100] = np.round(
                    frame[-100:, :100] * (1. - class_img[..., 3:] / 255) + class_img[..., :3] * (class_img[..., 3:] / 255)
                ).astype(np.uint8)

            return frame

    def process_frame(self, frame: np.array):
        """Process a single frame, i.e., detect the traffic sign in the frame and classify it.

        Parameters
        ----------
        frame : array
            The frame to process, array of shape `(H, W, 3)` with values between 0 and 255. Assumes RGB channels.
        """

        self._frame = frame

        # detection
        self._bounding_box, self._sign_category = \
            self.detect_sign(self._frame) if self.detect else ((0.5, 0.5, 1., 1.), 0)
        if self._sign_category < 0 or self._bounding_box[2] * self._bounding_box[3] <= 0:
            # no sign found
            self._clear_last_frame()
            return

        # extract image for classification
        h, w = self._frame.shape[:2]
        x, y, w0, h0 = self._bounding_box
        x = max(int(round(x * w)), 0)
        y = max(int(round(y * h)), 0)
        w0 = int(round(w0 * w * 0.5))   # half width
        h0 = int(round(h0 * h * 0.5))   # half height
        s = min(max(w0, h0), w - x, h - y, x, y)

        self._prediction = self.classifier.classify(self._frame[y - s:y + s, x - s:x + s])

    def detect_sign(self, x: np.ndarray, confidence_threshold: float = 0.5) \
            -> Tuple[Tuple[float, float, float, float], int]:
        """Apply a traffic sign detector to the give image `x` and return the bounding box and category of the sign
        found. If no sign is found the corresponding category is -1; if multiple signs are found only the one with the
        largest bounding box is returned.

        Parameters
        ----------
        x : array
            The image, an array of shape `(H, W, 3)` with values between 0 and 255. Assumes RGB channels.
        confidence_threshold : float
            Detection confidence threshold.

        Returns
        -------
        bounding_box : 4-tuple
            Bounding box of the detected sign, consisting of the relative xy-position of its center and its relative
            size.
        category : int
            Category of the found sign, or -1.
        """
        if self.detector is not None:
            classes, scores, boxes = self.detector.detect(x, confidence_threshold)
            if len(boxes):
                i = np.argmax(boxes[:, 2] * boxes[:, 3])
                h, w = x.shape[:2]
                w0 = boxes[i, 2] / w
                h0 = boxes[i, 3] / h
                return (boxes[i, 0] / w + w0 * 0.5, boxes[i, 1] / h + h0 * 0.5, w0, h0), int(classes[i])
            else:
                return (0.5, 0.5, 1., 1.), -1
        return (0.5, 0.5, 1., 1.), 0

    def _clear_last_frame(self):
        self._sign_category = -1
        self._prediction = -1

    def run(self, cam_id: int = 0):
        cap = cv2.VideoCapture(cam_id)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame[:, :, 3::-1])                                # BGR -> RGB
            cv2.imshow('Crash Me If You Can', self.get_frame()[:, :, ::-1])       # RGB -> BGR
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
