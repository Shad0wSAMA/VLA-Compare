"""Visual test tool for ArUco table coordinates and YOLO cube detection.

This script opens only the configured perception camera; it never connects to or
commands the robot.  It can run before YOLO weights exist by passing
``--aruco-only`` (or by leaving ``perception.weights`` empty).

Controls in the OpenCV window:
  q / Esc  quit
  r        clear accumulated ArUco calibration samples
  s        save an annotated screenshot
  left click  inspect a pixel's table coordinate
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aruco_pick_place import (
    detect_marker_boundary_points,
    detect_marker_centers,
    estimate_table_homography,
    load_config,
    pixel_to_table_xy,
    resolve_local_path,
    undistort_if_configured,
)


LOGGER = logging.getLogger("visualize_aruco_yolo")
WINDOW_NAME = "SO101 ArUco + YOLO visual test"


@dataclass(frozen=True)
class VisualDetection:
    xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    pixel_xy: np.ndarray
    table_xy_m: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aruco_yolo_pick_place.example.json"),
    )
    parser.add_argument("--camera", help="Override robot.perception_camera", default='fixed')
    parser.add_argument("--aruco-only", action="store_true", help="Do not load or run YOLO")
    parser.add_argument("--weights", type=Path, help="Override perception.weights")
    parser.add_argument("--confidence", type=float, help="Override YOLO confidence threshold")
    parser.add_argument("--class-id", type=int, help="Override the cube class ID")
    parser.add_argument(
        "--inference-every",
        type=int,
        default=1,
        help="Run YOLO every N camera frames and reuse the last boxes between runs",
    )
    parser.add_argument("--grid-step-m", type=float, default=0.005)
    parser.add_argument("--sample-window", type=int, default=30)
    parser.add_argument("--screenshot-dir", type=Path, default=Path("outputs/vision_debug"))
    parser.add_argument("--output-video", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    return parser.parse_args()


def validate_visual_config(config: dict[str, Any], camera_name: str) -> list[str]:
    errors: list[str] = []
    robot = config.get("robot")
    perception = config.get("perception")
    if not isinstance(robot, dict):
        return ["robot must be an object"]
    if not isinstance(perception, dict):
        return ["perception must be an object"]
    cameras = robot.get("cameras")
    if not isinstance(cameras, dict) or camera_name not in cameras:
        errors.append(f"camera '{camera_name}' is not present in robot.cameras")
    else:
        camera = cameras[camera_name]
        if not isinstance(camera, dict):
            errors.append(f"robot.cameras.{camera_name} must be an object")
        else:
            for key in ("index_or_path", "width", "height", "fps"):
                if key not in camera or camera[key] in (None, ""):
                    errors.append(f"robot.cameras.{camera_name}.{key} is missing")
    dictionary = str(perception.get("aruco_dictionary", ""))
    if not hasattr(cv2.aruco, dictionary):
        errors.append(f"unknown ArUco dictionary: {dictionary}")
    marker_boundary = perception.get("marker_boundary")
    if not isinstance(marker_boundary, dict) or len(marker_boundary) != 4:
        errors.append("perception.marker_boundary must contain four marker IDs")
    else:
        try:
            marker_xy = np.asarray([entry["xy_m"] for entry in marker_boundary.values()], dtype=float)
            corner_indices = [int(entry["corner_index"]) for entry in marker_boundary.values()]
            if marker_xy.shape != (4, 2) or not np.isfinite(marker_xy).all():
                raise ValueError
            if any(index not in range(4) for index in corner_indices):
                raise ValueError
            if abs(float(cv2.contourArea(cv2.convexHull(marker_xy.astype(np.float32))))) < 1e-6:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append("marker_boundary needs four corner_index 0..3 and non-collinear xy_m points")
    try:
        bounds = np.asarray(perception["table_xy_bounds_m"], dtype=float)
        if bounds.shape != (2, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("perception.table_xy_bounds_m must be [[xmin,xmax],[ymin,ymax]]")
    matrix = perception.get("camera_matrix")
    distortion = perception.get("dist_coeffs")
    if (matrix is None) != (distortion is None):
        errors.append("camera_matrix and dist_coeffs must both be set or both be null")
    return errors


def camera_source(value: Any) -> int | str:
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return int(value)
    return str(Path(str(value)).expanduser())


def open_camera(camera: dict[str, Any]) -> cv2.VideoCapture:
    source = camera_source(camera["index_or_path"])
    if sys.platform == "win32" and isinstance(source, int):
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backend_candidates = [cv2.CAP_ANY]
    capture: cv2.VideoCapture | None = None
    backend_used = cv2.CAP_ANY
    for backend in backend_candidates:
        candidate = cv2.VideoCapture(source, backend)
        if candidate.isOpened():
            capture = candidate
            backend_used = backend
            break
        candidate.release()
    if capture is None:
        raise RuntimeError(
            f"Could not open camera source {source!r} with backends {backend_candidates}. "
            "Check the camera index and close other applications using it."
        )
    if camera.get("fourcc"):
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera["fourcc"]))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera["width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera["height"]))
    capture.set(cv2.CAP_PROP_FPS, int(camera["fps"]))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    LOGGER.info(
        "Camera opened: source=%r backend=%d requested=%dx%d@%d actual=%dx%d@%.1f",
        source,
        backend_used,
        int(camera["width"]),
        int(camera["height"]),
        int(camera["fps"]),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        capture.get(cv2.CAP_PROP_FPS),
    )
    return capture


def table_to_pixel(points_xy_m: np.ndarray, image_to_table: np.ndarray) -> np.ndarray:
    table_to_image = np.linalg.inv(image_to_table)
    points = np.asarray(points_xy_m, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, table_to_image).reshape(-1, 2)


def draw_line_clipped(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    height, width = image.shape[:2]
    p1 = tuple(np.rint(start).astype(int))
    p2 = tuple(np.rint(end).astype(int))
    visible, clipped_start, clipped_end = cv2.clipLine((0, 0, width, height), p1, p2)
    if visible:
        cv2.line(image, clipped_start, clipped_end, color, thickness, cv2.LINE_AA)


def draw_table_overlay(
    image: np.ndarray,
    homography: np.ndarray,
    perception: dict[str, Any],
    grid_step_m: float,
) -> None:
    bounds = np.asarray(perception["table_xy_bounds_m"], dtype=float)
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]

    outline_table = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)
    outline_px = np.rint(table_to_pixel(outline_table, homography)).astype(np.int32)
    cv2.polylines(image, [outline_px], True, (255, 180, 0), 2, cv2.LINE_AA)

    if grid_step_m > 0:
        x_values = np.arange(math.ceil(xmin / grid_step_m) * grid_step_m, xmax + 1e-9, grid_step_m)
        y_values = np.arange(math.ceil(ymin / grid_step_m) * grid_step_m, ymax + 1e-9, grid_step_m)
        for x in x_values:
            endpoints = table_to_pixel(np.array([[x, ymin], [x, ymax]]), homography)
            draw_line_clipped(image, endpoints[0], endpoints[1], (90, 90, 90))
        for y in y_values:
            endpoints = table_to_pixel(np.array([[xmin, y], [xmax, y]]), homography)
            draw_line_clipped(image, endpoints[0], endpoints[1], (90, 90, 90))

    marker_xy = np.asarray(
        [entry["xy_m"] for entry in perception["marker_boundary"].values()], dtype=float
    )
    axis_length = max(
        grid_step_m,
        min(np.ptp(marker_xy[:, 0]), np.ptp(marker_xy[:, 1])) * 0.25,
    )
    origin = marker_xy[np.argmin(np.linalg.norm(marker_xy, axis=1))]
    axes_table = np.array([origin, origin + [axis_length, 0], origin + [0, axis_length]])
    axes_px = table_to_pixel(axes_table, homography)
    draw_line_clipped(image, axes_px[0], axes_px[1], (0, 0, 255), 3)
    draw_line_clipped(image, axes_px[0], axes_px[2], (0, 255, 0), 3)
    cv2.putText(image, "X", tuple(np.rint(axes_px[1]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "Y", tuple(np.rint(axes_px[2]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def marker_error_mm(
    boundary_points: dict[int, np.ndarray], homography: np.ndarray, perception: dict[str, Any]
) -> tuple[float, float] | None:
    expected = {
        int(key): np.asarray(value["xy_m"], dtype=float)
        for key, value in perception["marker_boundary"].items()
    }
    errors = []
    for marker_id, target_xy in expected.items():
        if marker_id in boundary_points:
            errors.append(
                np.linalg.norm(pixel_to_table_xy(boundary_points[marker_id], homography) - target_xy)
                * 1000.0
            )
    if not errors:
        return None
    return float(np.mean(errors)), float(np.max(errors))


def run_yolo(
    model: Any,
    image_bgr: np.ndarray,
    homography: np.ndarray | None,
    perception: dict[str, Any],
    confidence: float,
    class_id: int,
) -> list[VisualDetection]:
    predict_args: dict[str, Any] = {
        "source": image_bgr,
        "conf": confidence,
        "imgsz": int(perception.get("image_size", 640)),
        "verbose": False,
    }
    if perception.get("device") not in (None, ""):
        predict_args["device"] = perception["device"]
    result = model.predict(**predict_args)[0]
    if result.boxes is None:
        return []
    names = result.names
    detections: list[VisualDetection] = []
    for xyxy, score, detected_class in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
        strict=True,
    ):
        detected_class = int(detected_class)
        if detected_class != class_id:
            continue
        center = np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2], dtype=float)
        table_xy = pixel_to_table_xy(center, homography) if homography is not None else None
        class_name = str(names.get(detected_class, detected_class) if isinstance(names, dict) else names[detected_class])
        detections.append(
            VisualDetection(
                xyxy=np.asarray(xyxy, dtype=float),
                confidence=float(score),
                class_id=detected_class,
                class_name=class_name,
                pixel_xy=center,
                table_xy_m=table_xy,
            )
        )
    return detections


def draw_detections(
    image: np.ndarray,
    detections: list[VisualDetection],
    perception: dict[str, Any],
) -> None:
    bounds = np.asarray(perception["table_xy_bounds_m"], dtype=float)
    for detection in detections:
        x1, y1, x2, y2 = np.rint(detection.xyxy).astype(int)
        inside = detection.table_xy_m is None or bool(
            np.all(detection.table_xy_m >= bounds[:, 0]) and np.all(detection.table_xy_m <= bounds[:, 1])
        )
        color = (0, 220, 0) if inside else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(
            image,
            tuple(np.rint(detection.pixel_xy).astype(int)),
            color,
            cv2.MARKER_CROSS,
            16,
            2,
        )
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if detection.table_xy_m is not None:
            label += f"  table=({detection.table_xy_m[0]:.3f}, {detection.table_xy_m[1]:.3f})m"
        cv2.putText(image, label, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def put_status_lines(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 24
    for message, color in lines:
        (width, height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (5, y - height - 5), (12 + width, y + baseline + 3), (0, 0, 0), -1)
        cv2.putText(image, message, (9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += height + 12


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.inference_every <= 0 or args.sample_window <= 0 or args.grid_step_m <= 0:
        raise SystemExit("--inference-every, --sample-window, and --grid-step-m must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    if args.no_display and args.output_video is None and args.max_frames is None:
        raise SystemExit("--no-display requires --output-video or --max-frames")

    config, config_dir = load_config(args.config)
    camera_name = args.camera or str(config.get("robot", {}).get("perception_camera", ""))
    errors = validate_visual_config(config, camera_name)
    if errors:
        print("Visual-test configuration is not ready:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    camera = config["robot"]["cameras"][camera_name]
    perception = config["perception"]
    confidence = args.confidence if args.confidence is not None else float(perception.get("confidence", 0.5))
    class_id = args.class_id if args.class_id is not None else int(perception.get("class_id", 0))
    if not 0.0 <= confidence <= 1.0:
        raise SystemExit("YOLO confidence must be between 0 and 1")

    weights = args.weights.expanduser().resolve() if args.weights else resolve_local_path(perception.get("weights"), config_dir)
    model = None
    if not args.aruco_only and weights is not None and weights.is_file():
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit("Missing ultralytics. Install it with: pip install ultralytics") from exc
        model = YOLO(str(weights))
        LOGGER.info("YOLO weights loaded: %s", weights)
    elif not args.aruco_only:
        LOGGER.warning("YOLO weights are unset/missing; continuing in ArUco-only mode")

    capture = open_camera(camera)
    samples: deque[dict[int, np.ndarray]] = deque(maxlen=args.sample_window)
    homography: np.ndarray | None = None
    detections: list[VisualDetection] = []
    clicked_pixel: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    writer: cv2.VideoWriter | None = None
    screenshot_dir = args.screenshot_dir.expanduser().resolve()
    output_video = args.output_video.expanduser().resolve() if args.output_video else None
    frame_index = 0
    fps_ema = 0.0
    previous_time = time.perf_counter()

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        nonlocal clicked_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixel = np.array([x, y], dtype=float)

    try:
        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        while True:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError("Camera read failed")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = undistort_if_configured(frame_rgb, perception)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            centers, corners, ids = detect_marker_centers(frame_rgb, perception)
            boundary_points = detect_marker_boundary_points(corners, ids, perception)
            expected_ids = {int(marker_id) for marker_id in perception["marker_boundary"]}
            complete = expected_ids.issubset(boundary_points)
            if complete:
                samples.append(boundary_points)
            minimum = int(perception.get("minimum_complete_aruco_samples", 5))
            if len(samples) >= minimum:
                try:
                    candidate, _centers = estimate_table_homography(list(samples), perception)
                    homography = candidate
                except RuntimeError as exc:
                    LOGGER.debug("Homography not updated: %s", exc)

            annotated = frame_bgr.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            for marker_id, center in centers.items():
                cv2.putText(
                    annotated,
                    f"ID {marker_id}",
                    tuple(np.rint(center + [8, -8]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
            for marker_id, point in boundary_points.items():
                cv2.drawMarker(
                    annotated,
                    tuple(np.rint(point).astype(int)),
                    (255, 0, 255),
                    cv2.MARKER_DIAMOND,
                    14,
                    2,
                )

            if homography is not None:
                draw_table_overlay(annotated, homography, perception, args.grid_step_m)
                placement = config.get("placement", {}).get("table_xyz_m")
                if isinstance(placement, list) and len(placement) >= 2:
                    place_px = table_to_pixel(np.asarray(placement[:2], dtype=float)[None], homography)[0]
                    cv2.drawMarker(
                        annotated,
                        tuple(np.rint(place_px).astype(int)),
                        (255, 0, 255),
                        cv2.MARKER_TILTED_CROSS,
                        20,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        "PLACE",
                        tuple(np.rint(place_px + [8, -8]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2,
                    )

            if model is not None and frame_index % args.inference_every == 0:
                detections = run_yolo(model, frame_bgr, homography, perception, confidence, class_id)
            draw_detections(annotated, detections, perception)

            status: list[tuple[str, tuple[int, int, int]]] = []
            detected_ids = sorted(centers)
            status.append((f"camera={camera_name}  fps={fps_ema:.1f}  markers={detected_ids}", (255, 255, 255)))
            if homography is None:
                status.append((f"TABLE: collecting complete samples {len(samples)}/{minimum}", (0, 180, 255)))
            else:
                error = marker_error_mm(boundary_points, homography, perception)
                error_text = ""
                if error is not None:
                    error_text = f"  boundary error mean/max={error[0]:.2f}/{error[1]:.2f}mm"
                status.append((f"TABLE: ready{error_text}", (0, 255, 0)))
            status.append((f"YOLO: {'ready' if model is not None else 'off'}  cubes={len(detections)}", (0, 255, 0) if model is not None else (0, 180, 255)))

            if clicked_pixel is not None:
                cv2.drawMarker(
                    annotated,
                    tuple(np.rint(clicked_pixel).astype(int)),
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )
                if homography is not None:
                    clicked_table = pixel_to_table_xy(clicked_pixel, homography)
                    status.append(
                        (
                            f"CLICK pixel=({clicked_pixel[0]:.0f},{clicked_pixel[1]:.0f}) table=({clicked_table[0]:.4f},{clicked_table[1]:.4f})m",
                            (255, 255, 0),
                        )
                    )
                else:
                    status.append((f"CLICK pixel=({clicked_pixel[0]:.0f},{clicked_pixel[1]:.0f}) table=N/A", (255, 255, 0)))
            status.append(("keys: q quit | r recalibrate | s screenshot | click inspect", (190, 190, 190)))
            put_status_lines(annotated, status)

            if output_video is not None:
                if writer is None:
                    output_video.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(camera["fps"]),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {output_video}")
                writer.write(annotated)

            last_frame = annotated
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    samples.clear()
                    homography = None
                    detections = []
                    clicked_pixel = None
                    LOGGER.info("ArUco calibration samples cleared")
                if key == ord("s"):
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    screenshot = screenshot_dir / f"vision_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    if not cv2.imwrite(str(screenshot), annotated):
                        raise RuntimeError(f"Could not save screenshot: {screenshot}")
                    LOGGER.info("Screenshot saved: %s", screenshot)

            frame_index += 1
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-9)
            fps_ema = instant_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * instant_fps
            previous_time = now
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if last_frame is not None and args.no_display:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot = screenshot_dir / f"vision_last_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(screenshot), last_frame)
        LOGGER.info("Final annotated frame saved: %s", screenshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
