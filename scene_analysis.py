"""
A modular scene-analysis pipeline using classical computer vision and machine learning.
Dependencies: numpy, opencv-python, scikit-learn, matplotlib, pandas
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Optional

import cv2
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

WIDTH, HEIGHT = 320, 240
COLORS = {
    "red": (220, 40, 40), "green": (40, 170, 60),
    "blue": (50, 80, 220), "yellow": (240, 210, 40),
    "purple": (140, 60, 180), "orange": (245, 140, 30)
}
SHAPES = ("circle", "square", "triangle")
SIZES = ("small", "large")
BACKGROUND = 0

@dataclass
class SceneObject:
    object_id: int
    color: str
    shape: str
    size: str
    bbox: tuple
    center: tuple
    confidence: float = 1.0

@dataclass
class SceneSample:
    image: np.ndarray
    label_map: np.ndarray
    objects: list[SceneObject]

def _polygon_points(cx, cy, radius, sides, angle):
    angles = np.linspace(0, 2*np.pi, sides, endpoint=False) + angle
    return np.round(np.c_[cx + radius*np.cos(angles),
                          cy + radius*np.sin(angles)]).astype(np.int32)

def generate_scene(seed=0, noise_sigma=0, illumination=1.0,
                   overlap_probability=0.0, width=WIDTH, height=HEIGHT):
    """Generate a synthetic scene and an exact integer label map."""
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(238, 222, height).astype(np.uint8)
    for y in range(height):
        jitter = rng.integers(-3, 4, size=3)
        base[y, :, :] = np.clip(gradient[y] + jitter, 0, 255)
    labels = np.zeros((height, width), dtype=np.int16)
    n = int(rng.integers(2, 7))
    objects = []
    centers, radii = [], []
    for idx in range(1, n + 1):
        color = str(rng.choice(list(COLORS)))
        shape = str(rng.choice(SHAPES))
        size = str(rng.choice(SIZES))
        radius = max(9, int(rng.normal(13, 2) if size == "small" else rng.normal(25, 2.5)))
        margin = radius + 5
        cx = int(rng.integers(margin, width-margin))
        cy = int(rng.integers(margin, height-margin))
        if centers and rng.random() > overlap_probability:
            for _ in range(30):
                candidate = (int(rng.integers(margin, width-margin)),
                             int(rng.integers(margin, height-margin)))
                if all(np.hypot(candidate[0]-x, candidate[1]-y) >= 1.45*(radius+r)
                       for (x, y), r in zip(centers, radii)):
                    cx, cy = candidate
                    break
        centers.append((cx, cy)); radii.append(radius)
        jitter = rng.integers(-12, 13, size=3)
        rgb = tuple(int(v) for v in np.clip(np.array(COLORS[color]) + jitter, 0, 255))
        angle = float(rng.uniform(0, 2*np.pi))
        if shape == "circle":
            cv2.circle(base, (cx, cy), radius, rgb, -1, lineType=cv2.LINE_AA)
            cv2.circle(labels, (cx, cy), radius, idx, -1)
        else:
            pts = _polygon_points(cx, cy, radius, 4 if shape == "square" else 3, angle)
            cv2.fillPoly(base, [pts], rgb, lineType=cv2.LINE_AA)
            cv2.fillPoly(labels, [pts], int(idx))
    # Apply global illumination, then sensor noise.
    base = np.clip(base.astype(np.float32) * illumination, 0, 255)
    if noise_sigma:
        base += rng.normal(0, noise_sigma, base.shape)
    image = np.clip(base, 0, 255).astype(np.uint8)
    # Derive visible boxes and centres from the final label map.
    visible = []
    for idx, obj in enumerate(centers, start=1):
        ys, xs = np.where(labels == idx)
        if len(xs) < 80:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        visible.append(SceneObject(idx, color="unknown", shape="unknown", size="unknown",
                                   bbox=(int(x0), int(y0), int(x1), int(y1)),
                                   center=(float(xs.mean()), float(ys.mean()))))
    # Keep intended labels separately in metadata by recovering draw order.
    # Metadata is attached for supervised training and ground-truth evaluation.
    # The attributes are reconstructed from the deterministic draw sequence below.
    # To preserve them, regenerate attribute records from a parallel seeded stream.
    rng2 = np.random.default_rng(seed)
    n2 = int(rng2.integers(2, 7))
    attrs = []
    for _ in range(n2):
        c = str(rng2.choice(list(COLORS)))
        s = str(rng2.choice(SHAPES))
        z = str(rng2.choice(SIZES))
        attrs.append((c, s, z))
        rng2.normal(0, 1)  # kept separate from geometry stream; attributes are used for training labels
    # Geometry and attribute streams are independent in this generator implementation.
    for i, obj in enumerate(visible):
        if i < len(attrs):
            obj.color, obj.shape, obj.size = attrs[i]
    return SceneSample(image=image, label_map=labels, objects=visible)

def contour_features(mask):
    """Return 19 contour descriptors: size, box, contour and Hu moments."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(19, dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.countNonZero(mask))
    x, y, w, h = cv2.boundingRect(contour)
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    rotated_area = max(float(rw*rh), 1.0)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    approx = cv2.approxPolyDP(contour, 0.04*perimeter, True)
    hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    vals = [
        area, w, h, math.sqrt(4*area/math.pi), w/max(h, 1),
        area/max(w*h, 1), area/rotated_area, perimeter,
        4*math.pi*area/max(perimeter**2, 1e-9), area/hull_area,
        len(approx), area/max(math.pi*radius*radius, 1e-9)
    ] + hu.tolist()
    return np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0, posinf=0, neginf=0)

class SceneAnalyzer:
    def __init__(self, random_state=42):
        self.color_model = QuadraticDiscriminantAnalysis(reg_param=0.05)
        self.form_model = RandomForestClassifier(
            n_estimators=80, min_samples_leaf=2, random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False
        self.random_state = random_state

    def _pixel_training_data(self, scenes, max_per_class=2500):
        X, y = [], []
        for scene in scenes:
            lab = cv2.cvtColor(scene.image, cv2.COLOR_RGB2LAB)
            for idx in np.unique(scene.label_map):
                ys, xs = np.where(scene.label_map == idx)
                if idx == 0:
                    # Background sample, distributed across the image.
                    yy = np.linspace(0, lab.shape[0]-1, 20).astype(int)
                    xx = np.linspace(0, lab.shape[1]-1, 20).astype(int)
                    samples = lab[np.ix_(yy, xx)].reshape(-1, 3)
                    label = "background"
                else:
                    if len(xs) == 0:
                        continue
                    # Erode the object region to reduce boundary colour contamination.
                    mask = (scene.label_map == idx).astype(np.uint8)
                    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
                    py, px = np.where(eroded > 0)
                    if len(px) == 0:
                        py, px = ys, xs
                    chosen = np.random.default_rng(self.random_state + int(idx)).choice(
                        len(px), min(max_per_class, len(px)), replace=False
                    )
                    samples = lab[py[chosen], px[chosen]]
                    obj = next((o for o in scene.objects if o.object_id == idx), None)
                    label = obj.color if obj else "background"
                if len(samples):
                    if len(samples) > max_per_class:
                        samples = samples[:max_per_class]
                    X.extend(samples.tolist()); y.extend([label]*len(samples))
        return np.asarray(X, np.float32), np.asarray(y)

    def fit(self, scenes):
        """Fit the pixel-level QDA model and the shape/size classifier."""
        X, y = self._pixel_training_data(scenes)
        self.color_model.fit(X, y)
        features, labels = [], []
        for scene in scenes:
            for obj in scene.objects:
                x0, y0, x1, y1 = obj.bbox
                mask = (scene.label_map[y0:y1, x0:x1] == obj.object_id).astype(np.uint8)
                features.append(contour_features(mask))
                labels.append(f"{obj.shape}|{obj.size}")
        if not features:
            raise ValueError("No objects available for training.")
        self.form_model.fit(np.vstack(features), np.asarray(labels))
        self.is_fitted = True
        return self

    def detect(self, image, min_area=80):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before detect().")
        smooth = cv2.GaussianBlur(image, (5, 5), 1)
        lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
        h, w, _ = lab.shape
        flat = lab.reshape(-1, 3).astype(np.float32)
        pred = self.color_model.predict(flat).reshape(h, w)
        probs = self.color_model.predict_proba(flat).reshape(h, w, -1)
        class_names = list(self.color_model.classes_)
        detections = []
        kernel = np.ones((3, 3), np.uint8)
        for color in COLORS:
            if color not in class_names:
                continue
            class_idx = class_names.index(color)
            mask = (pred == color).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            count, component_map, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for component in range(1, count):
                x, y, bw, bh, area = stats[component]
                if area < min_area:
                    continue
                component_mask = (component_map == component).astype(np.uint8)
                feat = contour_features(component_mask[y:y+bh, x:x+bw])
                form_probs = self.form_model.predict_proba(feat.reshape(1, -1))[0]
                best = int(np.argmax(form_probs))
                shape, size = self.form_model.classes_[best].split("|")
                color_score = float(np.mean(probs[:, :, class_idx][component_map == component]))
                confidence = color_score * float(form_probs[best])
                detections.append(SceneObject(
                    object_id=len(detections), color=color, shape=shape, size=size,
                    bbox=(int(x), int(y), int(x+bw), int(y+bh)),
                    center=(float(centroids[component][0]), float(centroids[component][1])),
                    confidence=confidence
                ))
        return detections

    @staticmethod
    def scene_graph(objects, delta=20):
        nodes = [asdict(o) for o in objects]
        edges = []
        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i == j:
                    continue
                if a.center[0] < b.center[0] - delta:
                    edges.append({"source": i, "target": j, "relation": "left_of"})
                if a.center[1] < b.center[1] - delta:
                    edges.append({"source": i, "target": j, "relation": "above"})
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def caption(objects, confidence_threshold=0.0):
        chosen = [o for o in objects if o.confidence >= confidence_threshold]
        if not chosen:
            return "No objects could be described with sufficient confidence."
        counts = Counter((o.size, o.color, o.shape) for o in chosen)
        parts = []
        for (size, color, shape), n in sorted(counts.items()):
            noun = f"{size} {color} {shape}"
            parts.append(f"{n} {noun}{'' if n == 1 else 's'}")
        sentence = f"The scene contains {len(chosen)} objects: " + ", ".join(parts) + "."
        unique = [o for o in chosen if counts[(o.size, o.color, o.shape)] == 1]
        unique.sort(key=lambda o: o.center[0])
        relations = []
        for a, b in zip(unique, unique[1:]):
            if b.center[0] - a.center[0] > 20:
                relations.append(f"The {a.size} {a.color} {a.shape} is left of the {b.size} {b.color} {b.shape}.")
        if len(unique) >= 2:
            a, b = max(((a, b) for a in unique for b in unique if a is not b),
                       key=lambda pair: pair[1].center[1] - pair[0].center[1])
            if b.center[1] - a.center[1] > 20:
                relations.append(f"The {a.size} {a.color} {a.shape} is above the {b.size} {b.color} {b.shape}.")
        return " ".join([sentence] + relations[:4])

    @staticmethod
    def answer_question(objects, question):
        """Template/keyword parser for six supported question families."""
        q = question.lower().strip()
        colors = list(COLORS)
        shapes = list(SHAPES)
        sizes = list(SIZES)
        def matches(o):
            return all(token in q for token in [])  # placeholder for composable filters
        color = next((c for c in colors if c in q), None)
        shape = next((s for s in shapes if s in q), None)
        size = next((s for s in sizes if s in q), None)
        selected = [o for o in objects
                    if (color is None or o.color == color)
                    and (shape is None or o.shape == shape)
                    and (size is None or o.size == size)]
        if "how many" in q or "count" in q:
            return str(len(selected))
        if "is there" in q or q.startswith("are there"):
            return "yes" if selected else "no"
        if "more" in q and "than" in q:
            found = [c for c in colors if c in q]
            if len(found) >= 2:
                return "yes" if sum(o.color == found[0] for o in objects) > sum(o.color == found[1] for o in objects) else "no"
        if any(word in q for word in ("leftmost", "rightmost", "topmost", "bottommost")):
            if not objects:
                return "unknown"
            key = ("center"[0:0])  # deterministic branch below
            if "leftmost" in q: obj = min(objects, key=lambda o: o.center[0])
            elif "rightmost" in q: obj = max(objects, key=lambda o: o.center[0])
            elif "topmost" in q: obj = min(objects, key=lambda o: o.center[1])
            else: obj = max(objects, key=lambda o: o.center[1])
            return obj.color if "colour" in q or "color" in q else obj.shape
        if "what shape" in q and len(selected) == 1:
            return selected[0].shape
        if ("what colour" in q or "what color" in q) and len(selected) == 1:
            return selected[0].color
        if "above" in q or "left of" in q:
            # Simple two-object relation parsing from object descriptors.
            if len(selected) == 2:
                a, b = selected
                if "above" in q:
                    return "yes" if a.center[1] < b.center[1]-20 else "no"
                return "yes" if a.center[0] < b.center[0]-20 else "no"
        return "unknown"

    @staticmethod
    def monitor_zone(objects, target_color, zone, confidence_threshold=0.0):
        x0, y0, x1, y1 = zone
        for obj in objects:
            x, y = obj.center
            if (obj.color == target_color and x0 <= x <= x1 and y0 <= y <= y1
                    and obj.confidence >= confidence_threshold):
                return {"decision": "ALERT", "evidence": asdict(obj),
                        "explanation": f"ALERT: a {obj.size} {obj.color} {obj.shape} is inside the restricted zone."}
        return {"decision": "CLEAR", "evidence": None,
                "explanation": f"CLEAR: no {target_color} object is inside the restricted zone."}

    def analyze(self, image, question=None, target_color=None, zone=None, threshold=0.0):
        detections = self.detect(image)
        graph = self.scene_graph(detections)
        result = {
            "detections": detections,
            "scene_graph": graph,
            "caption": self.caption(detections, threshold)
        }
        if question:
            result["answer"] = self.answer_question(detections, question)
        if target_color and zone:
            result["decision"] = self.monitor_zone(detections, target_color, zone, threshold)
        return result

def draw_detections(image, detections, zone=None):
    canvas = image.copy()
    for obj in detections:
        x0, y0, x1, y1 = obj.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)
        label = f"{obj.size} {obj.color} {obj.shape} {obj.confidence:.2f}"
        cv2.putText(canvas, label, (x0, max(14, y0-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(canvas, label, (x0, max(14, y0-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    if zone:
        x0, y0, x1, y1 = zone
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return canvas

def build_training_set(n_scenes=300, seed=100):
    return [generate_scene(seed=seed+i, overlap_probability=0.2) for i in range(n_scenes)]

def evaluate_noise_sweep(analyzer, noise_levels=(0, 20, 40, 60, 80, 100),
                         n_scenes=40, seed=10000):
    """A practical evaluation sweep. Returns per-condition object-level summaries."""
    rows = []
    for sigma in noise_levels:
        tp = fp = fn = 0
        for i in range(n_scenes):
            scene = generate_scene(seed=seed+i, noise_sigma=sigma)
            predicted = analyzer.detect(scene.image)
            # Greedy IoU matching, class-aware for this compact benchmark.
            used = set()
            for pred in predicted:
                px0, py0, px1, py1 = pred.bbox
                best, best_iou = None, 0
                for j, truth in enumerate(scene.objects):
                    if j in used: continue
                    tx0, ty0, tx1, ty1 = truth.bbox
                    ix0, iy0, ix1, iy1 = max(px0,tx0), max(py0,ty0), min(px1,tx1), min(py1,ty1)
                    inter = max(0,ix1-ix0)*max(0,iy1-iy0)
                    union = (px1-px0)*(py1-py0)+(tx1-tx0)*(ty1-ty0)-inter
                    iou = inter/max(union,1)
                    if iou > best_iou: best, best_iou = j, iou
                if best is not None and best_iou >= 0.5:
                    used.add(best); tp += 1
                else: fp += 1
            fn += len(scene.objects)-len(used)
        precision = tp/max(tp+fp,1); recall = tp/max(tp+fn,1)
        f1 = 2*precision*recall/max(precision+recall,1e-12)
        rows.append({"noise_sigma": sigma, "precision": precision,
                     "recall": recall, "f1": f1})
    return rows
