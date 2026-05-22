import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import random
import os
import math

@torch.no_grad()
def visualize_mmdet_mask2former(
    model,
    image_paths,
    device="cuda",
    score_threshold=0.5,
    num_images=3,
    include_bg=True  # 👈 optional toggle
):

    model.eval()

    # -------------------------
    # COLORS (SIMPLIFIED)
    # -------------------------
    fixed_colors = [
        (255, 0, 0),           # Red - Grasper
        (0, 255, 0),           # Green - Harmonic_Ace
        (0, 0, 255),           # Blue - Myoma_Screw
        (255, 255, 0),         # Yellow - Needle_Holder
        (0, 255, 255)          # Cyan - Trocar
    ]
    fixed_stuff_colors = [
        (255, 0, 255),         # Magenta
        (150, 75, 0),          # Brown
    ]

    class_names = ["Grasper", "Harmonic-ACE", "Myoma-Screw", "Needle-Holder", "Trocer", "Uterus"]

    # random image selection
    all_imgs = [os.path.join(image_paths, f)
                for f in os.listdir(image_paths)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    sample_imgs = random.sample(all_imgs, num_images)

    # Prepare grid layout
    cols = 4
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 4, 3 * rows))

    for idx, img_path in enumerate(sample_imgs):

        # -------------------------
        # LOAD IMAGE
        # -------------------------
        img = cv2.imread(img_path)
        orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = orig.shape[:2]
        overlay = orig.copy()

        # -------------------------
        # INFERENCE
        # -------------------------
        result = inference_detector(model, img_path)
        instances = result.pred_instances

        scores = instances.scores.cpu().numpy()
        labels = instances.labels.cpu().numpy()
        masks = instances.masks.cpu().numpy()

        # -------------------------
        # FILTER
        # -------------------------
        keep = scores > score_threshold
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

        # -------------------------
        # SORT BY SCORE (important)
        # -------------------------
        order = np.argsort(-scores)
        scores = scores[order]
        labels = labels[order]
        masks = masks[order]

        # -------------------------
        # BUILD PANOPTIC MAP (CORRECT)
        # -------------------------

        # 1. start from semantic map (stuff + background)
        sem_seg = result.pred_panoptic_seg.sem_seg.squeeze(0).cpu().numpy()

        panoptic_map = np.zeros((H, W), dtype=np.int32)

        # map semantic classes → stuff ids
        # classes: 0-4 things, 5 stuff
        for cls in range(5, 6):  # stuff only
            panoptic_map[sem_seg == cls] = (cls + 1) * 1000

        # optional background
        if include_bg:
            panoptic_map[panoptic_map == 0] = 7000

        # 2. overlay THINGS (priority, overwrite stuff)
        instance_counters = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
        things_instances = []

        for i in range(len(scores)):
            mask = masks[i]
            cls = int(labels[i])

            if mask.sum() == 0:
                continue

            if cls < 5:  # THINGS only
                inst_id = instance_counters[cls]

                # overwrite (this is the KEY for panoptic)
                panoptic_map[mask] = (cls + 1) * 1000 + inst_id

                things_instances.append((cls, inst_id, mask, scores[i]))
                instance_counters[cls] += 1

        # -------------------------
        # COLOR OVERLAY
        # -------------------------
        alpha = 0.7

        for val in np.unique(panoptic_map):
            if val == 0:
                continue

            mask = panoptic_map == val
            class_id = val // 1000 - 1  # back to 0-based
            inst_id = val % 1000

            # THINGS
            if class_id < 5:
                base_color = np.array(fixed_colors[class_id])

                # alternate brightness
                if inst_id % 2 == 0:
                    color = np.clip(base_color * 0.7, 0, 255)
                else:
                    color = base_color

            # STUFF
            else:
                stuff_idx = class_id - 5
                if stuff_idx < len(fixed_stuff_colors):
                    color = np.array(fixed_stuff_colors[stuff_idx])
                else:
                    continue

            overlay[mask] = (
                alpha * color + (1 - alpha) * overlay[mask]
            ).astype(np.uint8)

        # -------------------------
        # DRAW BBOX + LABEL
        # -------------------------
        for cls, inst_id, mask, score in things_instances:
            # remove tiny masks
            if mask.sum() < 100:
                continue

            # keep largest component
            mask_uint8 = mask.astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(mask_uint8)

            if num_labels > 1:
                largest_label = 1 + np.argmax([
                    (labels_im == i).sum() for i in range(1, num_labels)
                ])
                clean_mask = (labels_im == largest_label)
            else:
                clean_mask = mask

            ys, xs = np.where(clean_mask)

            if len(xs) == 0:
                continue

            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()

            color = fixed_colors[cls]

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

            label = class_names[cls]

            cv2.putText(
                overlay,
                f"{label} {score:.2f}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                4,
                cv2.LINE_AA
            )

        # -------------------------
        # SHOW
        # -------------------------
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(overlay)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

visualize_mmdet_mask2former(
    model=model,
    image_paths="datasets/sispanseg/valid/images/",
    device="cuda",  # or "cpu"
    score_threshold=0.5,
    num_images=8,
    include_bg=False)