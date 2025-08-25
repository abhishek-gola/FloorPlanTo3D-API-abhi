import os
import argparse
import json

import numpy as np
from PIL import Image
from numpy import expand_dims

from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image


class PredictionConfig(Config):
    NAME = "floorPlan_cfg"
    NUM_CLASSES = 1 + 3  # background + wall + window + door
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_class_name(class_id):
    return {1: "wall", 2: "window", 3: "door"}.get(class_id, "unknown")



def load_model():
    config = PredictionConfig()
    model_dir = os.path.join(os.getcwd(), "mrcnn")
    weights_path = os.path.join("weights", "maskrcnn_15_epochs.h5")

    model = MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    model.load_weights(weights_path, by_name=True)
    return model, config



def run_inference(model, config, image_path, output_json):
    # 1) Load image and get its dimensions
    img = Image.open(image_path).convert('RGB')
    image = np.array(img)
    height, width = image.shape[:2]

    # 2) Pre-process and run detection
    molded = mold_image(image, config)
    sample = expand_dims(molded, 0)
    result = model.detect(sample, verbose=0)[0]
    boxes, class_ids = result['rois'], result['class_ids']

    # 3) Build the "points" list, swapping axes per your spec:
    #    JSON.x1 ← original y1, JSON.y1 ← original x1, etc.
    points = []
    door_widths = []
    for (y1, x1, y2, x2), cid in zip(boxes, class_ids):
        new_x1, new_y1 = y1, x1
        new_x2, new_y2 = y2, x2
        points.append({
            "x1": int(new_x1),
            "y1": int(new_y1),
            "x2": int(new_x2),
            "y2": int(new_y2)
        })
        if cid == 3:  # door
            door_widths.append(int(new_x2 - new_x1))

    # 4) Build the "classes" list
    classes = [{"name": get_class_name(cid)} for cid in class_ids]

    # 5) Compute average door width (or 0 if no doors)
    averageDoor = float(sum(door_widths) / len(door_widths)) if door_widths else 0.0

    # 6) Assemble final payload
    payload = {
        "points":      points,
        "classes":     classes,
        "Width":       width,
        "Height":      height,
        "averageDoor": averageDoor
    }

    # 7) Write out JSON
    with open(output_json, 'w') as f:
        json.dump(payload, f, indent=4)
    print(f"JSON written to: {output_json}")



def main():
    parser = argparse.ArgumentParser(
        description="Run Mask R-CNN inference and output JSON in custom format"
    )
    parser.add_argument(
        '--image',  required=True,
        help='Path to input floor-plan image'
    )
    parser.add_argument(
        '--output', required=False, default='output.json',
        help='Path to write JSON results'
    )
    args = parser.parse_args()

    model, config = load_model()
    run_inference(
        model, config,
        image_path  = args.image,
        output_json = args.output
    )


if __name__ == '__main__':
    main()