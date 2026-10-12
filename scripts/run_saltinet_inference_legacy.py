#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import os

import cv2
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "theano")
import keras

keras.backend.set_image_dim_ordering("th")
from keras.models import load_model


def main():
    parser = argparse.ArgumentParser(description="Run the official Keras-1 SaltiNet checkpoint.")
    parser.add_argument("--manifest", default="data/processed/early_horizon_salient360_v1/manifest.jsonl")
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument(
        "--model",
        default="${SCANDMM_ROOT}/baselines/third_party/saliency-360salient-2017/src/pathnet_model_v2.h5",
    )
    parser.add_argument("--output-dir", default="runs/eh6_saltinet_pretrained_val/volumes")
    args = parser.parse_args()
    rows = []
    with open(args.manifest, "r") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == args.split:
                rows.append(row)
    if not rows:
        raise ValueError("No rows found for split %s" % args.split)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    model = load_model(args.model)
    if tuple(model.output_shape[1:]) != (12, 300, 600):
        raise ValueError("Unexpected SaltiNet output shape %r" % (model.output_shape,))
    for index, row in enumerate(rows):
        path = os.path.join(args.output_dir, row["image_id"] + ".npz")
        if os.path.isfile(path):
            print("[%d/%d] %s already exists" % (index + 1, len(rows), row["image_id"]))
            continue
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Cannot read %s" % row["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (600, 300), interpolation=cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1)).astype(np.float32)[None]
        volume = model.predict(image, batch_size=1)[0].astype(np.float32)
        np.savez_compressed(path, volume=volume)
        print("[%d/%d] %s -> %s" % (index + 1, len(rows), row["image_id"], path))


if __name__ == "__main__":
    main()
