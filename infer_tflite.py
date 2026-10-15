import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument(
    "--model",
    default="artifacts/plant_disease_dynamic_quant.tflite"
)
parser.add_argument("--labels", default="artifacts/labels.json")
args = parser.parse_args()

with open(args.labels, "r", encoding="utf-8") as f:
    labels = json.load(f)

image = Image.open(args.image).convert("RGB").resize((224, 224))
x = np.asarray(image, dtype=np.float32)[None, ...]

interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

x = x.astype(input_details["dtype"])
interpreter.set_tensor(input_details["index"], x)
interpreter.invoke()

scores = interpreter.get_tensor(output_details["index"])[0]
top_indices = np.argsort(scores)[::-1][:5]

print("Top predictions:")
for idx in top_indices:
    name = labels[int(idx)].replace("___", " — ").replace("_", " ")
    print(f"{name}: {scores[idx]:.4f}")
