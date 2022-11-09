from tvm.driver import tvmc
import numpy as np
import io
import os
from PIL import Image

MOUNT_DIR = '/sample/data/'

records_filename = MOUNT_DIR + 'tune_resnet50_1_3_224_224.json'
model_filename = MOUNT_DIR + 'resnet50-v1-7.onnx'
labels_filename = MOUNT_DIR + 'synset.txt'
test_filename = MOUNT_DIR + 'kitten.jpg'


def load_file(img):
    with open(img, "rb") as fh:
        data = fh.read()
    img = Image.open(io.BytesIO(data))

    return img

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [label.rstrip() for label in f]
        return labels

def preprocess(img):
    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    img = img.crop((16, 16, 240, 240))
    img = img.convert('RGB')
    img = np.array(img).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406])
    stddev = np.array([0.229, 0.224, 0.225])
    img = ((img / 255.0) - mean) / stddev

    img = np.expand_dims(img, 0)
    img = np.transpose(img, [0, 3, 1, 2]).astype(np.float32)

    return img

def postprocess(scores):
    scores = np.squeeze(scores)
    return softmax(scores)

def top_class(N, probs, labels):
    results = np.argsort(probs)[::-1]

    classes = []
    for i in range(N):
        item = results[i]
        classes.append({
            "class": labels[item],
            "prob": float(probs[item])
        })

    return classes

def run_inference():
    image = load_file(test_filename)
    input_data = preprocess(image)

    model = tvmc.load(
        model_filename,
        shape_dict={'data' : [1, 3, 224, 224]}
    )
    package = tvmc.compile(
        model,
        target="llvm",
        tuning_records=records_filename,
        package_path="tvm_package"
    )

    scores = tvmc.run(
        package,
        device="cpu",
        inputs={"data": input_data}
    )
    scores = scores.get_output("output_0")
    probs = postprocess(scores)

    labels = load_labels(labels_filename)
    response = top_class(5, probs, labels)

    print(response)

run_inference()
