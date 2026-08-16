# Training the leaf-disease classifier (maize + beans)

The image classifier (MobileNetV3-small, ONNX) is **trained off-device
on Google Colab's free GPU**, not on the kiosk or a dev laptop. Rationale:

- Training is the only heavy, thermally-intensive step. The ADTC rubric
  penalizes heat/throttling **on the target machine**, so keeping training
  off it entirely is the safe choice.
- The kiosk only ever *runs* the exported model via onnxruntime on CPU
  (milliseconds, tiny RAM), no torch at serve time.
- A free Colab T4 finishes MobileNetV3-small transfer-learning on this data
  in a few minutes.

## What produces the model

`notebooks/train_classifier_colab.ipynb`, a self-contained Colab notebook
(no repo checkout needed). It:

1. Uses the Colab GPU.
2. Downloads **maize** (the four `Corn_(maize)___*` classes from the
   PlantVillage dataset, GitHub `spMohanty`) and **beans** (the iBean/Makerere
   dataset from Hugging Face: angular leaf spot, bean rust, healthy).
3. Assembles an `ImageFolder` with `Crop___Condition` folder names.
4. Transfer-learns MobileNetV3-small (ImageNet backbone, fresh head).
5. Exports `plant_classifier.onnx` + `plant_classifier.labels.json`.

The notebook's preprocessing (224×224, ImageNet normalization, resize-shorter-
side + center-crop for eval) is **kept identical** to
`backend/app/vision/preprocess.py`, so training and serving never drift. The
label map is written in the exact shape `app.vision.labels.ClassLabels` loads.

Local equivalent (if ever training on your own hardware):
`training/train_classifier.py --data <ImageFolder>` does the same thing.

## Where the outputs go

Drop both downloaded files into the repo:

```
models/plant_classifier.onnx
models/plant_classifier.labels.json
```

`models/*.onnx` is gitignored, so the weights stay a local artifact (document
the accuracy in REPORT.md rather than committing the binary). Restart the
backend; `/health` reports `"classifier": {"available": true}`, and `/classify`
+ the kiosk photo upload begin returning real diagnoses.

## Classes (7)

| Crop | Conditions |
|-------|------------|
| maize | Cercospora/Gray leaf spot, Common Rust, Northern Leaf Blight, healthy |
| beans | Angular Leaf Spot, Bean Rust, healthy |

Extend by adding more `Crop___Condition` folders (e.g. tomato from
PlantVillage) and re-running, the label map and ONNX head resize
automatically to the classes present.
