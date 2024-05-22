import fiftyone as fo
import fiftyone.zoo as foz

# Download the full training set with all annotations
train_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    max_samples=20000,
)

# Visualize the dataset in the FiftyOne app
session = fo.launch_app(train_dataset)