# testing/eval_4ch.py
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.data_loader_4ch import EEGFeatureLoader4Ch
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP

PIPE = "data/weights/feature_pipeline_4ch.joblib"
WEI  = "data/weights/feature_mlp_4ch.pth"

def main():
    loader = EEGFeatureLoader4Ch(subjects=[1], tmin=0.5, tmax=2.5)
    X, y, _ = loader.load_dataset()  # (N,4,500), 0/1
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    fp: FeaturePipeline = FeaturePipeline.load(PIPE)
    Xtr_f = fp.transform(Xtr)
    Xva_f = fp.transform(Xva)

    sd = torch.load(WEI, map_location="cpu")
    # infer hidden sizes from checkpoint
    hid = []
    if "net.0.weight" in sd: hid.append(sd["net.0.weight"].shape[0])
    if "net.4.weight" in sd: hid.append(sd["net.4.weight"].shape[0])
    if not hid: hid = [64, 64]

    model = EEGFeatureMLP(input_dim=Xtr_f.shape[1], hidden_dims=tuple(hid), n_classes=2, dropout=0.2)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(Xva_f).float())
        pred = logits.argmax(1).numpy()

    print("Val accuracy:", (pred == yva).mean())
    print("\nConfusion matrix:\n", confusion_matrix(yva, pred))
    print("\nClassification report:\n", classification_report(yva, pred, target_names=["left","right"]))

if __name__ == "__main__":
    main()
