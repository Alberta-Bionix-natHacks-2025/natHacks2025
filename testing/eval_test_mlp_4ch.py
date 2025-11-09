# testing/eval_test_mlp_4ch.py
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.data_loader_4ch import EEGFeatureLoader4Ch
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP

SAVE_DIR = Path("data/weights")
RUNTIME_JSON = SAVE_DIR / "manifest_4ch.json"

def main():
    cfg = json.loads(Path(RUNTIME_JSON).read_text())
    fp = FeaturePipeline.load(cfg["pipeline_path"])

    # load fresh windows and use SAME split recipe for a fair check
    loader = EEGFeatureLoader4Ch(subjects=[1])
    X, y, _ = loader.load_dataset()
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    F_va = fp.transform(X_va)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGFeatureMLP(input_dim=F_va.shape[1], hidden_dim=64, n_classes=2, dropout=0.2)
    sd = torch.load(cfg["model"]["weights_path"], map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(F_va).float().to(device)).argmax(1).cpu().numpy()

    print("Val accuracy:", accuracy_score(y_va, preds))
    print("Confusion matrix:\n", confusion_matrix(y_va, preds))
    print("\nClassification report:\n", classification_report(y_va, preds, target_names=["left","right"]))

if __name__ == "__main__":
    main()
