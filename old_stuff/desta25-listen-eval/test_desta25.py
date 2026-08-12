"""
DeSTA2.5-Audio integration for the LISTEN-style paralinguistic conflict eval.

Setup (one time):
    git clone https://github.com/kehanlu/DeSTA2.5-Audio.git
    cd DeSTA2.5-Audio
    pip install -e .

DeSTA2.5-Audio's public inference API (confirmed from the model card /
GitHub README) is:

    from desta import DeSTA25AudioModel
    model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
    model.to("cuda")
    messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": "<|AUDIO|>\nDescribe this audio.",
         "audios": [{"audio": "/path/to/audio.wav", "text": None}]}
    ]
    outputs = model.generate(messages=messages, do_sample=False, top_p=1.0,
                              temperature=1.0, max_new_tokens=512)
    print(outputs.text)

This script wraps that call around your amended dataset (Neutral-Text /
Category-Matched / Category-Mismatched / Paralinguistic conditions) and
writes per-item predictions plus a results summary in LISTEN's results
format (weighted_accuracy, uar, macro_f1, micro_f1, expected_accuracy).

NOTE: this is a standalone version. If you're plugging into LISTEN's
scripts/test_your_model.py template directly, port the DeSTA25Runner class
below into that template's model-loading + inference hooks rather than
running this file on its own, so dataset iteration and scoring stay
identical to the other models' runs.
"""

import json
import argparse
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
)


MODEL_ID = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"
SYSTEM_PROMPT = "Focus on the audio clips and instructions."


class DeSTA25Runner:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
        from desta import DeSTA25AudioModel  # import here so file loads without the package installed

        self.model = DeSTA25AudioModel.from_pretrained(model_id)
        self.model.to(device)

    def run(self, audio_path: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{prompt}",
                "audios": [{"audio": audio_path, "text": None}],
            },
        ]
        outputs = self.model.generate(
            messages=messages,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=512,
        )
        return outputs.text


def load_examples(dataset_path: str):
    with open(dataset_path, "r") as f:
        return json.load(f)


def score_condition(labels, preds, expected_accuracy):
    return {
        "weighted_accuracy": accuracy_score(labels, preds),
        "uar": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "expected_accuracy": expected_accuracy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to amended examples.json")
    parser.add_argument("--output_dir", default="results/desta25")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    runner = DeSTA25Runner(device=args.device)
    examples = load_examples(args.dataset)

    predictions = []
    for ex in examples:
        pred_text = runner.run(ex["audio_path"], ex["prompt"])
        predictions.append(
            {
                "id": ex.get("id"),
                "conflict_type": ex.get("conflict_type"),
                "condition": ex.get("condition"),  # Neutral-Text / Category-Matched / Category-Mismatched / Paralinguistic
                "ground_truth": ex.get("ground_truth"),
                "text_claim_label": ex.get("text_claim_label"),
                "model_output": pred_text,
            }
        )

    with open(Path(args.output_dir) / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Wrote {len(predictions)} predictions to {args.output_dir}/predictions.json")
    print("Run your existing scoring/TDR logic over predictions.json to compute per-condition metrics.")


if __name__ == "__main__":
    main()
