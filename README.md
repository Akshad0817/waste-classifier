# ♻️ Waste Classifier

AI-powered image classifier that predicts the disposal category of a waste item from a photo.

**Live demo:** https://waste-classifier-cagarueyrrdhmhrfnakehb.streamlit.app/

---

## Problem

Household and institutional waste sorting relies almost entirely on manual judgment, which leads to high contamination rates in recycling streams — a single wrongly-sorted item can render an entire batch non-recyclable. This project builds an AI-powered waste classifier that takes a photo of a discarded item and predicts its correct disposal category, giving users an instant, low-friction way to sort correctly at the point of disposal rather than relying on memorized rules or guesswork.

## Approach

- **Model:** MobileNetV2 (ImageNet-pretrained), fine-tuned with a custom classification head on top of a frozen backbone.
- **Architecture:** `MobileNetV2 → GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(6, softmax)`
- **Dataset:** [TrashNet](https://github.com/garythung/trashnet) — 2,527 labeled images across 6 classes (cardboard, glass, metal, paper, plastic, trash), split 80/20 train/validation.
- **Class imbalance:** Addressed with class-weighted loss (`sklearn.utils.class_weight`), since `trash` (137 images) is ~4x smaller than `paper` (594 images).
- **Training:** 15 epochs max with early stopping (patience=3, monitoring validation loss, best weights restored). Validation set is evaluated on clean, unaugmented images — data augmentation (rotation, flip, zoom) is applied to training data only.

## Results

Overall validation accuracy: **77%** (503 held-out images)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| cardboard | 0.91 | 0.76 | 0.83 | 80 |
| glass | 0.75 | 0.77 | 0.76 | 100 |
| metal | 0.78 | 0.82 | 0.80 | 82 |
| paper | 0.79 | 0.90 | 0.84 | 118 |
| plastic | 0.77 | 0.59 | 0.67 | 96 |
| trash | 0.44 | 0.63 | 0.52 | 27 |

`trash` has the weakest precision (0.44) — the model over-predicts this class relative to its true frequency, likely due to the combination of aggressive class weighting (needed to address imbalance) and a small sample size (only 27 validation images, so a handful of misclassifications swings the metric heavily).

## Known limitations (found through manual testing after deployment)

These were discovered by testing the deployed app with real and out-of-distribution photos — not visible in the validation metrics above, since TrashNet doesn't contain examples of most of these cases.

- **Broken/shattered glass is misclassified as paper, confidently (82–92% confidence).** TrashNet's glass images are all intact bottles/jars; the model has no training exposure to shattered glass and doesn't recognize it. This is a real safety concern if the app's guidance were followed literally — broken glass should never go in paper recycling.
- **Texture-based shortcut confusion.** Several images with matte, fibrous, or brown/tan textured surfaces get confidently misclassified as cardboard or paper regardless of actual content — observed on a rock (89% "cardboard"), a red panda photo (83% "cardboard"), and a translucent geodesic dome structure (78% "paper"). This suggests the model is partly keying on surface texture rather than true object identity, likely from TrashNet's limited, studio-lit training images.
- **No out-of-distribution detection.** The model always outputs a prediction across the 6 trained classes, even for completely unrelated images (cars, buildings, people, rockets). A confidence threshold (see below) catches genuinely low-confidence cases, but not confidently-wrong ones like the two above.
- **Trained on clean, single-object, well-lit studio images.** Real-world photos (cluttered backgrounds, poor lighting, multiple items, unusual angles) generally still classify correctly in testing, but confidence is more variable than on TrashNet-style images.

### Confidence threshold
The app treats any prediction below **60% confidence** as "Uncertain" rather than showing a hard answer, to reduce the number of confidently-displayed but shaky guesses (e.g., cars, rockets, and other non-waste objects are usually caught this way). This does **not** catch confidently-wrong predictions like the broken-glass or texture-confusion cases above — a genuine model limitation that would require more diverse training data to fix, not a UI-level threshold.

## What a v2 would need
- Expand training data with broken/damaged item photos, not just intact objects
- Add real out-of-distribution / negative examples as a 7th trained class (not just a confidence cutoff)
- More validation data for `trash`, specifically, to get a reliable precision estimate

## Run locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
Requires `waste_classifier.keras` and `class_indices.json` in the same directory (see repo).

## Tech stack
Python, TensorFlow/Keras, Streamlit, deployed on Streamlit Community Cloud.
