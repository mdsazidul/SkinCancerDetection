# SkinCancerDetection

An AI-powered web application for dermoscopic skin lesion classification across 7 diagnostic categories, built with a fine-tuned **Swin Transformer Base** and deployed via **Streamlit**.

> ⚠️ **Medical Disclaimer:** This is a screening tool only — not a clinical diagnostic device. Always consult a dermatologist.

---

## Demo

| Benign Result | Malignant Alert |
|---|---|
| Green panel · routine action prompt | Red panel · urgent referral triggered |

**Live App:** [Hugging Face Spaces](https://huggingface.co/sazidshovon/SkinCancerDetection) *(or run locally — see below)*

---

## Features

- **7-class classification** — akiec, bcc, bkl, df, mel, nv, vasc (HAM10000)
- **Monte Carlo Dropout** — 15-sample uncertainty estimation flags unreliable predictions
- **Grad-CAM heatmaps** — visual attention overlay showing model focus regions
- **Risk-stratified UI** — color-coded results (🔴 malignant / 🟠 precancerous / 🟢 benign) with plain-language action prompts
- **Image validation** — HSV-based skin-tone check guards against non-dermatoscopic uploads
- **Adjustable thresholds** — confidence and uncertainty sliders via sidebar

---

## Quickstart

```bash
git clone https://github.com/your-username/dermai
cd dermai
pip install -r requirements.txt
```

Set your Hugging Face token (model is in a private repo):
```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
# or: huggingface-cli login
```

Run the app:
```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
torch
timm
torchvision
opencv-python
plotly
huggingface_hub
Pillow
numpy
scikit-learn
```

---

## Model

| Property | Value |
|---|---|
| Architecture | Swin Transformer Base (384×384) |
| Pre-training | ImageNet-21k |
| Dataset | HAM10000 (7 classes) |
| Training | Google Colab Pro+ · A100 GPU |
| Hosted at | `sazidshovon/SkinCancerDetection` on HF Hub |

**Key training details:** WeightedRandomSampler + class-weighted Focal Loss (γ=2.5) for class imbalance · two-phase fine-tuning · gradient checkpointing · mixed precision (AMP).

---

## Project Structure

```
├── app.py              # Streamlit application
├── train.py            # Training script (Colab)
├── requirements.txt
└── README.md
```

---

## Authors

**MD Sazidul Islam** & **Dr. Shakil Akhtar**  
[www.mdsazidulislam.site](https://www.sazidshovon.tech)

---

## License

MIT License — free to use for research and educational purposes.
