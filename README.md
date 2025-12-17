# Future Token Prediction in Small GPT-Style Models

Code + experiments for a graduate independent study (CS7391) on **future-token prediction** in decoder-only transformers: what information about *t+k* is already present in hidden states, and when (if ever) **auxiliary output heads** or **multi-token prediction (MTP)** training can reliably exploit it.

**Paper (PDF):** `Final_.pdf` (Dec 15, 2025)  
**Main notebook:** `CS7391-LLM-Multi-Token-Predictions.ipynb` (experiments + plots + logs)

> This is a research repo. Expect some exploratory code and “notebook-first” workflows. 

---

## What this project studies

Autoregressive LMs are trained for next-token prediction, but prior work suggests their hidden states can carry information about tokens *multiple steps ahead*. This project asks:

1) How much future-token predictability can we recover **when the GPT-2 trunk is frozen** and only new heads are trained?   
2) Does initializing a new *t+1* head from the pretrained LM head help *t+2 / t+3* heads learn?   
3) What happens if we train a small **two-head MTP** model end-to-end (t+1 and t+2) on natural text?   
4) How do results relate to **breadcrumbs vs. pre-caching** (Wu et al., 2024) and Meta’s MTP framing?
5) What information useful for the future is present in hidden states?

---

## Repository layout

Top-level directories:

- `Documentation/` — paper drafts, figures, writeups   
- `References/` — PDFs + notes for related work   
- `Check Points/` — saved checkpoints (excluded from GitHub due to size)   

Core artifacts in this public repo:

- `CS7391-LLM-Multi-Token-Predictions.ipynb` — **primary experiment logbook**
- `Final_.pdf` — paper write-up
- `README.md` — you are here

---

## Getting started

### Option A: Run the notebook in Colab (recommended)
The notebook includes a “minimal installs” cell:
```bash
pip install transformers datasets torch
```
(You may also want `accelerate`, `tqdm`, and `matplotlib` depending on which sections you run.)  
Some sections assume a CUDA GPU for reasonable runtime.

### Option B: Run locally
A typical setup:
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install --upgrade pip
pip install torch transformers datasets
```

Notes:
- Many experiments use HuggingFace `transformers` + `datasets`, including WikiText-2 and streaming FineWeb. 
- If you run FineWeb sections, expect **large** data and prefer `streaming=True` (as used in the notebook). 

---

## Reproducing the main experiments

Most results are reproduced by running selected notebook sections end-to-end.

### 1) Toy / intuition-building models (NumPy)
- Decoder-only transformer from scratch (masking, attention mechanics)
- Toy multi-head character-level MTP sanity checks 

### 2) Frozen-trunk GPT-2 auxiliary-head probing (WikiText-2)
- Load GPT-2 small, remove LM head, freeze trunk
- Train linear heads to predict offsets **t+1 / t+2 / t+3** 
- Evaluate with cross-entropy and top-k accuracy; significance via McNemar tests 

Two key variants (paper §4.3 / §5):
- **Experiment 1:** random t+1 and t+2 heads
- **Experiment 2:** cloned pretrained t+1 head + random t+2 head 

### 3) From-scratch two-head MTP (“FutureGPT”) on FineWeb
- Small GPT-style model trained end-to-end with **two heads** (t+1 and t+2) 
- Evaluation reported on ~819,200 tokens from a streaming FineWeb slice 

---

## Datasets & compute

- **WikiText-2** (HuggingFace `datasets`) for GPT-2 auxiliary head experiments   
- **FineWeb** (streaming) for from-scratch / some dual-head experiments   
- Tokenization: GPT-2 BPE 

Compute expectations:
- Frozen-trunk heads are comparatively lightweight.
- From-scratch MTP training is substantially heavier; a GPU is strongly recommended.

---

## “What this repo is / is not”

This repo **is** a research codebase for controlled experiments on representations and training dynamics.   
This repo **is not** a production library, nor a full reproduction of Meta’s large-scale MTP results. 

---

## Citation

If you build on this project, please cite the accompanying paper:

> Harley Trae Gribble. *Future Token Prediction in Small GPT-Style Models: Auxiliary Heads, Myopia, and Multi-Token Training.* 2025. 

(You can also cite this repository URL.)

---

## Related work

- Wu et al. (2024) — myopia vs. pre-caching framing  
- Gloeckle et al. (2024) — multi-token prediction (Meta)   
- Vaswani et al. (2017); Radford et al. (2019) 

---
