# Speech Commands Recognition

<!-- Short repo description (≤ 350 characters) -->
Speech‑Commands word spotting on 105 k clips / 35 classes with 1 600‑dim spectrograms. Compares mean‑var, MaxAbs & L2 normalisation and several MLP topologies; best model (1600‑140‑70‑35, LR 0.001, batch 100) tops 90 % test accuracy with detailed confusion‑matrix analysis.

---

## 🎯 Goal
Build a lightweight pipeline that recognises a single spoken word (out of 35) in Google’s Speech Commands dataset, relying on compact MLPs instead of heavier CNN architectures.

## 📦 Dataset
* **Google Speech Commands v0.02** – 105 829 one‑second WAV clips labelled with one of 35 trigger words, split into train/val/test folds.  
* Each clip is pre‑converted to a **1 600‑element log‑spectrogram vector** (40 × 40) for quick experimentation.

## 🛠️ Pipeline
1. **Exploration & visualisation** – sanity‑check spectrograms and compute mean/variance images.  
2. **Normalisation study** – compare *mean‑var*, *MaxAbs* and *L2* scaling; L2 + small batches converge quickest.  
3. **Model sweep** – MLPs with 0–2 hidden layers; grid‑search over batch size, learning rate and epochs.  
4. **Chosen model** – `1600 → 140 → 70 → 35`, LR 1e‑3, batch 100: stable > 90 % validation and test accuracy, < 150 s per epoch on CPU.  
5. **Evaluation** – full confusion matrix and per‑class error analysis (classes 8, 14 and 28 are hardest).

## 🔬 Key findings

| Aspect        | Insight                                                                                   |
| ------------- | ----------------------------------------------------------------------------------------- |
| Normalisation | Mean‑var smooths training; L2 edges slightly ahead on final accuracy.                     |
| Batch size    | 100 balances speed and generalisation; very large batches hurt performance.              |
| Depth         | One hidden layer captures most variance; deeper nets risk over‑fitting without gains.    |
