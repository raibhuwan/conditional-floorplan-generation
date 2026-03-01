# Conditional Semantic Floor Plan Generation

This project investigates conditional semantic floor plan generation using deep learning.  
The objective is to generate structured residential layouts from simple inputs and evaluate both semantic accuracy and spatial quality.

The framework supports:
- A UNet-based conditional segmentation baseline
- A GAN-based extension (planned)
- Lightweight post-generation refinement
- Structured spatial evaluation metrics

---

## 1. Project Overview

The system takes:

- A binary building outline mask
- A scalar room count condition

and predicts a multi-class semantic layout at **256 × 256 resolution**.

The full pipeline consists of:

1. **Preprocessing**
   - SVG parsing
   - Semantic mask generation
   - Class consolidation
   - Dataset filtering

2. **Generation**
   - Conditional UNet baseline
   - (Planned) Conditional GAN extension

3. **Refinement**
   - Hill-climbing refinement (experimental)
   - Morphological refinement (adopted method)

4. **Evaluation**
   - Mean Intersection over Union (mIoU)
   - Adjacency similarity
   - Compactness
   - Consistency to baseline prediction

---

## 2. Dataset

This project uses the **CubiCasa5K** dataset:

Kalervo et al. (2019)  
CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis  
https://zenodo.org/record/2613548

The dataset is not included in this repository due to size constraints.

After downloading, place it in:

```
data/cubicasa5k/
```