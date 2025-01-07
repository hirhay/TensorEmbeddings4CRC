# README

## Related paper
- Journal: Scientific Reports
- DOI : 10.1038/s41598-024-85062-z
- Title : Data-driven prediction of chemically relevant compositions in multi-component systems using tensor embeddings

## Overview
In this repository, there are **two scripts** that work together to predict multi-component (e.g., pseudo-ternary) compositions:

1. **tucker_ps2.py** (tensor decomposition workflow):  
   - Performs **Tucker decomposition** on labeled data (e.g., `data_ps2`), in which 'ps2' means pseudo-binary compounds.
   - Handles **label encoding**, permutations, cross-validation, **Optuna** hyperparameter tuning for decomposition rank, and more.
   - Produces factor matrices (`factor_axis_...` files) and a decomposed tensor (score reconstruction).

2. **multi_rs.py** (multinary composition prediction):  
   - Takes the factor matrices (produced by the `tucker_ps2.py` script) to create embeddings.
   - Applies various **machine-learning classification** methods (RandomForest, LightGBM, GaussianProcessClassifier, or Autoencoder-based anomaly detection) on the embedded data.
   - Conducts cross-validation, **Optuna** tuning for model hyperparameters, and outputs predictions/plots.

You will typically **run the first script** to produce factor matrices and/or decompose your data, then **use the second script** to train a classifier (or regressor) on the embedded features.

---

## Data Format (Example: `data_ps2`)
Your data files named `data_ps?` should follow a **space-separated** format similar to:

```
score ion1 ion2 ratio
2 Ge+4.00 Mg+2.00 1:2
2 Sr+2.00 Ti+4.00 1:1
```

- **score**: An integer indicating whether a sample is a positive example (`2`) or a negative example (`1`).  
- **ion1, ion2, ...**: Columns for different ions, typically with their oxidation states (`+4.00`, `+2.00`, etc.).  
- **ratio**: A ratio like `1:2` or `1:1`, specifying how ions combine into a compound.

### Example Interpretation
- **Row 1**: `2 Ge+4.00 Mg+2.00 1:2`  
  - **score** = `2` (positive)  
  - Suggests a composition “Mg2GeO4” (because Mg is in the +2 state and Ge in +4, with ratio 1:2).  
- **Row 2**: `2 Sr+2.00 Ti+4.00 1:1`  
  - **score** = `2` (also positive)  
  - Represents “SrTiO3” (where Sr is +2, Ti is +4, ratio 1:1).

Note that if a row had a score of `1`, it would indicate a **negative example**.

---

## Usage Workflow

### 1. Run the First Script (Tucker Decomposition)
1. **Prepare your CSV** (e.g., `data_ps2` and `data_ps3`) in the described format.  
2. **Adjust** any parameters or command-line arguments for the first script (e.g., ranks for decomposition, location of the CSV).  
3. **Execute** the first script:
   ```bash
   python tucker_ps2.py [eval_metrics] [eval_top] [plot-option]
   ```
4. **Outputs**:
   - It will produce **factor-axis** files (`factor_axis_*`) containing factor matrices from Tucker decomposition.
   - A result file from the decomposition (`result_tucker_*`), possibly subfolder organization, etc.

### 2. Run the Second Script (Condition Prediction)
1. **Check** that you have factor-axis files in the current directory (or specify them in your config) generated by the first script.  
2. **Set up** the `config` dictionary in the second script (paths to your data, how many folds for cross-validation, which model to use, etc.).  
3. **Execute** the second script:
   ```bash
   python multi_rs.py [eval_metrics] [eval_top] [pred_model]
   ```
   - `eval_metrics`: The evaluation metric (e.g., `roc_auc`, `f1`, etc.).

**During this step**, the script uses the factor matrices to embed each sample as vectors (e.g., computing weighted means, covariances) and trains a chosen ML model (RF, LGBM, GPC, or Autoencoder). It also can produce **predictions** for any hypothetical or leftover samples.

---

## Additional Notes

- The example given (`2 Ge+4.00 Mg+2.00 1:2` → Mg2GeO4, `2 Sr+2.00 Ti+4.00 1:1` → SrTiO3) shows **positive** examples (score = 2). If you have negative examples, **change** score to 1 in the same data format.
- The second script will often combine real and hypothetical data (combinatorial expansions of `ion? + ratio`), embedding them via factor-axis DataFrames, and then run classification/regression.
- Check **plots** (ROC, PR, violin, etc.) and output CSVs after each run for insight into model performance and recommended compositions.

---

## Contact / Issues
If you have any questions about these scripts or want to suggest improvements, feel free to open an issue or contact the contributors. We hope this workflow helps you explore both **tensor decomposition** and **ML-based condition prediction** for multi-component materials.
