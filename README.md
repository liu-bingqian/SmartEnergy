
#  SmartEnergy: Unified Framework for DNN and Non-DNN Energy Models

This project provides a unified framework for training, evaluating, and comparing deep learning and classical machine learning models for energy system design tasks.

---

##  Features

-  Integrated support for both **deep neural networks (DNNs)** and **scikit-learn regressors**
-  One-line model testing across all regressors
-  Easy configuration for DNNs via `Config.py`
-  Built-in **confidence estimation** for DNN and tree-based models
-  REHO-compatible: directly works with REHO-generated building-energy datasets

---

##  Quick Start

### Train and validate all models:

```python
from energy_ml.models.test_all_regressors import test_all_regressors

validation_results, regressor_list = test_all_regressors(
    X_train, y_train, X_valid, y_valid, verbose=1
)
```

---

##  Model Configuration

-  DNN parameters (layers, activation, etc.) are defined in:
  ```
  energy_ml/Config.py
  ```

-  Other regressors use default settings from `scikit-learn`:
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
  - `KNeighborsRegressor`
  - ... and more

---

##  Dataset Information

Input data is generated using the [REHO framework](https://reho.readthedocs.io) developed by the IPESE group.

- **Inputs**:  
  - Building properties  
  - Weather data  
  - Market data  
- **Target**:  
  - Size of internal energy supply devices (e.g., heat pumps, boilers)

---

##  Jupyter Notebook

You can find a complete end-to-end example in:

```bash
notebooks/REHO_surrogate_model.ipynb
```

This notebook demonstrates:
- Data loading and preprocessing
- Model training and comparison
- Confidence score estimation

---

##  Confidence Score Estimation

This framework supports confidence estimation for both DNN and tree-based models:

| Model Type     | Method                       |
|----------------|------------------------------|
| DNN            | Monte Carlo Dropout (MC Dropout) |
| Tree-based     | Variance across estimators   |

Use the following call:

```python
from energy_ml.utils import confidence_rate

confidence_vector = confidence_rate(model, X_test)
```

---

