"""qfd — Quantum Feature extraction for financial Fraud Detection (QCNC revision).

Pipeline: dataset -> leakage-free split/preprocess -> Sequential Forward Selection (SFS)
-> quantum feature map -> <Z_i> expectation features -> MLP / XGBoost / SVM -> metrics.
"""
__version__ = "1.0.0"
