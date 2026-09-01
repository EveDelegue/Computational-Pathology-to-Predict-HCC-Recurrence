from setuptools import setup, find_packages

setup(
    name="Computational-Pathology-to-Predict-HCC-Recurrence",
    version="0.1.0",
    description="ML Model Integrating Computational Pathology to Predict Early Recurrence of HCC",
    author="Aymen Sadraoui & Eve Delegue",
    author_email="eve.delegue@centralesupelec.fr",
    url="https://github.com/EveDelegue/Computational-Pathology-to-Predict-HCC-Recurrence",
    license="Apache-2.0 License",
    # --- Package Configuration ---
    packages=find_packages(),
    # --- Dependencies ---
    install_requires=[
        "openpyxl",
        "torch",
        "scikit-image",
        "joblib",
        "torchvision",
        "opencv-python==4.8.0.74",
        "pandas",
        "PYyaml",
        "matplotlib",
        "tqdm",
        "openslide-bin",
        "openslide-python",
        "termcolor",
        "seaborn","scikit-learn","catboost",
        "xgboost",
        "scikit-learn",
        "cellseg-models-pytorch",
        "albumentations"
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
