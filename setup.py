from setuptools import setup, find_packages

setup(
    name="ML-HCC-Recurrence-Prediction",
    version="0.1.0",
    description="ML Model Integrating Computational Pathology to Predict Early Recurrence of HCC",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aymen Sadraoui",
    author_email="aymen.sadraoui@centralesupelec.fr",
    url="https://github.com/aymenSadraoui/ML-Model-Integrating-Computational-Pathology-to-Predict-Early-Recurrence-of-HCC",
    license="Apache-2.0 License",
    # --- Package Configuration ---
    packages=find_packages(),
    # --- Dependencies ---
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0",
        "torch>=1.10.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "albumentations>=1.3.0",
        "numcodecs==0.12.1",
        "zarr==2.17.0",
        "openpyxl",
        "tiatoolbox",
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
