from setuptools import setup, find_packages

setup(
    name="Computational-Pathology-to-Predict-HCC-Recurrence",
    version="0.1.0",
    description="ML Model Integrating Computational Pathology to Predict Early Recurrence of HCC",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aymen Sadraoui & Eve Delegue",
    author_email="eve.delegue@centralesupelec.fr",
    url="https://github.com/EveDelegue/Computational-Pathology-to-Predict-HCC-Recurrence",
    license="Apache-2.0 License",
    # --- Package Configuration ---
    packages=find_packages(),
    # --- Dependencies ---
    install_requires=[
        "openpyxl",
        "tiatoolbox==2.0.1",
        "openslide-bin"
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
