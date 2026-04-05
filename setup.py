from setuptools import setup, find_packages

setup(
    name="whisper-domain-adaptation",
    version="0.1.0",
    description="Fine-tuning Whisper for domain-specific vocabulary: medical and financial speech",
    author="Kartik Munjal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "peft>=0.10.0",
        "jiwer>=3.0.4",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
)
