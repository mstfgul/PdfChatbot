# setup.py
from setuptools import setup, find_packages

setup(
    name="pdf_qa_app",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.27.0",
        "openai>=1.3.0",
        "langchain>=0.0.267",
        "langchain-openai>=0.0.1",
        "python-dotenv>=1.0.0",
        "PyPDF2>=3.0.0",
        "tiktoken>=0.5.0",
    ],
    python_requires=">=3.8",
)
