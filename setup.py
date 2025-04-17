from setuptools import setup, find_packages

setup(
    name="document-processor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pinecone",
        "python-dotenv",
        "loguru",
        "PyPDF2",
        "python-docx",
        "openpyxl",
    ],
)
