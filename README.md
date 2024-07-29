# Invoice Similarity Matching

## Overview

This script identifies and matches similar invoices based on their content and structure. It uses text extraction, feature extraction, and TF-IDF cosine similarity to find the most similar invoices from a database.

## Setup

### Dependencies

- Python 3.7+
- PyPDF2
- scikit-learn

 Install dependencies using pip:
```bash
pip install PyPDF2 scikit-learn
```

# Usage

### Add an Invoice to the Database
  ```bash
python similarity_check.py add "path/to/invoice.pdf"
```

### Match a New Invoice

```bash
python similarity_check.py match "path/to/new_invoice.pdf"
```

## Example

### Adding Invoices:

```bash
python similarity_check.py add "D:/similarity_check/document similarity/train/1.pdf"
python similarity_check.py add "D:/similarity_check/document similarity/train/2.pdf"
python similarity_check.py add "D:/similarity_check/document similarity/train/3.pdf"
python similarity_check.py add "D:/similarity_check/document similarity/train/4.pdf"
python similarity_check.py add "D:/similarity_check/document similarity/train/5.pdf"
```

### Matching an Invoice:

```bash
python similarity_check.py match "D:/similarity_check/document similarity/test/invoice.pdf"
```

## Output

For the matching command, the script will output the most similar invoice from the database along with the similarity score.

## Approach

1. **Text Extraction**: Extracts text from PDFs using PyPDF2.
2. **Feature Extraction**: Extracts key features like invoice number, date, and total amount.
3. **Similarity Calculation**: Uses TF-IDF vectorization and cosine similarity to compare invoices.

## Notes

- The database of invoices is stored in `database.json`.
- Make sure to add invoices to the database before trying to match a new one.

## Contributing
If you wish to contribute to this project, please fork the repository and submit a pull request with your changes.
