import re
import json
import argparse
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATABASE_FILE = 'database.json'


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_features(text):
    features = {}
    invoice_number_match = re.search(r'Invoice Number:\s*(\w+)', text)
    date_match = re.search(r'Date:\s*(\d{2}/\d{2}/\d{4})', text)
    amount_match = re.search(r'Total Amount:\s*\$?(\d+\.\d{2})', text)

    features['invoice_number'] = invoice_number_match.group(1) if invoice_number_match else 'N/A'
    features['date'] = date_match.group(1) if date_match else 'N/A'
    features['amount'] = amount_match.group(1) if amount_match else 'N/A' 
    return features


def calculate_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def load_database():
    try:
        with open(DATABASE_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def save_database(database):
    with open(DATABASE_FILE, 'w') as file:
        json.dump(database, file)


def add_invoice_to_database(file_path):
    database = load_database()
    text = extract_text_from_pdf(file_path)
    features = extract_features(text)
    database.append({'file_path': file_path, 'features': features, 'text': text})
    save_database(database)
    print(f"Database now contains {len(database)} entries.")


def find_most_similar_invoice(new_invoice_path):
    database = load_database()
    new_text = extract_text_from_pdf(new_invoice_path)
    print(f"Text extracted from new invoice: {new_text[:200]}...")
    similarities = []
    for entry in database:
        print(f"Comparing with invoice: {entry['file_path']}")
        similarity = calculate_similarity(new_text, entry['text'])
        similarities.append((similarity, entry['file_path']))
        print(f"Similarity with {entry['file_path']}: {similarity}")
    if not similarities:
        print("No similarities found.")
        return None, None
    most_similar = max(similarities, key=lambda x: x[0])
    return most_similar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoice Similarity Matching")
    parser.add_argument('action', choices=['add', 'match'], help="Add a new invoice to the database or match a new invoice.")
    parser.add_argument('file_path', help="Path to the invoice PDF file.")
    args = parser.parse_args()
    if args.action == 'add':
        add_invoice_to_database(args.file_path)
        print(f"Added {args.file_path} to database.")
    elif args.action == 'match':
        similarity, matched_invoice = find_most_similar_invoice(args.file_path)
        if matched_invoice:
            print(f"Most similar invoice: {matched_invoice} with similarity score of {similarity:.2f}")
        else:
            print("No similar invoices found.")
