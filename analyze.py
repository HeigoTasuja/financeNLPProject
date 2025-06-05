import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("nlp_financial_dataset.csv")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# NER extraction
df["entities"] = df["text_column"].apply(lambda text: [(ent.text, ent.label_) for ent in nlp(text).ents])

# TF-IDF keyword analysis
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text_column"])
features = vectorizer.get_feature_names_out()

# Print output
print(df[["text_column", "entities"]])
