#!/usr/bin/env python3
"""
Production-Ready Ultra-Enhanced Model
Fixed prediction interface for deployment
"""

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import pickle
import json
from datetime import datetime
from scipy.sparse import hstack

warnings.filterwarnings('ignore')

class ProductionExpenseClassifier:
    def __init__(self, model, tfidf, scaler, numeric_features, label_encoder):
        self.model = model
        self.tfidf = tfidf
        self.scaler = scaler
        self.numeric_features = numeric_features
        self.label_encoder = label_encoder
        self.keyword_categories = {
            'food': ['food', 'restaurant', 'cafe', 'meal', 'lunch', 'dinner', 'breakfast', 
                    'snacks', 'grocery', 'milk', 'bread', 'delivery', 'dining', 'pizza', 
                    'burger', 'chicken', 'rice', 'curry', 'naan', 'biryani', 'idli', 'vada',
                    'dosa', 'hotel', 'dhaba', 'swiggy', 'zomato', 'mess', 'canteen'],
            'transport': ['auto', 'taxi', 'train', 'bus', 'metro', 'fuel', 'gas', 'parking', 
                         'uber', 'ola', 'transport', 'vehicle', 'car', 'petrol', 'diesel',
                         'rapido', 'flight', 'cab', 'toll', 'railway', 'place', 'station'],
            'bills': ['bill', 'electric', 'electricity', 'water', 'internet', 'phone', 
                     'mobile', 'subscription', 'utility', 'service', 'recharge', 'broadband',
                     'wifi', 'airtel', 'jio', 'bsnl', 'vodafone', 'postpaid', 'prepaid',
                     'maid', 'cook', 'rent', 'garbage', 'booster', 'data', 'month'],
            'shopping': ['shopping', 'store', 'mall', 'amazon', 'flipkart', 'clothes', 'electronics',
                        'meesho', 'myntra', 'ajio', 'purchase', 'buy', 'market'],
            'health': ['hospital', 'doctor', 'pharmacy', 'medical', 'health', 'clinic', 'medicine',
                      'apollo', 'diagnostic', 'lab', 'test', 'scan', 'dental'],
            'entertainment': ['movie', 'cinema', 'game', 'netflix', 'entertainment', 'concert',
                             'event', 'pvr', 'inox', 'bookmyshow'],
            'investment': ['investment', 'mutual fund', 'sip', 'stock', 'share', 'deposit',
                          'insurance', 'ppf', 'fd', 'rd', 'equity', 'portfolio', 'saving'],
            'income': ['salary', 'income', 'interest', 'dividend', 'bonus', 'cashback',
                      'reward', 'refund', 'gpay reward'],
        }

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def _count_keywords(self, text, keywords):
        return sum(1 for keyword in keywords if keyword in text.lower())

    def _extract_features(self, note, amount):
        note_clean = self._clean_text(note)
        features = {
            'Amount': amount, 'LogAmount': np.log1p(amount),
            'AmountRange': self._get_amount_range(amount),
            'SqrtAmount': np.sqrt(amount),
            'DayOfWeek': pd.Timestamp.now().dayofweek,
            'Month': pd.Timestamp.now().month, 'Day': pd.Timestamp.now().day,
            'IsWeekend': 1 if pd.Timestamp.now().dayofweek >= 5 else 0,
            'IsMonthEnd': 1 if pd.Timestamp.now().day >= 25 else 0,
            'IsMonthStart': 1 if pd.Timestamp.now().day <= 5 else 0,
            'TextLength': len(note_clean), 'WordCount': len(note_clean.split()),
            'UpperCaseRatio': sum(1 for c in note if c.isupper()) / len(note) if note else 0,
            'DigitRatio': sum(1 for c in note if c.isdigit()) / len(note) if note else 0,
            'UniqueWordRatio': len(set(note_clean.split())) / len(note_clean.split()) if note_clean.split() else 0,
        }
        for category, keywords in self.keyword_categories.items():
            features[f'{category}_keywords'] = self._count_keywords(note_clean, keywords)
        features['HasAmountPattern'] = 1 if re.search(r'\d+\s*(rs|rupees|inr|\$)', note.lower()) else 0
        features['HasTimePattern'] = 1 if re.search(r'\d{1,2}:\d{2}', note) else 0
        features['HasPlacePattern'] = 1 if re.search(r'place\s+\d+', note.lower()) else 0
        features['HasUPIPattern'] = 1 if re.search(r'upi|gpay|phonepe|paytm', note.lower()) else 0
        features['HasMonthPattern'] = 1 if re.search(r'\d+\s*month', note.lower()) else 0
        return note_clean, features

    def _get_amount_range(self, amount):
        if amount < 50: return 0
        elif amount < 200: return 1
        elif amount < 500: return 2
        elif amount < 1000: return 3
        elif amount < 5000: return 4
        else: return 5

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            note = data['Note'].iloc[0] if 'Note' in data.columns else ''
            amount = data['Amount'].iloc[0] if 'Amount' in data.columns else 0
        else:
            note = data.get('Note', '') if isinstance(data, dict) else ''
            amount = data.get('Amount', 0) if isinstance(data, dict) else 0
        note_clean, numeric_features_dict = self._extract_features(note, amount)
        text_features = self.tfidf.transform([note_clean])
        numeric_array = [numeric_features_dict.get(feat, 0) for feat in self.numeric_features]
        if len(numeric_array) < len(self.numeric_features):
            numeric_array.extend([0] * (len(self.numeric_features) - len(numeric_array)))
        else:
            numeric_array = numeric_array[:len(self.numeric_features)]
        numeric_scaled = self.scaler.transform([numeric_array])
        X_combined = hstack([text_features, numeric_scaled])
        prediction_encoded = self.model.predict(X_combined)[0]
        return self.label_encoder.inverse_transform([prediction_encoded])[0]


def train_production_model():
    print("🚀 Training Production-Ready Ultra Model")
    print("=" * 60)

    df = pd.read_csv("../data/exp.csv")
    print(f"Dataset: {df.shape}")

    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['Category'])
    df_clean = df_clean[df_clean['Category'].str.strip() != '']
    df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
    df_clean = df_clean[df_clean['Amount'] > 0]
    df_clean['Note'] = df_clean['Note'].fillna('Unknown Transaction').astype(str)

    # IMPROVED category mapping — groups more categories together
    category_mapping = {
        'Food': 'Food & Dining', 'food': 'Food & Dining', 'Dinner': 'Food & Dining',
        'Lunch': 'Food & Dining', 'breakfast': 'Food & Dining', 'Grocery': 'Food & Dining',
        'snacks': 'Food & Dining', 'Milk': 'Food & Dining', 'Ice cream': 'Food & Dining',
        'Transportation': 'Transportation', 'Train': 'Transportation', 'auto': 'Transportation',
        'Tourism': 'Transportation',
        'subscription': 'Bills & Utilities', 'Household': 'Bills & Utilities',
        'water (jar /tanker)': 'Bills & Utilities', 'maid': 'Bills & Utilities',
        'Cook': 'Bills & Utilities', 'garbage disposal': 'Bills & Utilities',
        'Rent': 'Bills & Utilities',
        'Family': 'Personal & Family', 'Festivals': 'Personal & Family',
        'Culture': 'Personal & Family', 'Social Life': 'Personal & Family',
        'Grooming': 'Personal & Family',
        'Salary': 'Income', 'Interest': 'Income', 'Dividend earned on Shares': 'Income',
        'Bonus': 'Income', 'Tax refund': 'Income', 'Amazon pay cashback': 'Income',
        'Gpay Reward': 'Income', 'scrap': 'Income',
        'Investment': 'Investment', 'Recurring Deposit': 'Investment',
        'Public Provident Fund': 'Investment', 'Equity Mutual Fund E': 'Investment',
        'Equity Mutual Fund A': 'Investment', 'Equity Mutual Fund F': 'Investment',
        'Equity Mutual Fund C': 'Investment', 'Equity Mutual Fund D': 'Investment',
        'Equity Mutual Fund B': 'Investment', 'Small Cap fund 2': 'Investment',
        'Small cap fund 1': 'Investment', 'Share Market': 'Investment',
        'Life Insurance': 'Investment', 'Fixed Deposit': 'Investment',
        'Saving Bank account 1': 'Investment', 'Saving Bank account 2': 'Investment',
        'Maturity amount': 'Investment',
        'Other': 'Miscellaneous', 'Petty cash': 'Miscellaneous',
        'Documents': 'Miscellaneous', 'Self-development': 'Miscellaneous',
        'Apparel': 'Apparel', 'Gift': 'Gift',
        'Healthcare': 'Health', 'Medical/Healthcare': 'Health', 'Health': 'Health',
        'Education': 'Education', 'Money transfer': 'Money transfer', 'Beauty': 'Beauty',
    }
    df_clean['Category'] = df_clean['Category'].replace(category_mapping)

    # Keep categories with 15+ samples
    category_counts = df_clean['Category'].value_counts()
    valid_categories = category_counts[category_counts >= 15].index
    df_clean = df_clean[df_clean['Category'].isin(valid_categories)]

    print(f"Cleaned: {df_clean.shape}")
    print(f"Categories: {sorted(df_clean['Category'].unique())}")

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def count_keywords(text, keywords):
        return sum(1 for kw in keywords if kw in text.lower())

    df_clean['Note_clean'] = df_clean['Note'].apply(clean_text)
    df_clean['LogAmount'] = np.log1p(df_clean['Amount'])
    df_clean['SqrtAmount'] = np.sqrt(df_clean['Amount'])
    df_clean['AmountRange'] = pd.cut(df_clean['Amount'],
                                     bins=[0, 50, 200, 500, 1000, 5000, float('inf')],
                                     labels=[0, 1, 2, 3, 4, 5]).astype(int)
    df_clean['TextLength'] = df_clean['Note_clean'].str.len()
    df_clean['WordCount'] = df_clean['Note_clean'].str.split().str.len()
    df_clean['UpperCaseRatio'] = df_clean['Note'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0)
    df_clean['DigitRatio'] = df_clean['Note'].apply(
        lambda x: sum(1 for c in x if c.isdigit()) / len(x) if x else 0)
    df_clean['UniqueWordRatio'] = df_clean['Note_clean'].apply(
        lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0)

    keyword_categories = {
        'food': ['food', 'restaurant', 'chicken', 'pizza', 'meal', 'dining', 'naan', 'curry',
                 'biryani', 'lunch', 'dinner', 'breakfast', 'cafe', 'snack', 'burger', 'rice',
                 'bread', 'milk', 'grocery', 'swiggy', 'zomato', 'hotel', 'dhaba', 'mess',
                 'idli', 'vada', 'dosa', 'canteen', 'atta', 'egg'],
        'transport': ['taxi', 'auto', 'fuel', 'parking', 'uber', 'transport', 'gas', 'metro',
                      'bus', 'train', 'ola', 'rapido', 'petrol', 'diesel', 'flight', 'ticket',
                      'cab', 'vehicle', 'toll', 'railway', 'place', 'station', 'residence'],
        'bills': ['bill', 'electric', 'internet', 'phone', 'subscription', 'utility',
                  'recharge', 'broadband', 'wifi', 'airtel', 'jio', 'bsnl', 'vodafone',
                  'electricity', 'water', 'postpaid', 'prepaid', 'maid', 'cook', 'rent',
                  'garbage', 'booster', 'data', 'month'],
        'shopping': ['shopping', 'amazon', 'store', 'clothes', 'electronics', 'mall',
                     'flipkart', 'meesho', 'myntra', 'ajio', 'purchase', 'buy', 'market'],
        'health': ['hospital', 'doctor', 'medical', 'health', 'pharmacy', 'clinic',
                   'medicine', 'apollo', 'diagnostic', 'lab', 'test', 'scan', 'dental'],
        'entertainment': ['movie', 'game', 'entertainment', 'netflix', 'cinema',
                          'concert', 'event', 'pvr', 'inox', 'bookmyshow'],
        'investment': ['investment', 'mutual fund', 'sip', 'stock', 'share', 'deposit',
                       'insurance', 'ppf', 'fd', 'rd', 'equity', 'portfolio', 'saving'],
        'income': ['salary', 'income', 'interest', 'dividend', 'bonus', 'cashback',
                   'reward', 'refund', 'gpay reward'],
    }

    for category, keywords in keyword_categories.items():
        df_clean[f'{category}_keywords'] = df_clean['Note_clean'].apply(
            lambda x: count_keywords(x, keywords))

    df_clean['HasAmountPattern'] = df_clean['Note'].apply(
        lambda x: 1 if re.search(r'\d+\s*(rs|rupees|inr)', x.lower()) else 0)
    df_clean['HasTimePattern'] = df_clean['Note'].apply(
        lambda x: 1 if re.search(r'\d{1,2}:\d{2}', x) else 0)
    df_clean['HasPlacePattern'] = df_clean['Note'].apply(
        lambda x: 1 if re.search(r'place\s+\d+', x.lower()) else 0)
    df_clean['HasUPIPattern'] = df_clean['Note'].apply(
        lambda x: 1 if re.search(r'upi|gpay|phonepe|paytm', x.lower()) else 0)
    df_clean['HasMonthPattern'] = df_clean['Note'].apply(
        lambda x: 1 if re.search(r'\d+\s*month', x.lower()) else 0)

    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date'])
        df_clean['DayOfWeek'] = df_clean['Date'].dt.dayofweek
        df_clean['Month'] = df_clean['Date'].dt.month
        df_clean['Day'] = df_clean['Date'].dt.day
        df_clean['IsWeekend'] = (df_clean['DayOfWeek'] >= 5).astype(int)
        df_clean['IsMonthEnd'] = (df_clean['Day'] >= 25).astype(int)
        df_clean['IsMonthStart'] = (df_clean['Day'] <= 5).astype(int)
        df_clean['Quarter'] = df_clean['Date'].dt.quarter

    print("🔧 Preparing features...")

    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 3),
                            min_df=1, max_df=0.9, sublinear_tf=True)
    text_features = tfidf.fit_transform(df_clean['Note_clean'])

    numeric_features = [
        'Amount', 'LogAmount', 'AmountRange', 'SqrtAmount',
        'TextLength', 'WordCount', 'UpperCaseRatio', 'DigitRatio', 'UniqueWordRatio',
        'food_keywords', 'transport_keywords', 'bills_keywords', 'shopping_keywords',
        'health_keywords', 'entertainment_keywords', 'investment_keywords', 'income_keywords',
        'HasAmountPattern', 'HasTimePattern', 'HasPlacePattern', 'HasUPIPattern', 'HasMonthPattern'
    ]
    if 'DayOfWeek' in df_clean.columns:
        numeric_features.extend(['DayOfWeek', 'Month', 'Day', 'IsWeekend',
                                  'IsMonthEnd', 'IsMonthStart', 'Quarter'])

    available_features = [f for f in numeric_features if f in df_clean.columns]
    X_numeric = df_clean[available_features].fillna(0)
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_combined = hstack([text_features, X_numeric_scaled])
    y = df_clean['Category']

    print(f"Features: {X_combined.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print(f"Training: {X_train.shape}, Testing: {X_test.shape}")

    print("🤖 Training optimized ensemble...")
    models = {
        'logistic_regression': LogisticRegression(
            C=10.0, max_iter=5000, class_weight='balanced', solver='saga'),
        'random_forest': RandomForestClassifier(
            n_estimators=400, max_depth=25, min_samples_split=2,
            class_weight='balanced', random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=7,
            subsample=0.8, random_state=42),
        'svm': CalibratedClassifierCV(
            LinearSVC(C=5.0, max_iter=5000, class_weight='balanced'))
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        trained_models[name] = model
        print(f"  {name}: {score:.4f} ({score*100:.2f}%)")

    best_name = max(trained_models, key=lambda n: trained_models[n].score(X_test, y_test))
    best_score = trained_models[best_name].score(X_test, y_test)

    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in trained_models.items()], voting='soft')
    ensemble.fit(X_train, y_train)
    ensemble_score = ensemble.score(X_test, y_test)
    print(f"\n🏆 ENSEMBLE ACCURACY: {ensemble_score:.4f} ({ensemble_score*100:.2f}%)")

    if best_score > ensemble_score:
        final_model = trained_models[best_name]
        final_score = best_score
        print(f"✅ Using {best_name}: {final_score*100:.2f}%")
    else:
        final_model = ensemble
        final_score = ensemble_score
        print(f"✅ Using ensemble: {final_score*100:.2f}%")

    y_pred = final_model.predict(X_test)
    print("\n📋 Classification Report:")
    unique_test_classes = np.unique(y_test)
    target_names_subset = [label_encoder.classes_[i] for i in unique_test_classes]
    print(classification_report(y_test, y_pred, labels=unique_test_classes,
                                target_names=target_names_subset))

    production_model = ProductionExpenseClassifier(
        model=final_model, tfidf=tfidf, scaler=scaler,
        numeric_features=available_features, label_encoder=label_encoder)

    print("\n🧪 Testing production model...")
    test_cases = [
        ("CHICKEN ANGARA GARLIC NAAN restaurant food dining", 200),
        ("electricity bill payment utility service", 500),
        ("uber taxi ride transport vehicle", 150),
        ("amazon shopping electronics purchase", 1000),
        ("Place 2 station to Permanent Residence", 50),
        ("1 month subscription netflix", 199),
        ("mutual fund sip investment", 5000),
        ("salary credited income", 50000),
    ]
    for case, amt in test_cases:
        pred = production_model.predict({'Note': case, 'Amount': amt})
        print(f"  '{case[:45]}' → {pred}")

    print("\n💾 Saving production model...")
    with open("../models/expense_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    with open("../models/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(tfidf, f)
    with open("../models/feature_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open("../models/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

    model_info = {
        "accuracy": float(final_score),
        "model_type": "Production Ensemble (Logistic + RF + GradientBoost + SVM)",
        "categories": label_encoder.classes_.tolist(),
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features_count": int(X_combined.shape[1]),
        "training_date": datetime.now().isoformat()
    }
    with open("../models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print("✅ Production model saved successfully!")
    print(f"\n🎯 FINAL ACCURACY: {final_score:.4f} ({final_score*100:.2f}%)")
    return production_model

if __name__ == "__main__":
    train_production_model()