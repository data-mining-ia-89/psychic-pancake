# app/models/llm_finetuning.py

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class LLMFineTuner:
    def __init__(self, model_name="distilbert-base-uncased", task="sentiment"):
        self.model_name = model_name
        self.task = task
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model(self, num_labels=3):
        """Charger le modèle pré-entraîné"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
        # Ajouter un token de padding si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_data(self, texts, labels):
        """Préparer les données pour l'entraînement"""
        
        # Tokenisation
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Créer dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return dataset
    
    def train_sentiment_model(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Fine-tuning pour analyse de sentiment"""
        
        # Préparer les données
        train_dataset = self.prepare_data(train_texts, train_labels)
        
        if val_texts is not None:
            val_dataset = self.prepare_data(val_texts, val_labels)
        else:
            # Split automatique si pas de validation fournie
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42
            )
            train_dataset = self.prepare_data(train_texts, train_labels)
            val_dataset = self.prepare_data(val_texts, val_labels)
        
        # Configuration de l'entraînement
        training_args = TrainingArguments(
            output_dir=f'./models/{self.task}_model',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs/{self.task}',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True
        )
        
        # Métriques d'évaluation
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }
        
        # Créer le trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )
        
        # Lancer l'entraînement
        print("🚀 Début du fine-tuning...")
        self.trainer.train()
        
        # Sauvegarder le modèle
        self.trainer.save_model()
        self.tokenizer.save_pretrained(f'./models/{self.task}_model')
        
        print("✅ Fine-tuning terminé!")
        
        return self.trainer.evaluate()
    
    def predict(self, texts):
        """Prédiction avec le modèle fine-tuné"""
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        return predicted_labels.numpy(), predictions.numpy()
    
    def load_fine_tuned_model(self, model_path):
        """Charger un modèle déjà fine-tuné"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Exemple d'utilisation pour fine-tuning sentiment
def create_sentiment_dataset():
    """Créer un dataset d'exemple pour sentiment"""
    
    # Données d'exemple (à remplacer par vraies données Hadoop)
    positive_texts = [
        "This product is amazing, I love it!",
        "Great service, highly recommended",
        "Excellent quality and fast delivery",
        "Best purchase I've made this year",
        "Outstanding performance, very satisfied"
    ]
    
    negative_texts = [
        "Terrible product, waste of money",
        "Poor quality, broke after one day",
        "Awful customer service experience",
        "Not worth the price at all",
        "Disappointed with this purchase"
    ]
    
    neutral_texts = [
        "The product is okay, nothing special",
        "Standard quality for the price",
        "It works as expected",
        "Average performance",
        "Decent but could be better"
    ]
    
    # Combiner les données
    all_texts = positive_texts + negative_texts + neutral_texts
    all_labels = [2] * len(positive_texts) + [0] * len(negative_texts) + [1] * len(neutral_texts)
    
    return all_texts, all_labels

# Script principal pour fine-tuning
def main():
    # Créer les données d'entraînement
    texts, labels = create_sentiment_dataset()
    
    # Initialiser le fine-tuner
    fine_tuner = LLMFineTuner(task="sentiment")
    fine_tuner.load_model(num_labels=3)  # positive, negative, neutral
    
    # Lancer le fine-tuning
    results = fine_tuner.train_sentiment_model(texts, labels)
    print(f"Résultats finaux: {results}")
    
    # Test du modèle
    test_texts = ["This is great!", "This is terrible", "This is okay"]
    predictions, probabilities = fine_tuner.predict(test_texts)
    
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    for i, text in enumerate(test_texts):
        pred_label = label_map[predictions[i]]
        confidence = max(probabilities[i])
        print(f"Text: '{text}' -> {pred_label} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()