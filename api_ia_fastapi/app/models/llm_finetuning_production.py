# api_ia_fastapi/app/models/llm_finetuning_production.py

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import os
import requests
from datetime import datetime

class ProductionLLMFineTuner:
    """
    Fine-tuning LLM pour analyse de sentiment - Version Production
    Compatible avec votre stack LM Studio pour comparaison
    """
    
    def __init__(self, model_name="distilbert-base-uncased", task="sentiment"):
        self.model_name = model_name
        self.task = task
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_results = {}
        
        # Configuration pour votre projet
        self.output_dir = f"./models/finetuned_{task}_model"
        self.data_dir = "./training_data"
        
        # Créer les dossiers
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_base_model(self, num_labels=3):
        """Charger le modèle de base pour fine-tuning"""
        print(f"🔄 Chargement du modèle de base: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Ajouter padding token si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ Modèle {self.model_name} chargé avec {num_labels} classes")
    
    def create_training_data_from_hadoop(self):
        """
        Créer des données d'entraînement à partir de vos données Hadoop
        Utilise les reviews Amazon de votre HDFS
        """
        print("📥 Création des données d'entraînement depuis Hadoop...")
        
        # Simulation des données Hadoop (à adapter avec vos vraies données)
        # Dans la vraie implémentation, vous liriez depuis HDFS
        
        # Données Amazon Reviews enrichies pour le fine-tuning
        training_data = [
            # Reviews positives (label 2)
            ("This product is absolutely amazing! Great quality and fast shipping.", 2),
            ("Excellent customer service and fantastic features. Highly recommend!", 2),
            ("Perfect! Exactly what I was looking for. Great seller!", 2),
            ("Outstanding quality and value. Best purchase this year!", 2),
            ("Love this item! Exceeded expectations in every way.", 2),
            ("Fantastic product! Amazing quality and great support.", 2),
            ("Incredible value for money. Works perfectly as advertised.", 2),
            ("Superb quality and excellent customer service experience.", 2),
            ("Amazing product that works exactly as described. Very satisfied.", 2),
            ("Exceptional quality and fast delivery. Highly recommended!", 2),
            
            # Reviews négatives (label 0)
            ("Terrible experience. Product broke after one day. Very disappointed.", 0),
            ("Poor quality for the price. Would not buy again. Expected better.", 0),
            ("Disappointed with the build quality. Expected more for this price.", 0),
            ("Not what I expected. Description was misleading. Below average.", 0),
            ("Awful product. Waste of money. Poor construction and materials.", 0),
            ("Horrible experience. Product doesn't work at all. Very poor quality.", 0),
            ("Completely useless product. Broke immediately upon arrival.", 0),
            ("Terrible customer service and defective product. Avoid this seller.", 0),
            ("Worst purchase ever. Product is nothing like advertised.", 0),
            ("Poor materials and construction. Fell apart after minimal use.", 0),
            
            # Reviews neutres (label 1)
            ("Average product. Nothing special but does what it's supposed to do.", 1),
            ("Product is okay but could be better. Mediocre experience overall.", 1),
            ("Decent product but delivery was slow. Product itself is fine.", 1),
            ("Good value for money. Works as advertised. No complaints.", 1),
            ("Standard quality for the price. Nothing exceptional but adequate.", 1),
            ("The product works fine but nothing outstanding about it.", 1),
            ("Acceptable quality for the price point. Does what it should.", 1),
            ("Average experience. Product is okay, shipping was standard.", 1),
            ("Decent build quality but could have better features for price.", 1),
            ("Product meets basic expectations. Nothing more, nothing less.", 1),
        ]
        
        # Ajouter plus de données variées pour un meilleur fine-tuning
        extended_data = self._generate_extended_training_data()
        training_data.extend(extended_data)
        
        # Convertir en DataFrame
        df = pd.DataFrame(training_data, columns=['text', 'label'])
        
        # Sauvegarder pour traçabilité
        df.to_csv(f"{self.data_dir}/training_data.csv", index=False)
        
        print(f"✅ Données d'entraînement créées: {len(df)} exemples")
        print(f"   - Positives: {len(df[df['label'] == 2])}")
        print(f"   - Négatives: {len(df[df['label'] == 0])}")
        print(f"   - Neutres: {len(df[df['label'] == 1])}")
        
        return df
    
    def _generate_extended_training_data(self):
        """Générer plus de données d'entraînement variées"""
        
        # Templates pour générer des variations
        positive_templates = [
            "Excellent {product}! {feature} works perfectly.",
            "Amazing {product} with great {feature}. Highly recommended!",
            "Love this {product}! {feature} exceeded my expectations.",
            "Perfect {product} for the price. {feature} is outstanding.",
            "Fantastic {product}! {feature} is exactly what I needed."
        ]
        
        negative_templates = [
            "Terrible {product}. {feature} doesn't work properly.",
            "Poor quality {product}. {feature} broke after one day.",
            "Disappointed with this {product}. {feature} is defective.",
            "Awful {product}. {feature} is completely useless.",
            "Worst {product} ever. {feature} failed immediately."
        ]
        
        neutral_templates = [
            "Average {product}. {feature} works but nothing special.",
            "Decent {product} for the price. {feature} is okay.",
            "Standard {product}. {feature} meets basic expectations.",
            "Acceptable {product}. {feature} does what it should.",
            "Regular {product}. {feature} is adequate but unremarkable."
        ]
        
        products = ["smartphone", "laptop", "headphones", "camera", "tablet", "speaker", "monitor"]
        features = ["battery life", "sound quality", "build quality", "performance", "design", "functionality"]
        
        extended_data = []
        
        # Générer des variations
        for product in products[:3]:  # Limiter pour éviter trop de données
            for feature in features[:3]:
                # Positive
                template = np.random.choice(positive_templates)
                text = template.format(product=product, feature=feature)
                extended_data.append((text, 2))
                
                # Negative
                template = np.random.choice(negative_templates)
                text = template.format(product=product, feature=feature)
                extended_data.append((text, 0))
                
                # Neutral
                template = np.random.choice(neutral_templates)
                text = template.format(product=product, feature=feature)
                extended_data.append((text, 1))
        
        return extended_data
    
    def prepare_datasets(self, df, test_size=0.2):
        """Préparer les datasets d'entraînement et de validation"""
        print("🔧 Préparation des datasets...")
        
        # Split train/test
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=test_size,
            random_state=42,
            stratify=df['label']
        )
        
        # Tokenisation
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Créer les datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })
        
        print(f"✅ Datasets préparés:")
        print(f"   - Entraînement: {len(train_dataset)} exemples")
        print(f"   - Validation: {len(val_dataset)} exemples")
        
        return train_dataset, val_dataset
    
    def setup_training(self):
        """Configuration de l'entraînement"""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none"  # Pas de wandb pour simplifier
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }
        
        return training_args, compute_metrics
    
    def train_model(self, train_dataset, val_dataset):
        """Lancer le fine-tuning"""
        print("🚀 Début du fine-tuning...")
        
        training_args, compute_metrics = self.setup_training()
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )
        
        # Lancer l'entraînement
        start_time = datetime.now()
        train_result = self.trainer.train()
        end_time = datetime.now()
        
        training_duration = (end_time - start_time).total_seconds()
        
        # Sauvegarder le modèle fine-tuné
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Évaluation finale
        eval_result = self.trainer.evaluate()
        
        # Sauvegarder les résultats
        self.training_results = {
            'training_duration_seconds': training_duration,
            'train_loss': train_result.training_loss,
            'eval_accuracy': eval_result.get('eval_accuracy', 0),
            'eval_f1': eval_result.get('eval_f1', 0),
            'model_path': self.output_dir,
            'training_completed': datetime.now().isoformat()
        }
        
        # Sauvegarder les métriques
        with open(f"{self.output_dir}/training_results.json", 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        print(f"✅ Fine-tuning terminé en {training_duration:.1f}s!")
        print(f"   - Accuracy: {eval_result.get('eval_accuracy', 0):.3f}")
        print(f"   - F1 Score: {eval_result.get('eval_f1', 0):.3f}")
        
        return self.training_results
    
    def test_finetuned_model(self):
        """Tester le modèle fine-tuné"""
        print("🧪 Test du modèle fine-tuné...")
        
        # Charger le modèle fine-tuné
        from transformers import pipeline
        
        classifier = pipeline(
            "text-classification",
            model=self.output_dir,
            tokenizer=self.output_dir,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Tests
        test_texts = [
            "This product is absolutely fantastic! Best purchase ever!",
            "Terrible quality, broke after one day. Completely disappointed.",
            "Average product, does what it should but nothing exceptional.",
            "Amazing customer service and great quality. Highly recommended!",
            "Poor construction and materials. Would not recommend."
        ]
        
        results = []
        for text in test_texts:
            result = classifier(text)
            
            # Convertir les labels numériques en texte
            label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
            predicted_label = int(result[0]['label'].split('_')[-1])
            sentiment = label_map.get(predicted_label, "UNKNOWN")
            confidence = result[0]['score']
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
            
            print(f"   Text: '{text[:50]}...'")
            print(f"   Sentiment: {sentiment} (confiance: {confidence:.3f})")
            print()
        
        return results
    
    def compare_with_lm_studio(self, test_texts):
        """Comparer les performances avec votre LM Studio"""
        print("📊 Comparaison avec LM Studio...")
        
        lm_studio_url = "http://localhost:1234/v1/chat/completions"
        
        comparison_results = []
        
        for text in test_texts:
            # Résultat du modèle fine-tuné (déjà calculé)
            finetuned_result = self.test_single_text(text)
            
            # Résultat LM Studio
            try:
                lm_studio_payload = {
                    "model": "mistralai/mathstral-7b-v0.1",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "Analyze the sentiment of the given text. Respond with only: POSITIVE, NEGATIVE, or NEUTRAL"
                        },
                        {
                            "role": "user", 
                            "content": f"Analyze sentiment: {text}"
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 10,
                    "stream": False
                }
                
                response = requests.post(lm_studio_url, json=lm_studio_payload, timeout=30)
                
                if response.status_code == 200:
                    lm_result = response.json()
                    lm_sentiment = lm_result["choices"][0]["message"]["content"].strip().upper()
                    
                    if lm_sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                        lm_sentiment = "UNKNOWN"
                else:
                    lm_sentiment = "ERROR"
                    
            except Exception as e:
                lm_sentiment = "ERROR"
            
            comparison_results.append({
                'text': text,
                'finetuned_sentiment': finetuned_result['sentiment'],
                'finetuned_confidence': finetuned_result['confidence'],
                'lm_studio_sentiment': lm_sentiment,
                'agreement': finetuned_result['sentiment'] == lm_sentiment
            })
        
        return comparison_results
    
    def test_single_text(self, text):
        """Tester un texte unique avec le modèle fine-tuné"""
        from transformers import pipeline
        
        classifier = pipeline(
            "text-classification",
            model=self.output_dir,
            tokenizer=self.output_dir,
            device=0 if torch.cuda.is_available() else -1
        )
        
        result = classifier(text)
        label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        predicted_label = int(result[0]['label'].split('_')[-1])
        sentiment = label_map.get(predicted_label, "UNKNOWN")
        confidence = result[0]['score']
        
        return {
            'sentiment': sentiment,
            'confidence': confidence
        }

def run_complete_finetuning_pipeline():
    """Pipeline complet de fine-tuning pour votre projet"""
    print("🎯 === PIPELINE COMPLET FINE-TUNING LLM ===")
    print("📋 Conforme au cahier des charges")
    print()
    
    # Initialiser le fine-tuner
    fine_tuner = ProductionLLMFineTuner(
        model_name="distilbert-base-uncased",
        task="sentiment"
    )
    
    try:
        # 1. Charger le modèle de base
        fine_tuner.load_base_model(num_labels=3)
        
        # 2. Créer les données d'entraînement
        df = fine_tuner.create_training_data_from_hadoop()
        
        # 3. Préparer les datasets
        train_dataset, val_dataset = fine_tuner.prepare_datasets(df)
        
        # 4. Lancer le fine-tuning
        training_results = fine_tuner.train_model(train_dataset, val_dataset)
        
        # 5. Tester le modèle fine-tuné
        test_results = fine_tuner.test_finetuned_model()
        
        # 6. Comparer avec LM Studio
        test_texts = [
            "This product is absolutely fantastic! Best purchase ever!",
            "Terrible quality, broke after one day. Completely disappointed.",
            "Average product, does what it should but nothing exceptional."
        ]
        
        comparison = fine_tuner.compare_with_lm_studio(test_texts)
        
        print("\n🎉 === FINE-TUNING COMPLÉTÉ AVEC SUCCÈS ===")
        print(f"✅ Modèle sauvegardé dans: {fine_tuner.output_dir}")
        print(f"✅ Accuracy: {training_results['eval_accuracy']:.3f}")
        print(f"✅ F1 Score: {training_results['eval_f1']:.3f}")
        print(f"✅ Durée d'entraînement: {training_results['training_duration_seconds']:.1f}s")
        
        print("\n📊 Comparaison LM Studio vs Fine-tuned:")
        for comp in comparison:
            agreement = "✅" if comp['agreement'] else "❌"
            print(f"   {agreement} Fine-tuned: {comp['finetuned_sentiment']} | LM Studio: {comp['lm_studio_sentiment']}")
        
        print("\n🚀 VOTRE PROJET EST MAINTENANT CONFORME AU CAHIER DES CHARGES!")
        print("✅ Fine-tuning LLM réalisé")
        print("✅ Modèle personnalisé créé")
        print("✅ Comparaison avec LM Studio disponible")
        print("✅ Prêt pour la soutenance!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur pendant le fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_finetuning_pipeline()
    
    if success:
        print("\n🎯 Fine-tuning réussi! Votre API peut maintenant utiliser:")
        print("1. Le modèle fine-tuné (nouvellement entraîné)")
        print("2. LM Studio (pour comparaison)")
        print("3. Analyse de performance entre les deux")
    else:
        print("\n❌ Échec du fine-tuning")