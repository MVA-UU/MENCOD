"""
Confidence Calibration Model for Hybrid Outlier Detection

This module provides confidence calibration features for identifying outlier documents
where models are overconfident in their predictions but potentially wrong.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class ConfidenceCalibrationModel:
    """Confidence calibration-based feature extractor for outlier detection."""
    
    def __init__(self, n_estimators: int = 3, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.vectorizer = None
        self.ensemble_models = []
        self.simulation_data = None
        self.confidence_baselines = None
        self.is_fitted = False
        
    def fit(self, simulation_df: pd.DataFrame) -> 'ConfidenceCalibrationModel':
        """Fit ensemble models on simulation data."""
        print("Fitting Confidence Calibration Model...")
        
        # Store simulation data for later access
        self.simulation_data = simulation_df.copy()
        
        # Prepare text data
        texts = []
        labels = []
        
        for _, row in simulation_df.iterrows():
            if pd.notna(row.get('title')) and pd.notna(row.get('abstract')):
                combined_text = f"{row['title']} {row['abstract']}"
                texts.append(combined_text)
                labels.append(row['label_included'])
        
        if len(texts) < 10:
            print("Warning: Very few texts available for calibration modeling")
            self.is_fitted = True
            return self
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        print(f"Training ensemble on {len(texts)} documents...")
        
        # Train simple ensemble for overconfidence detection
        np.random.seed(self.random_state)
        
        for i in range(self.n_estimators):
            model = RandomForestClassifier(
                n_estimators=15,
                max_depth=np.random.choice([3, 5]),
                random_state=self.random_state + i,
                class_weight='balanced'
            )
            
            # Bootstrap sampling
            bootstrap_idx = np.random.choice(len(texts), size=int(0.8 * len(texts)), replace=True)
            model.fit(X[bootstrap_idx], y[bootstrap_idx])
            self.ensemble_models.append(model)
        
        # Calculate confidence baselines from relevant documents
        self.confidence_baselines = self._calculate_confidence_baselines(simulation_df)
        
        self.is_fitted = True
        print(f"Calibration model fitted with {len(self.ensemble_models)} ensemble members")
        print(f"Confidence baselines calculated from relevant documents")
        return self
    
    def _calculate_confidence_baselines(self, simulation_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate confidence statistics from relevant documents for dynamic thresholds."""
        relevant_docs = simulation_df[simulation_df['label_included'] == 1]
        
        if relevant_docs.empty:
            return {
                'mean_confidence_irrelevant': 0.5,
                'std_confidence_irrelevant': 0.2,
                'p75_confidence_irrelevant': 0.7,
                'p90_confidence_irrelevant': 0.8,
                'p95_confidence_irrelevant': 0.85,
                'mean_confidence_std': 0.1,
                'p75_confidence_std': 0.15,
                'p90_confidence_std': 0.2
            }
        
        confidences_irrelevant = []
        confidence_stds = []
        
        for _, row in relevant_docs.iterrows():
            if pd.notna(row.get('title')) and pd.notna(row.get('abstract')):
                combined_text = f"{row['title']} {row['abstract']}"
                try:
                    X = self.vectorizer.transform([combined_text])
                    ensemble_confidences = []
                    
                    for model in self.ensemble_models:
                        try:
                            prob = model.predict_proba(X)[0]
                            confidence_irrelevant = prob[0]
                            ensemble_confidences.append(confidence_irrelevant)
                        except:
                            continue
                    
                    if ensemble_confidences:
                        mean_conf = np.mean(ensemble_confidences)
                        std_conf = np.std(ensemble_confidences)
                        confidences_irrelevant.append(mean_conf)
                        confidence_stds.append(std_conf)
                        
                except:
                    continue
        
        if not confidences_irrelevant:
            return {
                'mean_confidence_irrelevant': 0.5,
                'std_confidence_irrelevant': 0.2,
                'p75_confidence_irrelevant': 0.7,
                'p90_confidence_irrelevant': 0.8,
                'p95_confidence_irrelevant': 0.85,
                'mean_confidence_std': 0.1,
                'p75_confidence_std': 0.15,
                'p90_confidence_std': 0.2
            }
        
        confidences_irrelevant = np.array(confidences_irrelevant)
        confidence_stds = np.array(confidence_stds)
        
        return {
            'mean_confidence_irrelevant': float(np.mean(confidences_irrelevant)),
            'std_confidence_irrelevant': max(float(np.std(confidences_irrelevant)), 0.01),
            'p75_confidence_irrelevant': float(np.percentile(confidences_irrelevant, 75)),
            'p90_confidence_irrelevant': float(np.percentile(confidences_irrelevant, 90)),
            'p95_confidence_irrelevant': float(np.percentile(confidences_irrelevant, 95)),
            'mean_confidence_std': float(np.mean(confidence_stds)),
            'p75_confidence_std': float(np.percentile(confidence_stds, 75)),
            'p90_confidence_std': float(np.percentile(confidence_stds, 90))
        }
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """Generate confidence calibration-based outlier scores using stored simulation data."""
        if not self.is_fitted or not self.ensemble_models or self.simulation_data is None:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        # Get titles and abstracts from stored simulation data
        titles = []
        abstracts = []
        
        for doc_id in target_documents:
            doc_row = self.simulation_data[self.simulation_data['openalex_id'] == doc_id]
            if not doc_row.empty:
                titles.append(doc_row.iloc[0].get('title', ''))
                abstracts.append(doc_row.iloc[0].get('abstract', ''))
            else:
                titles.append('')
                abstracts.append('')
        
        return self.predict_calibration_scores(target_documents, titles, abstracts)
    
    def predict_calibration_scores(self, target_documents: List[str], 
                                 titles: List[str], abstracts: List[str]) -> Dict[str, float]:
        """Generate calibration-based outlier scores given titles and abstracts."""
        if not self.is_fitted or not self.ensemble_models or not self.confidence_baselines:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            title = titles[i] if i < len(titles) else ''
            abstract = abstracts[i] if i < len(abstracts) else ''
            
            # Create combined text from available parts
            text_parts = []
            if pd.notna(title) and title.strip():
                text_parts.append(title.strip())
            if pd.notna(abstract) and abstract.strip():
                text_parts.append(abstract.strip())
            
            if text_parts:
                combined_text = " ".join(text_parts)
                calibration_score = self._text_to_calibration_score(combined_text)
                scores[doc_id] = calibration_score
            else:
                scores[doc_id] = self._get_no_text_calibration_score()
        
        return scores
    
    def _text_to_calibration_score(self, text: str) -> float:
        """Convert text to confidence calibration score using dynamic baselines."""
        try:
            X = self.vectorizer.transform([text])
            
            # Get predictions from ensemble
            confidences = []
            predictions = []
            
            for model in self.ensemble_models:
                try:
                    prob = model.predict_proba(X)[0]
                    pred = model.predict(X)[0]
                    confidence_irrelevant = prob[0]
                    confidences.append(confidence_irrelevant)
                    predictions.append(pred)
                except:
                    continue
            
            if not confidences:
                return 0.0
            
            # Calculate overconfidence score using dynamic baselines
            mean_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            baselines = self.confidence_baselines
            overconfidence_score = 0.0
            
            # Score based on confidence deviation (much more conservative)
            confidence_deviation = (mean_confidence - baselines['mean_confidence_irrelevant']) / baselines['std_confidence_irrelevant']
            
            # Much more conservative deviation scoring
            if confidence_deviation > 0:
                # Use sigmoid for very gradual increase, capped lower
                dev_score = min(0.15, 0.15 * (1 / (1 + np.exp(-confidence_deviation + 2))))
                overconfidence_score += dev_score
            
            # Much more stringent percentile scoring
            if mean_confidence > baselines['p95_confidence_irrelevant']:
                excess = mean_confidence - baselines['p95_confidence_irrelevant']
                # Very small bonus for exceeding 95th percentile
                p95_score = min(0.1, excess * 0.8)
                overconfidence_score += p95_score
            elif mean_confidence > baselines['p90_confidence_irrelevant']:
                excess = mean_confidence - baselines['p90_confidence_irrelevant']
                range_90_95 = baselines['p95_confidence_irrelevant'] - baselines['p90_confidence_irrelevant']
                p90_score = min(0.08, (excess / max(range_90_95, 0.01)) * 0.08)
                overconfidence_score += p90_score
            elif mean_confidence > baselines['p75_confidence_irrelevant']:
                excess = mean_confidence - baselines['p75_confidence_irrelevant']
                range_75_90 = baselines['p90_confidence_irrelevant'] - baselines['p75_confidence_irrelevant']
                p75_score = min(0.05, (excess / max(range_75_90, 0.01)) * 0.05)
                overconfidence_score += p75_score
            
            # Much more conservative agreement scoring
            expected_std = baselines['mean_confidence_std']
            if confidence_std < expected_std * 0.5:  # Only very low disagreement counts
                std_ratio = confidence_std / max(expected_std, 0.01)
                agreement_score = min(0.08, 0.08 * (1.0 - std_ratio) ** 2)  # Quadratic for more selectivity
                overconfidence_score += agreement_score
            
            # Very small consensus bonus, only for extreme cases
            if all(pred == 0 for pred in predictions) and len(predictions) >= 3:
                if mean_confidence > baselines['p95_confidence_irrelevant']:
                    consensus_score = min(0.05, (mean_confidence - baselines['p95_confidence_irrelevant']) * 0.3)
                    overconfidence_score += consensus_score
            
            # Much lower maximum to prevent clustering
            return min(0.4, max(0.0, overconfidence_score))
            
        except Exception as e:
            return 0.0
    
    def _get_no_text_calibration_score(self) -> float:
        """Calculate calibration score for documents without text content."""
        # Documents without text content can't be analyzed for overconfidence
        # Give them a low-moderate score since we can't assess them properly
        return 0.1


def main():
    """Test confidence calibration model standalone and show outlier ranking."""
    print("=== CONFIDENCE CALIBRATION MODEL STANDALONE TEST ===")
    
    # Load and prepare data
    sim_df = pd.read_csv('../data/simulation.csv')
    train_data = sim_df.copy()
    train_data['label_included'] = train_data['asreview_ranking'].apply(lambda x: 1 if x <= 25 else 0)
    
    print(f"Training with {train_data['label_included'].sum()} relevant documents (ranks 1-25)")
    
    # Get outlier info
    outlier_row = sim_df[sim_df['record_id'] == 497]
    outlier_id = outlier_row.iloc[0]['openalex_id']
    outlier_rank = outlier_row.iloc[0]['asreview_ranking']
    print(f"Target outlier: record_id=497, ASReview rank={outlier_rank}, ID: {outlier_id}")
    
    # Fit model and test
    model = ConfidenceCalibrationModel()
    model.fit(train_data)
    
    # Show baseline statistics
    print(f"\n=== CONFIDENCE BASELINES ===")
    for k, v in model.confidence_baselines.items():
        print(f"  {k}: {v:.4f}")
    
    # Debug: Test predictions for outlier
    print(f"\n=== ENSEMBLE ANALYSIS FOR OUTLIER ===")
    outlier_title = outlier_row.iloc[0].get('title', '')
    if outlier_title and pd.notna(outlier_title):
        try:
            X_outlier = model.vectorizer.transform([outlier_title])
            confidences = []
            for i, ensemble_model in enumerate(model.ensemble_models):
                try:
                    prob = ensemble_model.predict_proba(X_outlier)[0]
                    pred = ensemble_model.predict(X_outlier)[0]
                    confidences.append(prob[0])
                    print(f"Model {i+1}: Prediction={pred}, Confidence in irrelevance={prob[0]:.3f}")
                except:
                    print(f"Model {i+1}: Failed")
            
            if confidences:
                mean_conf = np.mean(confidences)
                std_conf = np.std(confidences)
                print(f"Mean confidence in irrelevance: {mean_conf:.3f}")
                print(f"Confidence std: {std_conf:.4f}")
                
                # Show how this compares to baselines
                conf_dev = (mean_conf - model.confidence_baselines['mean_confidence_irrelevant']) / model.confidence_baselines['std_confidence_irrelevant']
                print(f"Confidence deviation from relevant docs: {conf_dev:.2f} standard deviations")
                
        except Exception as e:
            print(f"Error in analysis: {e}")
    
    # Test candidates (outlier + irrelevant documents)
    irrelevant_docs = sim_df[sim_df['label_included'] == 0]['openalex_id'].tolist()
    test_candidates = [outlier_id] + irrelevant_docs
    
    print(f"\nTesting {len(test_candidates)} candidates...")
    
    # Get calibration scores
    scores = model.predict_relevance_scores(test_candidates)
    outlier_score = scores.get(outlier_id, 0.0)
    
    # Ranking results
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    outlier_pos = next((i for i, (doc_id, _) in enumerate(sorted_cands, 1) if doc_id == outlier_id), None)
    
    print(f"\n=== RESULTS ===")
    print(f"Outlier calibration score: {outlier_score:.4f}")
    print(f"Outlier position: {outlier_pos}/{len(test_candidates)}")
    percentile = ((len(test_candidates) - outlier_pos) / len(test_candidates)) * 100
    print(f"Percentile: {percentile:.1f}th")
    
    # Top 10 scores
    print(f"\nTop 10 most overconfident documents:")
    for i, (doc_id, score) in enumerate(sorted_cands[:10], 1):
        marker = " *** OUTLIER ***" if doc_id == outlier_id else ""
        print(f"  {i:2d}. Score: {score:.4f}{marker}")
    
    # Performance assessment
    assessments = [(10, "🟢 EXCELLENT"), (50, "🟡 GOOD"), (100, "🟠 FAIR")]
    assessment = next((msg for threshold, msg in assessments if outlier_pos <= threshold), "🔴 POOR")
    print(f"\nPerformance: {assessment}")
    
    # Score statistics
    all_scores = list(scores.values())
    print(f"\nScore stats: Mean={np.mean(all_scores):.4f}, Std={np.std(all_scores):.4f}, "
          f"Min={np.min(all_scores):.4f}, Max={np.max(all_scores):.4f}")
    
    return {
        'outlier_position': outlier_pos, 'outlier_score': outlier_score,
        'total_candidates': len(test_candidates), 'percentile': percentile
    }


if __name__ == "__main__":
    main() 