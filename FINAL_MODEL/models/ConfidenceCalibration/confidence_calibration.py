"""
Confidence Calibration Model for Outlier Detection

This module identifies documents where machine learning models are overconfident
in their predictions, which often indicates potential outliers in systematic reviews.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import warnings

# Machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfidenceCalibrationModel:
    """
    Confidence calibration model for outlier detection.
    
    Identifies documents where ensemble models show high confidence but potentially
    incorrect predictions, which often indicates outliers in systematic reviews.
    """
    
    def __init__(self, 
                 n_estimators: int = 5,
                 random_state: int = 42,
                 calibration_method: str = 'isotonic'):
        """
        Initialize the confidence calibration model.
        
        Args:
            n_estimators: Number of ensemble models to train
            random_state: Random seed for reproducibility
            calibration_method: Method for probability calibration ('isotonic' or 'sigmoid')
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.calibration_method = calibration_method
        
        # Model components
        self.vectorizer = None
        self.ensemble_models = []
        self.calibrated_models = []
        
        # Data and statistics
        self.simulation_data = None
        self.confidence_baselines = None
        self.is_fitted = False
        
        # Feature extraction parameters
        self.tfidf_params = {
            'max_features': 1000,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8,
            'sublinear_tf': True
        }
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    def _load_simulation_data(self, dataset_name: str) -> pd.DataFrame:
        """Load simulation data for the specified dataset."""
        project_root = self._get_project_root()
        simulation_path = os.path.join(project_root, 'data', 'simulations', f'{dataset_name}.csv')
        
        if not os.path.exists(simulation_path):
            raise FileNotFoundError(f"Simulation file not found: {simulation_path}")
        
        return pd.read_csv(simulation_path)
    
    def fit(self, simulation_df: pd.DataFrame, 
            dataset_name: Optional[str] = None) -> 'ConfidenceCalibrationModel':
        """
        Fit the confidence calibration model on simulation data.
        
        Args:
            simulation_df: DataFrame with simulation results
            dataset_name: Optional dataset name for reference
        
        Returns:
            self: Returns the fitted model
        """
        logger.info("Fitting Confidence Calibration Model...")
        
        # Store simulation data
        self.simulation_data = simulation_df.copy()
        
        # Prepare text data and labels
        texts, labels = self._prepare_training_data(simulation_df)
        
        if len(texts) < 20:
            logger.warning("Very few texts available for calibration modeling")
            self._set_default_fitted_state()
            return self
        
        logger.info(f"Training on {len(texts)} documents...")
        
        # Create and fit vectorizer
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train ensemble of diverse models
        self._train_ensemble(X, y)
        
        # Calibrate models for better probability estimates
        self._calibrate_models(X, y)
        
        # Calculate confidence baselines from the data
        self.confidence_baselines = self._calculate_confidence_baselines(X, y)
        
        self.is_fitted = True
        logger.info(f"Confidence calibration model fitted with {len(self.ensemble_models)} models")
        
        return self
    
    def _prepare_training_data(self, simulation_df: pd.DataFrame) -> tuple[List[str], List[int]]:
        """Prepare text data and labels for training."""
        texts = []
        labels = []
        
        for _, row in simulation_df.iterrows():
            # Combine title and abstract
            text_parts = []
            
            if pd.notna(row.get('title')) and str(row.get('title')).strip():
                text_parts.append(str(row['title']).strip())
            
            if pd.notna(row.get('abstract')) and str(row.get('abstract')).strip():
                text_parts.append(str(row['abstract']).strip())
            
            if text_parts:
                combined_text = ' '.join(text_parts)
                texts.append(combined_text)
                labels.append(int(row.get('label_included', 0)))
        
        return texts, labels
    
    def _train_ensemble(self, X, y):
        """Train ensemble of diverse models."""
        logger.info("Training ensemble models...")
        
        np.random.seed(self.random_state)
        
        # Define diverse model configurations
        model_configs = [
            {'n_estimators': 20, 'max_depth': 3, 'min_samples_split': 10},
            {'n_estimators': 30, 'max_depth': 5, 'min_samples_split': 5},
            {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 8},
            {'n_estimators': 35, 'max_depth': 6, 'min_samples_split': 4},
            {'n_estimators': 15, 'max_depth': 7, 'min_samples_split': 12}
        ]
        
        for i in range(self.n_estimators):
            config = model_configs[i % len(model_configs)]
            
            model = RandomForestClassifier(
                random_state=self.random_state + i,
                class_weight='balanced',
                **config
            )
            
            # Bootstrap sampling for diversity
            bootstrap_idx = np.random.choice(len(y), size=int(0.8 * len(y)), replace=True)
            X_bootstrap = X[bootstrap_idx]
            y_bootstrap = y[bootstrap_idx]
            
            model.fit(X_bootstrap, y_bootstrap)
            self.ensemble_models.append(model)
    
    def _calibrate_models(self, X, y):
        """Calibrate model probabilities for better confidence estimates."""
        logger.info("Calibrating model probabilities...")
        
        for model in tqdm(self.ensemble_models, desc="Calibrating models"):
            try:
                calibrated_model = CalibratedClassifierCV(
                    model, 
                    method=self.calibration_method, 
                    cv=3
                )
                calibrated_model.fit(X, y)
                self.calibrated_models.append(calibrated_model)
            except Exception as e:
                logger.warning(f"Calibration failed for a model: {e}")
                # Use uncalibrated model as fallback
                self.calibrated_models.append(model)
    
    def _calculate_confidence_baselines(self, X, y) -> Dict[str, float]:
        """Calculate confidence baselines for adaptive thresholding."""
        logger.info("Calculating confidence baselines...")
        
        if not self.calibrated_models:
            return self._get_default_baselines()
        
        # Get predictions for relevant documents
        relevant_indices = np.where(y == 1)[0]
        irrelevant_indices = np.where(y == 0)[0]
        
        if len(relevant_indices) == 0 or len(irrelevant_indices) == 0:
            return self._get_default_baselines()
        
        # Collect confidence statistics
        relevant_confidences = []
        irrelevant_confidences = []
        confidence_stds = []
        
        for idx in tqdm(range(len(y)), desc="Computing confidence baselines"):
            try:
                ensemble_probs = []
                for model in self.calibrated_models:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X[idx:idx+1])[0]
                        # Confidence in the predicted class
                        confidence = max(prob)
                        ensemble_probs.append(confidence)
                
                if ensemble_probs:
                    mean_conf = np.mean(ensemble_probs)
                    std_conf = np.std(ensemble_probs)
                    
                    if idx in relevant_indices:
                        relevant_confidences.append(mean_conf)
                    else:
                        irrelevant_confidences.append(mean_conf)
                    
                    confidence_stds.append(std_conf)
                    
            except Exception as e:
                logger.debug(f"Failed to compute confidence for sample {idx}: {e}")
                continue
        
        # Calculate statistics
        baselines = {}
        
        if relevant_confidences:
            relevant_confidences = np.array(relevant_confidences)
            baselines.update({
                'relevant_mean_confidence': float(np.mean(relevant_confidences)),
                'relevant_std_confidence': float(np.std(relevant_confidences)),
                'relevant_p75_confidence': float(np.percentile(relevant_confidences, 75)),
                'relevant_p90_confidence': float(np.percentile(relevant_confidences, 90)),
            })
        
        if irrelevant_confidences:
            irrelevant_confidences = np.array(irrelevant_confidences)
            baselines.update({
                'irrelevant_mean_confidence': float(np.mean(irrelevant_confidences)),
                'irrelevant_std_confidence': float(np.std(irrelevant_confidences)),
                'irrelevant_p75_confidence': float(np.percentile(irrelevant_confidences, 75)),
                'irrelevant_p90_confidence': float(np.percentile(irrelevant_confidences, 90)),
            })
        
        if confidence_stds:
            confidence_stds = np.array(confidence_stds)
            baselines.update({
                'mean_confidence_std': float(np.mean(confidence_stds)),
                'p75_confidence_std': float(np.percentile(confidence_stds, 75)),
                'p90_confidence_std': float(np.percentile(confidence_stds, 90)),
            })
        
        # Fill missing values with defaults
        return {**self._get_default_baselines(), **baselines}
    
    def _get_default_baselines(self) -> Dict[str, float]:
        """Get default baseline values when calculation fails."""
        return {
            'relevant_mean_confidence': 0.6,
            'relevant_std_confidence': 0.2,
            'relevant_p75_confidence': 0.7,
            'relevant_p90_confidence': 0.8,
            'irrelevant_mean_confidence': 0.7,
            'irrelevant_std_confidence': 0.2,
            'irrelevant_p75_confidence': 0.8,
            'irrelevant_p90_confidence': 0.9,
            'mean_confidence_std': 0.15,
            'p75_confidence_std': 0.2,
            'p90_confidence_std': 0.25,
        }
    
    def _set_default_fitted_state(self):
        """Set default fitted state when insufficient data."""
        self.is_fitted = True
        self.confidence_baselines = self._get_default_baselines()
        logger.warning("Using default confidence baselines due to insufficient training data")
    
    def extract_features(self, target_documents: List[str]) -> pd.DataFrame:
        """
        Extract confidence calibration features for target documents.
        
        Args:
            target_documents: List of document IDs to extract features for
        
        Returns:
            DataFrame with confidence calibration features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
        logger.info(f"Extracting confidence features for {len(target_documents)} documents")
        
        features = []
        for doc_id in tqdm(target_documents, desc="Extracting confidence features"):
            doc_features = self._extract_single_document_features(doc_id)
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def _extract_single_document_features(self, doc_id: str) -> Dict[str, Any]:
        """Extract confidence features for a single document."""
        if self.simulation_data is None:
            return self._get_zero_features(doc_id)
        
        # Find document in simulation data
        doc_row = self.simulation_data[self.simulation_data['openalex_id'] == doc_id]
        
        if doc_row.empty:
            return self._get_zero_features(doc_id)
        
        row = doc_row.iloc[0]
        
        # Prepare text
        text_parts = []
        if pd.notna(row.get('title')) and str(row.get('title')).strip():
            text_parts.append(str(row['title']).strip())
        
        if pd.notna(row.get('abstract')) and str(row.get('abstract')).strip():
            text_parts.append(str(row['abstract']).strip())
        
        if not text_parts:
            return self._get_zero_features(doc_id)
        
        combined_text = ' '.join(text_parts)
        return self._text_to_confidence_features(doc_id, combined_text)
    
    def _text_to_confidence_features(self, doc_id: str, text: str) -> Dict[str, Any]:
        """Convert text to confidence calibration features."""
        if not self.vectorizer or not self.calibrated_models:
            return self._get_zero_features(doc_id)
        
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Get ensemble predictions
            ensemble_probs = []
            ensemble_predictions = []
            
            for model in self.calibrated_models:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0]
                    pred = model.predict(X)[0]
                    ensemble_probs.append(prob)
                    ensemble_predictions.append(pred)
            
            if not ensemble_probs:
                return self._get_zero_features(doc_id)
            
            # Calculate confidence metrics
            ensemble_probs = np.array(ensemble_probs)
            ensemble_predictions = np.array(ensemble_predictions)
            
            # Confidence in predicted class for each model
            confidences = [max(prob) for prob in ensemble_probs]
            
            # Disagreement metrics
            prediction_variance = np.var(ensemble_predictions)
            confidence_variance = np.var(confidences)
            
            # Average probabilities
            avg_prob_relevant = np.mean([prob[1] if len(prob) > 1 else prob[0] for prob in ensemble_probs])
            avg_prob_irrelevant = np.mean([prob[0] for prob in ensemble_probs])
            avg_confidence = np.mean(confidences)
            
            # Calculate overconfidence indicators
            features = {
                'openalex_id': doc_id,
                'avg_confidence': float(avg_confidence),
                'confidence_std': float(np.std(confidences)),
                'avg_prob_relevant': float(avg_prob_relevant),
                'avg_prob_irrelevant': float(avg_prob_irrelevant),
                'prediction_variance': float(prediction_variance),
                'confidence_variance': float(confidence_variance),
                'ensemble_disagreement': float(np.std(ensemble_predictions)),
            }
            
            # Add overconfidence scores using baselines
            if self.confidence_baselines:
                features.update(self._calculate_overconfidence_scores(features))
            
            return features
            
        except Exception as e:
            logger.debug(f"Failed to extract confidence features for {doc_id}: {e}")
            return self._get_zero_features(doc_id)
    
    def _calculate_overconfidence_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overconfidence scores using baseline statistics."""
        scores = {}
        
        try:
            avg_conf = features['avg_confidence']
            conf_std = features['confidence_std']
            
            # High confidence with low agreement indicates overconfidence
            overconfidence_score = 0.0
            
            # Score 1: High confidence compared to typical irrelevant documents
            if avg_conf > self.confidence_baselines['irrelevant_p75_confidence']:
                excess_confidence = avg_conf - self.confidence_baselines['irrelevant_p75_confidence']
                overconfidence_score += min(1.0, excess_confidence * 2.0)
            
            # Score 2: High variance in confidence indicates uncertainty
            if conf_std > self.confidence_baselines['p75_confidence_std']:
                uncertainty_factor = min(1.0, conf_std / self.confidence_baselines['p90_confidence_std'])
                overconfidence_score += uncertainty_factor * 0.5
            
            # Score 3: Prediction disagreement
            disagreement = features['ensemble_disagreement']
            if disagreement > 0.3:  # High disagreement threshold
                overconfidence_score += min(1.0, disagreement * 1.5)
            
            scores['overconfidence_score'] = float(min(1.0, overconfidence_score))
            
            # Additional calibration metrics
            scores['confidence_excess'] = float(max(0.0, avg_conf - self.confidence_baselines['irrelevant_mean_confidence']))
            scores['uncertainty_indicator'] = float(conf_std / max(0.01, self.confidence_baselines['mean_confidence_std']))
            
        except Exception as e:
            logger.debug(f"Failed to calculate overconfidence scores: {e}")
            scores.update({
                'overconfidence_score': 0.0,
                'confidence_excess': 0.0,
                'uncertainty_indicator': 0.0,
            })
        
        return scores
    
    def _get_zero_features(self, doc_id: str) -> Dict[str, Any]:
        """Get zero features for documents without text or when processing fails."""
        return {
            'openalex_id': doc_id,
            'avg_confidence': 0.5,
            'confidence_std': 0.0,
            'avg_prob_relevant': 0.5,
            'avg_prob_irrelevant': 0.5,
            'prediction_variance': 0.0,
            'confidence_variance': 0.0,
            'ensemble_disagreement': 0.0,
            'overconfidence_score': 0.0,
            'confidence_excess': 0.0,
            'uncertainty_indicator': 0.0,
        }
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Generate confidence calibration-based relevance scores.
        
        Args:
            target_documents: List of document IDs to score
        
        Returns:
            Dictionary mapping document IDs to relevance scores (0-1)
        """
        if not self.is_fitted:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        logger.info(f"Computing confidence-based relevance scores for {len(target_documents)} documents")
        
        # Extract features
        features_df = self.extract_features(target_documents)
        
        # Calculate relevance scores focused on prediction difficulty
        scores = {}
        for _, row in features_df.iterrows():
            doc_id = row['openalex_id']
            
            # Focus on prediction difficulty rather than just overconfidence
            overconfidence = row.get('overconfidence_score', 0.0)
            uncertainty = row.get('uncertainty_indicator', 0.0)
            disagreement = row.get('ensemble_disagreement', 0.0)
            confidence_variance = row.get('confidence_variance', 0.0)
            
            # Rebalanced weighting for outlier detection
            # High disagreement and uncertainty are strong outlier indicators
            disagreement_weight = 0.4  # Increased - ensemble disagreement is key
            uncertainty_weight = 0.3   # Increased - prediction uncertainty matters
            overconfidence_weight = 0.2  # Decreased - less important for outliers
            variance_weight = 0.1      # Prediction variance
            
            # Normalize scores to 0-1 range with better scaling
            disagreement_score = min(1.0, disagreement * 2.5)  # More sensitive
            uncertainty_score = min(1.0, uncertainty / 2.0)    # Less extreme
            variance_score = min(1.0, confidence_variance * 3.0)
            
            final_score = (disagreement_weight * disagreement_score + 
                          uncertainty_weight * uncertainty_score + 
                          overconfidence_weight * overconfidence + 
                          variance_weight * variance_score)
            
            scores[doc_id] = float(max(0.0, min(1.0, final_score)))
        
        return scores
    
    def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Provide detailed confidence analysis for a specific document.
        
        Args:
            doc_id: Document ID to analyze
        
        Returns:
            Dictionary with detailed analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing documents")
        
        features = self._extract_single_document_features(doc_id)
        score = self.predict_relevance_scores([doc_id])[doc_id]
        
        analysis = {
            'document_id': doc_id,
            'relevance_score': score,
            'features': features,
            'model_status': {
                'is_fitted': self.is_fitted,
                'has_vectorizer': self.vectorizer is not None,
                'num_models': len(self.ensemble_models),
                'num_calibrated': len(self.calibrated_models),
            }
        }
        
        # Add interpretation
        if features['overconfidence_score'] > 0.5:
            analysis['interpretation'] = "High overconfidence detected - potential outlier"
        elif features['uncertainty_indicator'] > 1.5:
            analysis['interpretation'] = "High uncertainty in predictions"
        elif features['ensemble_disagreement'] > 0.4:
            analysis['interpretation'] = "High model disagreement"
        else:
            analysis['interpretation'] = "Normal confidence pattern"
        
        return analysis


def main():
    """Example usage of the ConfidenceCalibrationModel."""
    import sys
    import os
    
    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, project_root)
    
    try:
        # Load available datasets
        datasets_config_path = os.path.join(project_root, 'data', 'datasets.json')
        with open(datasets_config_path, 'r') as f:
            datasets_config = json.load(f)
        
        dataset_names = list(datasets_config.keys())
        
        print("Available datasets:")
        for i, name in enumerate(dataset_names, 1):
            print(f"{i}. {name}")
        
        # Select dataset
        choice = int(input("\nSelect dataset (enter number): ")) - 1
        if 0 <= choice < len(dataset_names):
            dataset_name = dataset_names[choice]
            
            # Load simulation data
            simulation_path = os.path.join(project_root, 'data', 'simulations', f'{dataset_name}.csv')
            simulation_df = pd.read_csv(simulation_path)
            
            # Initialize and fit model
            print(f"\nFitting confidence calibration model on {dataset_name}...")
            model = ConfidenceCalibrationModel()
            model.fit(simulation_df, dataset_name)
            
            # Get example documents
            example_docs = simulation_df['openalex_id'].head(10).tolist()
            
            # Extract features
            print("\nExtracting confidence features...")
            features_df = model.extract_features(example_docs)
            print(features_df[['openalex_id', 'avg_confidence', 'overconfidence_score']].head())
            
            # Compute scores
            print("\nComputing relevance scores...")
            scores = model.predict_relevance_scores(example_docs)
            for doc_id, score in list(scores.items())[:5]:
                print(f"{doc_id}: {score:.4f}")
                
        else:
            print("Invalid selection")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 