"""
Specialized Text Analysis Model for Outlier Detection

This model focuses on specific text patterns that distinguish outliers:
1. Specificity markers: Very specific terms that indicate unique subtopics
2. Methodology markers: Terms indicating different research approaches  
3. Terminology diversity: How different the vocabulary is from typical papers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter, defaultdict
import re
from scipy.stats import entropy


class ContentSimilarityModel:
    """
    Specialized text analysis for outlier detection.
    
    Focuses on specific linguistic and methodological patterns rather than
    overall topic similarity, which is more suitable for systematic reviews.
    """
    
    def __init__(self, dataset_name: str = "Appenzeller-Herzog_2019"):
        """Initialize the specialized text analysis model."""
        self.dataset_name = dataset_name
        self.is_fitted = False
        self.simulation_data = None
        
        # Methodology markers - terms that indicate different research approaches
        self.methodology_markers = {
            'case_study': ['case study', 'case report', 'case series', 'case presentation'],
            'longitudinal': ['longitudinal', 'follow-up', 'long-term', 'prospective'],
            'cross_sectional': ['cross-sectional', 'cross sectional', 'prevalence study'],
            'experimental': ['randomized', 'controlled trial', 'rct', 'placebo', 'intervention'],
            'review': ['review', 'meta-analysis', 'systematic review', 'literature review'],
            'diagnostic': ['diagnosis', 'diagnostic', 'screening', 'biomarker'],
            'therapeutic': ['treatment', 'therapy', 'therapeutic', 'drug', 'medication'],
            'genetic': ['genetic', 'mutation', 'gene', 'genomic', 'molecular'],
            'pathological': ['pathology', 'histology', 'autopsy', 'biopsy'],
            'biochemical': ['biochemical', 'enzyme', 'protein', 'metabolic']
        }
        
        # Text features storage
        self.typical_vocabulary = None
        self.rare_terms_threshold = None
        self.methodology_patterns = None
        self.vectorizer = None
        
    def fit(self, simulation_df: pd.DataFrame) -> 'ContentSimilarityModel':
        """
        Fit the specialized text analysis model.
        
        Args:
            simulation_df: DataFrame with simulation results
        
        Returns:
            self: Returns the fitted model
        """
        print("Fitting Specialized Text Analysis Model...")
        
        self.simulation_data = simulation_df.copy()
        
        # Get relevant documents to establish "typical" patterns
        relevant_docs = simulation_df[simulation_df['label_included'] == 1].copy()
        
        if len(relevant_docs) == 0:
            print("Warning: No relevant documents found")
            return self
        
        print(f"Analyzing {len(relevant_docs)} relevant documents to establish baseline patterns...")
        
        # Build typical vocabulary from relevant documents
        self._build_typical_vocabulary(relevant_docs)
        
        # Analyze methodology patterns
        self._analyze_methodology_patterns(simulation_df)
        
        self.is_fitted = True
        print("Specialized text analysis model fitted successfully!")
        
        return self
    
    def _build_typical_vocabulary(self, relevant_docs: pd.DataFrame):
        """Build vocabulary characteristics of typical relevant documents."""
        # Combine all relevant document texts
        texts = []
        for _, row in relevant_docs.iterrows():
            title = str(row.get('title', '') or '')
            abstract = str(row.get('abstract', '') or '')
            combined_text = f"{title} {abstract}".strip()
            if combined_text and len(combined_text) > 10:
                texts.append(self._preprocess_text(combined_text))
        
        if not texts:
            print("Warning: No text found in relevant documents")
            return
        
        # Create TF-IDF vectorizer to find characteristic terms
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.7
        )
        
        # Fit on relevant documents
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate term frequencies and IDF scores
        # High IDF = rare terms, Low IDF = common terms
        idf_scores = self.vectorizer.idf_
        
        # Store typical vocabulary characteristics
        self.typical_vocabulary = {
            'feature_names': feature_names,
            'idf_scores': idf_scores,
            'mean_idf': np.mean(idf_scores),
            'std_idf': np.std(idf_scores)
        }
        
        # Set threshold for "rare" terms (terms that appear in few documents)
        self.rare_terms_threshold = np.percentile(idf_scores, 80)  # Top 20% rarest terms
        
        print(f"Built vocabulary with {len(feature_names)} terms")
        print(f"Rare terms threshold (IDF): {self.rare_terms_threshold:.3f}")
    
    def _analyze_methodology_patterns(self, simulation_df: pd.DataFrame):
        """Analyze methodology marker patterns across all documents."""
        methodology_counts = defaultdict(list)
        
        for _, row in simulation_df.iterrows():
            title = str(row.get('title', '') or '').lower()
            abstract = str(row.get('abstract', '') or '').lower()
            text = f"{title} {abstract}"
            
            # Count methodology markers
            doc_methodology = {}
            for method_type, markers in self.methodology_markers.items():
                count = sum(text.count(marker) for marker in markers)
                doc_methodology[method_type] = count
                methodology_counts[method_type].append(count)
            
        # Calculate typical methodology patterns
        self.methodology_patterns = {}
        for method_type, counts in methodology_counts.items():
            self.methodology_patterns[method_type] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'max': np.max(counts)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-.]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_features(self, target_documents: List[str]) -> pd.DataFrame:
        """
        Extract specialized text features for target documents.
        
        Args:
            target_documents: List of OpenAlex IDs to analyze
        
        Returns:
            DataFrame with specialized text features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
        features = []
        
        for doc_id in target_documents:
            doc_row = self.simulation_data[self.simulation_data['openalex_id'] == doc_id]
            
            if doc_row.empty:
                features.append(self._get_zero_features(doc_id))
                continue
            
            row = doc_row.iloc[0]
            title = str(row.get('title', '') or '')
            abstract = str(row.get('abstract', '') or '')
            doc_text = f"{title} {abstract}".strip()
            
            if not doc_text or len(doc_text) <= 10:
                features.append(self._get_zero_features(doc_id))
                continue
            
            # Extract all feature types
            doc_features = {
                'openalex_id': doc_id,
                **self._extract_specificity_features(doc_text),
                **self._extract_methodology_features(doc_text),
                **self._extract_diversity_features(doc_text)
            }
            
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def _extract_specificity_features(self, text: str) -> Dict[str, float]:
        """Extract features related to term specificity."""
        if self.vectorizer is None:
            return {'specificity_score': 0.0, 'rare_terms_ratio': 0.0, 'unique_terms_count': 0.0}
        
        # Transform text to TF-IDF space
        processed_text = self._preprocess_text(text)
        try:
            doc_vector = self.vectorizer.transform([processed_text])
            doc_array = doc_vector.toarray()[0]
            
            # Calculate specificity score (focus on rare terms)
            idf_scores = self.typical_vocabulary['idf_scores']
            
            # Weight terms by their rarity (high IDF = rare = specific)
            specificity_score = np.sum(doc_array * idf_scores) / (np.sum(doc_array) + 1e-10)
            
            # Count rare terms
            rare_terms_count = np.sum((doc_array > 0) & (idf_scores > self.rare_terms_threshold))
            total_terms = np.sum(doc_array > 0)
            rare_terms_ratio = rare_terms_count / max(total_terms, 1)
            
            return {
                'specificity_score': float(specificity_score),
                'rare_terms_ratio': float(rare_terms_ratio),
                'unique_terms_count': float(rare_terms_count)
            }
            
        except Exception as e:
            print(f"Error extracting specificity features: {e}")
            return {'specificity_score': 0.0, 'rare_terms_ratio': 0.0, 'unique_terms_count': 0.0}
    
    def _extract_methodology_features(self, text: str) -> Dict[str, float]:
        """Extract features related to research methodology markers."""
        text_lower = text.lower()
        
        methodology_scores = {}
        total_methodology_markers = 0
        
        for method_type, markers in self.methodology_markers.items():
            count = sum(text_lower.count(marker) for marker in markers)
            methodology_scores[f'methodology_{method_type}'] = float(count)
            total_methodology_markers += count
        
        # Calculate methodology diversity (how many different methodologies mentioned)
        methodology_diversity = sum(1 for score in methodology_scores.values() if score > 0)
        
        # Calculate dominant methodology (which type has most markers)
        if total_methodology_markers > 0:
            dominant_method = max(methodology_scores.items(), key=lambda x: x[1])
            methodology_focus = dominant_method[1] / total_methodology_markers
        else:
            methodology_focus = 0.0
        
        return {
            **methodology_scores,
            'methodology_diversity': float(methodology_diversity),
            'methodology_focus': float(methodology_focus),
            'total_methodology_markers': float(total_methodology_markers)
        }
    
    def _extract_diversity_features(self, text: str) -> Dict[str, float]:
        """Extract features related to vocabulary diversity."""
        if self.typical_vocabulary is None:
            return {'vocabulary_diversity': 0.0, 'atypical_vocabulary_ratio': 0.0}
        
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        if len(words) == 0:
            return {'vocabulary_diversity': 0.0, 'atypical_vocabulary_ratio': 0.0}
        
        # Calculate vocabulary diversity (unique words / total words)
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words)
        
        # Calculate how much vocabulary differs from typical papers
        typical_terms = set(self.typical_vocabulary['feature_names'])
        
        # Find words not in typical vocabulary
        atypical_words = unique_words - typical_terms
        atypical_vocabulary_ratio = len(atypical_words) / len(unique_words)
        
        return {
            'vocabulary_diversity': float(vocabulary_diversity),
            'atypical_vocabulary_ratio': float(atypical_vocabulary_ratio)
        }
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Generate outlier scores based on text specialization features.
        Fixed scoring logic.
        """
        features_df = self.extract_features(target_documents)
        scores = {}
        
        for _, row in features_df.iterrows():
            # 1. High atypical vocabulary = strong outlier signal
            atypical_score = row['atypical_vocabulary_ratio']  # 0-1, higher = more outlier-like
            
            # 2. Extreme methodology diversity (very low OR very high) = outlier signal
            # Normal papers have moderate diversity, outliers are either very focused or very broad
            diversity = row['methodology_diversity']
            if diversity <= 1.5:  # Very focused
                diversity_score = 0.8  # High outlier score for focused papers
            elif diversity >= 6.0:  # Very broad
                diversity_score = 0.8  # High outlier score for broad papers  
            else:  # Normal range
                diversity_score = 0.2
            
            # 3. Specificity score relative to baseline
            # This needs to be relative to what's "normal" for the dataset
            specificity_baseline = 2.6  # Approximate from your results
            if row['specificity_score'] > specificity_baseline + 0.1:
                specificity_score = 0.7  # High specificity
            elif row['specificity_score'] < specificity_baseline - 0.1:
                specificity_score = 0.3  # Low specificity might also indicate different type
            else:
                specificity_score = 0.1  # Normal specificity
            
            # 4. Strong focus on single methodology can indicate outlier
            focus_score = row['methodology_focus'] if row['total_methodology_markers'] > 0 else 0.0
            
            # Combine scores - weight atypical vocabulary heavily since it's most discriminative
            outlier_score = (
                0.5 * atypical_score +        # Most important: different vocabulary
                0.2 * diversity_score +       # Extreme methodology patterns
                0.2 * specificity_score +     # Unusual specificity
                0.1 * focus_score            # Strong methodological focus
            )
            
            scores[row['openalex_id']] = outlier_score
        
        return scores
    
    def _get_zero_features(self, doc_id: str) -> Dict[str, float]:
        """Return zero features for documents with no text."""
        base_features = {
            'openalex_id': doc_id,
            'specificity_score': 0.0,
            'rare_terms_ratio': 0.0,
            'unique_terms_count': 0.0,
            'methodology_diversity': 0.0,
            'methodology_focus': 0.0,
            'total_methodology_markers': 0.0,
            'vocabulary_diversity': 0.0,
            'atypical_vocabulary_ratio': 0.0
        }
        
        # Add methodology-specific features
        for method_type in self.methodology_markers.keys():
            base_features[f'methodology_{method_type}'] = 0.0
        
        return base_features


def main():
    """Example usage of the specialized text analysis model."""
    # Load simulation data
    simulation_df = pd.read_csv('../data/simulation.csv')
    
    print(f"Loaded {len(simulation_df)} documents")
    print(f"Relevant documents: {simulation_df['label_included'].sum()}")
    
    # Initialize and fit model
    model = ContentSimilarityModel()
    model.fit(simulation_df)
    
    # Get the known outlier
    outlier_row = simulation_df[simulation_df['record_id'] == 497]
    if not outlier_row.empty:
        outlier_id = outlier_row.iloc[0]['openalex_id']
        print(f"\n=== Testing on Known Outlier ===")
        print(f"Outlier ID: {outlier_id}")
        print(f"ASReview Ranking: {outlier_row.iloc[0]['asreview_ranking']}")
        
        # Get outlier score
        outlier_score = model.predict_relevance_scores([outlier_id])[outlier_id]
        print(f"Outlier Score: {outlier_score:.4f}")
    
    # Test on sample of non-relevant documents
    non_relevant = simulation_df[simulation_df['label_included'] == 0]['openalex_id'].head(10).tolist()
    non_relevant_scores = model.predict_relevance_scores(non_relevant)
    
    print(f"\n=== Sample Non-Relevant Document Scores ===")
    sorted_scores = sorted(non_relevant_scores.items(), key=lambda x: x[1], reverse=True)
    for doc_id, score in sorted_scores[:5]:
        print(f"  {doc_id}: {score:.4f}")


if __name__ == "__main__":
    main() 