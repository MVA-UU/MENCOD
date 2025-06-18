"""
Content Similarity Model for Outlier Detection

This module analyzes text content patterns to identify outlier documents that differ
in specificity, methodology, or terminology from typical relevant documents.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import re
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
from tqdm import tqdm

# Text processing libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentSimilarityModel:
    """
    Content similarity model for outlier detection.
    
    Analyzes text patterns including specificity markers, methodology indicators,
    and terminology diversity to identify documents that differ from typical
    relevant documents in systematic reviews.
    """
    
    def __init__(self, enable_semantic_embeddings: bool = True):
        """
        Initialize the content similarity model.
        
        Args:
            enable_semantic_embeddings: Whether to include semantic embedding features
        """
        self.enable_semantic_embeddings = enable_semantic_embeddings
        
        # Model components
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.embeddings = None
        self.embeddings_metadata = None
        
        # Text analysis components
        self.typical_vocabulary = None
        self.methodology_patterns = None
        self.specificity_thresholds = None
        
        # Data and state
        self.simulation_data = None
        self.is_fitted = False
        
        # Methodology markers for different research approaches
        self.methodology_markers = {
            'case_study': ['case study', 'case report', 'case series', 'case presentation', 'case analysis'],
            'longitudinal': ['longitudinal', 'follow-up', 'long-term', 'prospective', 'cohort'],
            'cross_sectional': ['cross-sectional', 'cross sectional', 'prevalence study', 'survey'],
            'experimental': ['randomized', 'controlled trial', 'rct', 'placebo', 'intervention', 'clinical trial'],
            'review': ['review', 'meta-analysis', 'systematic review', 'literature review', 'scoping review'],
            'diagnostic': ['diagnosis', 'diagnostic', 'screening', 'biomarker', 'test accuracy'],
            'therapeutic': ['treatment', 'therapy', 'therapeutic', 'drug', 'medication', 'pharmacological'],
            'genetic': ['genetic', 'mutation', 'gene', 'genomic', 'molecular', 'dna', 'rna'],
            'pathological': ['pathology', 'histology', 'autopsy', 'biopsy', 'tissue', 'morphology'],
            'biochemical': ['biochemical', 'enzyme', 'protein', 'metabolic', 'biochemistry', 'assay']
        }
        
        # TF-IDF parameters
        self.tfidf_params = {
            'max_features': 5000,
            'stop_words': 'english',
            'ngram_range': (1, 3),
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
    
    def _load_embeddings(self, dataset_name: str) -> tuple[np.ndarray, Dict]:
        """Load SPECTER2 embeddings for the dataset."""
        if not self.enable_semantic_embeddings:
            return None, None
        
        project_root = self._get_project_root()
        
        # Load dataset config to get correct filenames
        datasets_config_path = os.path.join(project_root, 'data', 'datasets.json')
        try:
            with open(datasets_config_path, 'r') as f:
                datasets_config = json.load(f)
            
            if dataset_name in datasets_config:
                embeddings_filename = datasets_config[dataset_name].get('embeddings_filename', f'{dataset_name}.npy')
                metadata_filename = datasets_config[dataset_name].get('embeddings_metadata_filename', f'{dataset_name}_metadata.json')
            else:
                embeddings_filename = f'{dataset_name}.npy'
                metadata_filename = f'{dataset_name}_metadata.json'
        except:
            embeddings_filename = f'{dataset_name}.npy'
            metadata_filename = f'{dataset_name}_metadata.json'
        
        embeddings_path = os.path.join(project_root, 'data', 'embeddings', embeddings_filename)
        metadata_path = os.path.join(project_root, 'data', 'embeddings', metadata_filename)
        
        if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
            logger.warning(f"Embeddings not found for {dataset_name}, semantic features disabled")
            return None, None
        
        try:
            embeddings = np.load(embeddings_path)
            with open(metadata_path, 'r') as f:
                metadata_raw = json.load(f)
            
            # Transform metadata to have the expected structure
            if 'documents' in metadata_raw:
                # Extract openalex_ids from documents array
                openalex_ids = []
                for doc in metadata_raw['documents']:
                    openalex_id = doc.get('openalex_id', '')
                    # Keep the full URL format to match simulation data
                    openalex_ids.append(openalex_id)
                
                metadata = {
                    'openalex_id': openalex_ids,
                    'num_documents': metadata_raw.get('num_documents', len(openalex_ids)),
                    'embedding_dim': metadata_raw.get('embedding_dim', embeddings.shape[1] if embeddings.ndim > 1 else 0)
                }
            else:
                metadata = metadata_raw
            
            logger.info(f"Loaded embeddings: {embeddings.shape}")
            return embeddings, metadata
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None, None
    
    def fit(self, simulation_df: pd.DataFrame, 
            dataset_name: Optional[str] = None) -> 'ContentSimilarityModel':
        """
        Fit the content similarity model on simulation data.
        
        Args:
            simulation_df: DataFrame with simulation results
            dataset_name: Optional dataset name for embedding loading
        
        Returns:
            self: Returns the fitted model
        """
        logger.info("Fitting Content Similarity Model...")
        
        # Store simulation data
        self.simulation_data = simulation_df.copy()
        
        # Load embeddings if enabled and dataset name provided
        if dataset_name and self.enable_semantic_embeddings:
            self.embeddings, self.embeddings_metadata = self._load_embeddings(dataset_name)
        
        # Get relevant documents to establish "typical" patterns
        relevant_docs = simulation_df[simulation_df['label_included'] == 1].copy()
        
        if len(relevant_docs) == 0:
            logger.warning("No relevant documents found")
            self._set_default_fitted_state()
            return self
        
        logger.info(f"Analyzing {len(relevant_docs)} relevant documents to establish baseline patterns...")
        
        # Build typical vocabulary and patterns
        self._build_typical_vocabulary(relevant_docs)
        self._analyze_methodology_patterns(simulation_df)
        self._calculate_specificity_thresholds(relevant_docs)
        
        self.is_fitted = True
        logger.info("Content similarity model fitted successfully!")
        
        return self
    
    def _build_typical_vocabulary(self, relevant_docs: pd.DataFrame):
        """Build vocabulary characteristics of typical relevant documents."""
        logger.info("Building typical vocabulary patterns...")
        
        # Combine all relevant document texts
        texts = []
        for _, row in relevant_docs.iterrows():
            text = self._combine_text_fields(row)
            if text and len(text.strip()) > 10:
                processed_text = self._preprocess_text(text)
                texts.append(processed_text)
        
        if not texts:
            logger.warning("No text found in relevant documents")
            self._set_default_vocabulary()
            return
        
        # Create TF-IDF vectorizer for characteristic terms
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Create count vectorizer for frequency analysis
        self.count_vectorizer = CountVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        count_matrix = self.count_vectorizer.fit_transform(texts)
        
        # Store vocabulary characteristics
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        idf_scores = self.tfidf_vectorizer.idf_
        
        self.typical_vocabulary = {
            'feature_names': feature_names,
            'idf_scores': idf_scores,
            'mean_idf': np.mean(idf_scores),
            'std_idf': np.std(idf_scores),
            'tfidf_matrix': tfidf_matrix,
            'count_matrix': count_matrix
        }
        
        logger.info(f"Built vocabulary with {len(feature_names)} terms")
    
    def _analyze_methodology_patterns(self, simulation_df: pd.DataFrame):
        """Analyze methodology marker patterns across all documents."""
        logger.info("Analyzing methodology patterns...")
        
        methodology_counts = defaultdict(list)
        
        for _, row in simulation_df.iterrows():
            text = self._combine_text_fields(row).lower()
            
            # Count methodology markers
            for method_type, markers in self.methodology_markers.items():
                count = sum(text.count(marker) for marker in markers)
                methodology_counts[method_type].append(count)
        
        # Calculate typical methodology patterns
        self.methodology_patterns = {}
        for method_type, counts in methodology_counts.items():
            counts_array = np.array(counts)
            self.methodology_patterns[method_type] = {
                'mean': float(np.mean(counts_array)),
                'std': float(np.std(counts_array)),
                'max': float(np.max(counts_array)),
                'p75': float(np.percentile(counts_array, 75)),
                'p90': float(np.percentile(counts_array, 90))
            }
        
        logger.info(f"Analyzed {len(self.methodology_patterns)} methodology patterns")
    
    def _calculate_specificity_thresholds(self, relevant_docs: pd.DataFrame):
        """Calculate thresholds for identifying highly specific terms."""
        logger.info("Calculating specificity thresholds...")
        
        if not self.tfidf_vectorizer:
            self._set_default_specificity()
            return
        
        # Calculate document-level specificity scores
        specificity_scores = []
        
        for _, row in relevant_docs.iterrows():
            text = self._combine_text_fields(row)
            if text and len(text.strip()) > 10:
                try:
                    processed_text = self._preprocess_text(text)
                    tfidf_vector = self.tfidf_vectorizer.transform([processed_text])
                    
                    # Calculate specificity as mean of high IDF terms
                    high_idf_mask = self.typical_vocabulary['idf_scores'] > self.typical_vocabulary['mean_idf']
                    high_idf_scores = tfidf_vector.toarray()[0][high_idf_mask]
                    
                    if len(high_idf_scores) > 0:
                        specificity = np.mean(high_idf_scores)
                        specificity_scores.append(specificity)
                
                except Exception as e:
                    logger.debug(f"Failed to calculate specificity: {e}")
                    continue
        
        if specificity_scores:
            specificity_scores = np.array(specificity_scores)
            self.specificity_thresholds = {
                'mean': float(np.mean(specificity_scores)),
                'std': float(np.std(specificity_scores)),
                'p75': float(np.percentile(specificity_scores, 75)),
                'p90': float(np.percentile(specificity_scores, 90)),
                'p95': float(np.percentile(specificity_scores, 95))
            }
        else:
            self._set_default_specificity()
        
        logger.info("Specificity thresholds calculated")
    
    def _set_default_fitted_state(self):
        """Set default fitted state when insufficient data."""
        self.is_fitted = True
        self._set_default_vocabulary()
        self.methodology_patterns = {}
        self._set_default_specificity()
        logger.warning("Using default patterns due to insufficient training data")
    
    def _set_default_vocabulary(self):
        """Set default vocabulary when building fails."""
        self.typical_vocabulary = {
            'feature_names': np.array([]),
            'idf_scores': np.array([]),
            'mean_idf': 1.0,
            'std_idf': 0.5,
            'tfidf_matrix': None,
            'count_matrix': None
        }
    
    def _set_default_specificity(self):
        """Set default specificity thresholds."""
        self.specificity_thresholds = {
            'mean': 0.1,
            'std': 0.05,
            'p75': 0.12,
            'p90': 0.15,
            'p95': 0.18
        }
    
    def _combine_text_fields(self, row: pd.Series) -> str:
        """Combine title and abstract fields."""
        text_parts = []
        
        if pd.notna(row.get('title')) and str(row.get('title')).strip():
            text_parts.append(str(row['title']).strip())
        
        if pd.notna(row.get('abstract')) and str(row.get('abstract')).strip():
            text_parts.append(str(row['abstract']).strip())
        
        return ' '.join(text_parts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
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
        Extract content similarity features for target documents.
        
        Args:
            target_documents: List of document IDs to extract features for
        
        Returns:
            DataFrame with content similarity features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
        logger.info(f"Extracting content features for {len(target_documents)} documents")
        
        features = []
        for doc_id in tqdm(target_documents, desc="Extracting content features"):
            doc_features = self._extract_single_document_features(doc_id)
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def _extract_single_document_features(self, doc_id: str) -> Dict[str, Any]:
        """Extract content features for a single document."""
        if self.simulation_data is None:
            return self._get_zero_features(doc_id)
        
        # Find document in simulation data
        doc_row = self.simulation_data[self.simulation_data['openalex_id'] == doc_id]
        
        if doc_row.empty:
            return self._get_zero_features(doc_id)
        
        row = doc_row.iloc[0]
        text = self._combine_text_fields(row)
        
        if not text or len(text.strip()) <= 10:
            return self._get_zero_features(doc_id)
        
        processed_text = self._preprocess_text(text)
        
        # Extract different types of features
        features = {'openalex_id': doc_id}
        features.update(self._extract_specificity_features(processed_text))
        features.update(self._extract_methodology_features(processed_text))
        features.update(self._extract_diversity_features(processed_text))
        features.update(self._extract_semantic_features(doc_id))
        
        return features
    
    def _extract_specificity_features(self, text: str) -> Dict[str, float]:
        """Extract specificity-related features from text."""
        if not self.tfidf_vectorizer or not self.typical_vocabulary['feature_names'].size:
            return {
                'specificity_score': 0.0,
                'rare_terms_ratio': 0.0,
                'unique_ngrams': 0.0,
                'technical_term_density': 0.0
            }
        
        try:
            # TF-IDF based specificity
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            
            # Calculate specificity as weighted average of IDF scores
            tfidf_scores = tfidf_vector.toarray()[0]
            idf_scores = self.typical_vocabulary['idf_scores']
            
            # Weighted specificity score
            specificity_score = np.sum(tfidf_scores * idf_scores) / max(1, np.sum(tfidf_scores))
            
            # Rare terms (high IDF) ratio
            rare_threshold = self.typical_vocabulary['mean_idf'] + self.typical_vocabulary['std_idf']
            rare_mask = idf_scores > rare_threshold
            rare_terms_count = np.sum(tfidf_scores[rare_mask] > 0)
            total_terms_count = np.sum(tfidf_scores > 0)
            rare_terms_ratio = rare_terms_count / max(1, total_terms_count)
            
            # Count unique n-grams
            words = text.split()
            unique_ngrams = len(set(words)) / max(1, len(words))
            
            # Technical term density (terms with numbers, specialized patterns)
            technical_pattern = re.compile(r'\w*\d+\w*|[A-Z]{2,}|\w*-\w*')
            technical_matches = technical_pattern.findall(text)
            technical_term_density = len(technical_matches) / max(1, len(words))
            
            return {
                'specificity_score': float(specificity_score),
                'rare_terms_ratio': float(rare_terms_ratio),
                'unique_ngrams': float(unique_ngrams),
                'technical_term_density': float(technical_term_density)
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract specificity features: {e}")
            return {
                'specificity_score': 0.0,
                'rare_terms_ratio': 0.0,
                'unique_ngrams': 0.0,
                'technical_term_density': 0.0
            }
    
    def _extract_methodology_features(self, text: str) -> Dict[str, float]:
        """Extract methodology-related features from text."""
        if not self.methodology_patterns:
            return {f'methodology_{method_type}': 0.0 for method_type in self.methodology_markers.keys()}
        
        methodology_features = {}
        text_lower = text.lower()
        
        for method_type, markers in self.methodology_markers.items():
            # Count methodology markers
            count = sum(text_lower.count(marker) for marker in markers)
            
            # Normalize against typical patterns
            if method_type in self.methodology_patterns:
                patterns = self.methodology_patterns[method_type]
                # Calculate deviation from typical pattern
                deviation = (count - patterns['mean']) / max(0.1, patterns['std'])
                normalized_score = min(1.0, max(0.0, (deviation + 1) / 2))  # Scale to 0-1
            else:
                normalized_score = min(1.0, count / 5.0)  # Simple normalization
            
            methodology_features[f'methodology_{method_type}'] = float(normalized_score)
        
        return methodology_features
    
    def _extract_diversity_features(self, text: str) -> Dict[str, float]:
        """Extract vocabulary diversity features from text."""
        try:
            words = text.split()
            
            if len(words) < 5:
                return {
                    'vocabulary_entropy': 0.0,
                    'lexical_diversity': 0.0,
                    'term_frequency_variance': 0.0
                }
            
            # Calculate vocabulary entropy
            word_counts = Counter(words)
            total_words = len(words)
            word_probs = [count / total_words for count in word_counts.values()]
            vocab_entropy = entropy(word_probs, base=2)
            
            # Lexical diversity (type-token ratio)
            lexical_diversity = len(set(words)) / len(words)
            
            # Term frequency variance
            tf_values = list(word_counts.values())
            tf_variance = np.var(tf_values) if len(tf_values) > 1 else 0.0
            
            return {
                'vocabulary_entropy': float(vocab_entropy),
                'lexical_diversity': float(lexical_diversity),
                'term_frequency_variance': float(tf_variance)
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract diversity features: {e}")
            return {
                'vocabulary_entropy': 0.0,
                'lexical_diversity': 0.0,
                'term_frequency_variance': 0.0
            }
    
    def _extract_semantic_features(self, doc_id: str) -> Dict[str, float]:
        """Extract semantic similarity features using embeddings."""
        if not self.embeddings_metadata or doc_id not in self.embeddings_metadata.get('openalex_id', []):
            return {
                'semantic_similarity_mean': 0.0,
                'semantic_similarity_max': 0.0,
                'semantic_isolation': 1.0
            }
        
        try:
            # Get document embedding
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata['openalex_id'])}
            doc_idx = id_to_idx[doc_id]
            doc_embedding = self.embeddings[doc_idx:doc_idx+1]
            
            # Get embeddings of relevant documents in simulation data
            relevant_ids = self.simulation_data[
                self.simulation_data['label_included'] == 1
            ]['openalex_id'].tolist()
            
            relevant_with_embeddings = [doc for doc in relevant_ids if doc in id_to_idx]
            
            if not relevant_with_embeddings:
                return {
                    'semantic_similarity_mean': 0.0,
                    'semantic_similarity_max': 0.0,
                    'semantic_isolation': 1.0
                }
            
            relevant_indices = [id_to_idx[doc] for doc in relevant_with_embeddings]
            relevant_embeddings = self.embeddings[relevant_indices]
            
            # Compute similarities
            similarities = cosine_similarity(doc_embedding, relevant_embeddings)[0]
            
            return {
                'semantic_similarity_mean': float(np.mean(similarities)),
                'semantic_similarity_max': float(np.max(similarities)),
                'semantic_isolation': float(1.0 - np.mean(similarities))
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract semantic features for {doc_id}: {e}")
            return {
                'semantic_similarity_mean': 0.0,
                'semantic_similarity_max': 0.0,
                'semantic_isolation': 1.0
            }
    
    def _get_zero_features(self, doc_id: str) -> Dict[str, Any]:
        """Get zero features for documents without text or when processing fails."""
        features = {
            'openalex_id': doc_id,
            'specificity_score': 0.0,
            'rare_terms_ratio': 0.0,
            'unique_ngrams': 0.0,
            'technical_term_density': 0.0,
            'vocabulary_entropy': 0.0,
            'lexical_diversity': 0.0,
            'term_frequency_variance': 0.0,
            'semantic_similarity_mean': 0.0,
            'semantic_similarity_max': 0.0,
            'semantic_isolation': 1.0
        }
        
        # Add methodology features
        for method_type in self.methodology_markers.keys():
            features[f'methodology_{method_type}'] = 0.0
        
        return features
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Generate content-based relevance scores for target documents.
        
        Args:
            target_documents: List of document IDs to score
        
        Returns:
            Dictionary mapping document IDs to relevance scores (0-1)
        """
        if not self.is_fitted:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        logger.info(f"Computing content-based relevance scores for {len(target_documents)} documents")
        
        # Extract features
        features_df = self.extract_features(target_documents)
        
        # Calculate relevance scores
        scores = {}
        for _, row in features_df.iterrows():
            doc_id = row['openalex_id']
            score = self._calculate_content_relevance_score(row)
            scores[doc_id] = score
        
        return scores
    
    def _calculate_content_relevance_score(self, features: pd.Series) -> float:
        """Calculate relevance score based on content features."""
        # Specificity component (higher specificity may indicate outliers)
        specificity_component = min(1.0, features.get('specificity_score', 0.0) * 2.0)
        
        # Methodology diversity (documents with unusual methodology markers)
        methodology_scores = [features.get(f'methodology_{method}', 0.0) 
                             for method in self.methodology_markers.keys()]
        methodology_diversity = np.std(methodology_scores) if methodology_scores else 0.0
        
        # Vocabulary diversity (documents with unusual language patterns)
        vocab_entropy = features.get('vocabulary_entropy', 0.0)
        lexical_diversity = features.get('lexical_diversity', 0.0)
        diversity_component = (vocab_entropy + lexical_diversity) / 2.0
        
        # Semantic isolation (documents semantically distant from typical relevant docs)
        semantic_isolation = features.get('semantic_isolation', 0.0)
        
        # Technical term density (highly technical documents may be outliers)
        technical_density = features.get('technical_term_density', 0.0)
        
        # Adaptive weighting based on available features (optimized for outlier detection)
        weights = {
            'specificity': 0.2,    # Decreased slightly
            'methodology': 0.15,   # Decreased - less reliable
            'diversity': 0.2,      # Unchanged
            'semantic': 0.35,      # Increased - semantic isolation is key for outliers
            'technical': 0.1       # Unchanged
        }
        
        # Combine components
        final_score = (
            weights['specificity'] * specificity_component +
            weights['methodology'] * min(1.0, methodology_diversity) +
            weights['diversity'] * min(1.0, diversity_component) +
            weights['semantic'] * semantic_isolation +
            weights['technical'] * min(1.0, technical_density * 5.0)
        )
        
        return float(max(0.0, min(1.0, final_score)))
    
    def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Provide detailed content analysis for a specific document.
        
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
                'has_vectorizer': self.tfidf_vectorizer is not None,
                'has_embeddings': self.embeddings is not None,
                'vocab_size': len(self.typical_vocabulary['feature_names']) if self.typical_vocabulary else 0,
            }
        }
        
        # Add interpretation
        if features['specificity_score'] > 0.3:
            analysis['interpretation'] = "High specificity - highly specialized content"
        elif features['semantic_isolation'] > 0.7:
            analysis['interpretation'] = "High semantic isolation - content differs from typical relevant docs"
        elif features['technical_term_density'] > 0.2:
            analysis['interpretation'] = "High technical density - specialized terminology"
        else:
            analysis['interpretation'] = "Normal content pattern"
        
        return analysis


def main():
    """Example usage of the ContentSimilarityModel."""
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
            print(f"\nFitting content similarity model on {dataset_name}...")
            model = ContentSimilarityModel()
            model.fit(simulation_df, dataset_name)
            
            # Get example documents
            example_docs = simulation_df['openalex_id'].head(10).tolist()
            
            # Extract features
            print("\nExtracting content features...")
            features_df = model.extract_features(example_docs)
            print(features_df[['openalex_id', 'specificity_score', 'semantic_isolation']].head())
            
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