"""
Citation Network Outlier Detection Model

This module provides a streamlined, on-the-fly outlier detection system for citation networks
using research-backed techniques: LOF, Isolation Forest, DBSCAN, and One-Class SVM.
The model takes a dataset as input and immediately returns outlier scores without training sets.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import time
from tqdm import tqdm

# Core libraries for outlier detection
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Graph analysis
import networkx as nx
from scipy.spatial.distance import cosine
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CitationNetworkOutlierDetector:
    """
    On-the-fly Citation Network Outlier Detection using Multiple Research-Backed Methods
    
    Implements:
    - Local Outlier Factor (LOF) for density-based local anomaly detection
    - Isolation Forest for global anomaly detection
    - DBSCAN for cluster-based outlier identification  
    - One-Class SVM for boundary-based anomaly detection
    - Combined ensemble scoring for robust detection
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 enable_semantic: bool = True,
                 random_state: int = 42):
        """
        Initialize the outlier detector.
        
        Args:
            contamination: Expected proportion of outliers (0.05 to 0.2)
            enable_semantic: Whether to use semantic embeddings if available
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.enable_semantic = enable_semantic
        self.random_state = random_state
        
        # Initialize outlier detection algorithms
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            metric='euclidean',
            novelty=False,  # For direct prediction on training data
            n_jobs=-1
        )
        
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=contamination  # nu parameter controls the fraction of outliers
        )
        
        self.dbscan = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean',
            n_jobs=-1
        )
        
        # Scalers for different feature types
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # State variables
        self.is_fitted = False
        self.feature_names = []
        self.embeddings = None
        self.embeddings_metadata = None
        
    def fit_predict_outliers(self, 
                           simulation_df: pd.DataFrame,
                           dataset_name: str = None) -> Dict[str, np.ndarray]:
        """
        Main method: takes dataset and immediately returns outlier scores.
        No training phase - everything happens on-the-fly.
        
        Args:
            simulation_df: DataFrame with paper data and citation information
            dataset_name: Optional dataset name for loading embeddings
            
        Returns:
            Dictionary with outlier scores and predictions from all methods
        """
        logger.info(f"Starting on-the-fly outlier detection for {len(simulation_df)} documents")
        start_time = time.time()
        
        # Build citation network
        G = self._build_citation_network(simulation_df)
        logger.info(f"Built citation network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Load embeddings if available and enabled
        if self.enable_semantic and dataset_name:
            self._load_embeddings(dataset_name)
        
        # Extract comprehensive features
        features_df = self._extract_network_features(G, simulation_df)
        logger.info(f"Extracted features: {features_df.shape[1]-1} features for {features_df.shape[0]} documents")
        
        # Get feature matrix (excluding document ID)
        feature_matrix = features_df.drop('openalex_id', axis=1).values
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features for different algorithms
        features_standard = self.standard_scaler.fit_transform(feature_matrix)
        features_robust = self.robust_scaler.fit_transform(feature_matrix)
        
        # Apply all outlier detection methods
        outlier_results = {}
        
        # 1. Local Outlier Factor (LOF)
        logger.info("Applying Local Outlier Factor (LOF)...")
        lof_scores = self.lof.fit_predict(features_standard)
        lof_negative_scores = self.lof.negative_outlier_factor_
        outlier_results['lof_predictions'] = lof_scores
        outlier_results['lof_scores'] = -lof_negative_scores  # Convert to positive scores
        
        # 2. Isolation Forest
        logger.info("Applying Isolation Forest...")
        if_predictions = self.isolation_forest.fit_predict(features_standard)
        if_scores = self.isolation_forest.decision_function(features_standard)
        outlier_results['isolation_forest_predictions'] = if_predictions
        outlier_results['isolation_forest_scores'] = -if_scores  # Convert to anomaly scores
        
        # 3. One-Class SVM
        logger.info("Applying One-Class SVM...")
        svm_predictions = self.one_class_svm.fit_predict(features_robust)
        svm_scores = self.one_class_svm.decision_function(features_robust)
        outlier_results['one_class_svm_predictions'] = svm_predictions
        outlier_results['one_class_svm_scores'] = -svm_scores  # Convert to anomaly scores
        
        # 4. DBSCAN (cluster-based outliers)
        logger.info("Applying DBSCAN clustering...")
        dbscan_labels = self.dbscan.fit_predict(features_standard)
        dbscan_predictions = (dbscan_labels == -1).astype(int) * 2 - 1  # Convert noise to -1, clusters to 1
        # For DBSCAN scores, use distance to nearest cluster center
        dbscan_scores = self._compute_dbscan_scores(features_standard, dbscan_labels)
        outlier_results['dbscan_predictions'] = dbscan_predictions
        outlier_results['dbscan_scores'] = dbscan_scores
        
        # 5. Ensemble scoring
        logger.info("Computing ensemble scores...")
        ensemble_scores = self._compute_ensemble_scores(outlier_results)
        outlier_results['ensemble_scores'] = ensemble_scores
        outlier_results['ensemble_predictions'] = (ensemble_scores > np.percentile(ensemble_scores, 100 * (1 - self.contamination))).astype(int) * 2 - 1
        
        # Add document IDs to results
        outlier_results['openalex_ids'] = features_df['openalex_id'].values
        
        # Store results and mark as fitted
        self.outlier_results = outlier_results
        self.feature_matrix = feature_matrix
        self.features_df = features_df
        self.graph = G
        self.is_fitted = True
        
        # Log summary
        total_time = time.time() - start_time
        num_outliers_ensemble = sum(outlier_results['ensemble_predictions'] == -1)
        logger.info(f"Outlier detection completed in {total_time:.2f}s")
        logger.info(f"Detected {num_outliers_ensemble} outliers ({num_outliers_ensemble/len(simulation_df)*100:.1f}%) using ensemble method")
        
        return outlier_results
        
    def _build_citation_network(self, simulation_df: pd.DataFrame) -> nx.Graph:
        """Build citation network from simulation data."""
        G = nx.Graph()
        
        # Add all documents as nodes
        for _, row in simulation_df.iterrows():
            G.add_node(row['openalex_id'], 
                      title=row.get('title', ''),
                      year=row.get('year', 0),
                      label=row.get('label_included', 0))
        
        # Add citation edges
        citation_count = 0
        for _, row in tqdm(simulation_df.iterrows(), desc="Building citation network", total=len(simulation_df)):
            doc_id = row['openalex_id']
            
            # Process different reference formats
            references = self._extract_references(row)
            
            for ref_id in references:
                if ref_id in G.nodes and ref_id != doc_id:
                    G.add_edge(doc_id, ref_id, edge_type='citation')
                    citation_count += 1
        
        logger.info(f"Added {citation_count} citation edges")
        return G
    
    def _extract_references(self, row: pd.Series) -> List[str]:
        """Extract references from different possible columns."""
        references = []
        
        # Try different reference column names
        ref_columns = ['references', 'cited_by', 'cites', 'references_list']
        
        for col in ref_columns:
            if col in row and row[col] is not None:
                try:
                    # Handle different formats
                    if isinstance(row[col], str):
                        if row[col].startswith('['):
                            refs = json.loads(row[col])
                        else:
                            refs = [ref.strip() for ref in row[col].replace(';', ',').split(',') if ref.strip()]
                    elif isinstance(row[col], list):
                        refs = row[col]
                    else:
                        continue
                    
                    # Clean and validate references
                    for ref_id in refs:
                        if isinstance(ref_id, str) and ref_id.strip():
                            ref_id = ref_id.strip()
                            # Ensure proper OpenAlex URL format
                            if not ref_id.startswith('https://openalex.org/'):
                                if ref_id.startswith('W'):
                                    ref_id = f'https://openalex.org/{ref_id}'
                                else:
                                    continue
                            references.append(ref_id)
                    
                    if references:  # If we found references, return them
                        return references
                        
                except Exception as e:
                    logger.debug(f"Failed to parse {col} for {row.get('openalex_id', 'unknown')}: {e}")
                    continue
        
        return references
    
    def _load_embeddings(self, dataset_name: str):
        """Load SPECTER2 embeddings if available."""
        project_root = self._get_project_root()
        embeddings_path = os.path.join(project_root, 'data', 'embeddings', f'{dataset_name}.npy')
        metadata_path = os.path.join(project_root, 'data', 'embeddings', f'{dataset_name}_metadata.json')
        
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            try:
                self.embeddings = np.load(embeddings_path)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_raw = json.load(f)
                
                # Process metadata
                if 'documents' in metadata_raw:
                    openalex_ids = [doc.get('openalex_id', '') for doc in metadata_raw['documents']]
                    self.embeddings_metadata = {'openalex_id': openalex_ids}
                else:
                    self.embeddings_metadata = metadata_raw
                
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self.embeddings = None
                self.embeddings_metadata = None
        else:
            logger.info(f"No embeddings found for {dataset_name}")
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    def _extract_network_features(self, G: nx.Graph, simulation_df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive network and semantic features."""
        features = []
        
        # Get relevant documents (included papers)
        relevant_docs = set(simulation_df[simulation_df['label_included'] == 1]['openalex_id'].tolist())
        
        # Pre-compute centrality measures for efficiency
        logger.info("Computing graph centrality measures...")
        if len(G.nodes) > 0:
            try:
                # Only compute expensive measures for smaller graphs
                if len(G.nodes) < 5000:
                    pagerank = nx.pagerank(G, max_iter=50)
                    betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
                    closeness = nx.closeness_centrality(G)
                    eigenvector = nx.eigenvector_centrality(G, max_iter=100)
                else:
                    # For large graphs, use simpler measures
                    pagerank = {node: 1.0/len(G.nodes) for node in G.nodes}
                    betweenness = {node: 0.0 for node in G.nodes}
                    closeness = {node: 0.0 for node in G.nodes}
                    eigenvector = {node: 1.0/len(G.nodes) for node in G.nodes}
            except:
                # Fallback if centrality computation fails
                pagerank = {node: 1.0/len(G.nodes) for node in G.nodes}
                betweenness = {node: 0.0 for node in G.nodes}
                closeness = {node: 0.0 for node in G.nodes}
                eigenvector = {node: 1.0/len(G.nodes) for node in G.nodes}
        else:
            pagerank = betweenness = closeness = eigenvector = {}
        
        # Extract features for each document
        for _, row in tqdm(simulation_df.iterrows(), desc="Extracting features", total=len(simulation_df)):
            doc_id = row['openalex_id']
            doc_features = {'openalex_id': doc_id}
            
            if doc_id in G.nodes:
                # Basic connectivity features
                degree = G.degree(doc_id)
                neighbors = list(G.neighbors(doc_id))
                relevant_neighbors = len([n for n in neighbors if n in relevant_docs])
                
                doc_features.update({
                    'degree': float(degree),
                    'relevant_neighbors': float(relevant_neighbors),
                    'relevant_ratio': float(relevant_neighbors / max(1, degree)),
                    'clustering': float(nx.clustering(G, doc_id)),
                    'pagerank': float(pagerank.get(doc_id, 0.0)),
                    'betweenness': float(betweenness.get(doc_id, 0.0)),
                    'closeness': float(closeness.get(doc_id, 0.0)),
                    'eigenvector': float(eigenvector.get(doc_id, 0.0)),
                })
                
                # Neighborhood diversity
                if neighbors:
                    neighbor_degrees = [G.degree(n) for n in neighbors]
                    doc_features.update({
                        'mean_neighbor_degree': float(np.mean(neighbor_degrees)),
                        'std_neighbor_degree': float(np.std(neighbor_degrees)),
                        'max_neighbor_degree': float(np.max(neighbor_degrees)),
                    })
                else:
                    doc_features.update({
                        'mean_neighbor_degree': 0.0,
                        'std_neighbor_degree': 0.0,
                        'max_neighbor_degree': 0.0,
                    })
                
                # Path-based features
                if relevant_docs:
                    min_distance_to_relevant = float('inf')
                    for rel_doc in list(relevant_docs)[:10]:  # Sample to avoid expensive computation
                        if rel_doc in G.nodes and rel_doc != doc_id:
                            try:
                                dist = nx.shortest_path_length(G, doc_id, rel_doc)
                                min_distance_to_relevant = min(min_distance_to_relevant, dist)
                            except nx.NetworkXNoPath:
                                continue
                    
                    doc_features['min_distance_to_relevant'] = float(min_distance_to_relevant) if min_distance_to_relevant != float('inf') else 10.0
                else:
                    doc_features['min_distance_to_relevant'] = 10.0
                
            else:
                # Zero features for nodes not in graph
                doc_features.update({
                    'degree': 0.0,
                    'relevant_neighbors': 0.0,
                    'relevant_ratio': 0.0,
                    'clustering': 0.0,
                    'pagerank': 0.0,
                    'betweenness': 0.0,
                    'closeness': 0.0,
                    'eigenvector': 0.0,
                    'mean_neighbor_degree': 0.0,
                    'std_neighbor_degree': 0.0,
                    'max_neighbor_degree': 0.0,
                    'min_distance_to_relevant': 10.0,
                })
            
            # Semantic features if embeddings available
            if self.embeddings is not None and self.embeddings_metadata is not None:
                semantic_features = self._get_semantic_features(doc_id, relevant_docs)
                doc_features.update(semantic_features)
            else:
                doc_features.update({
                    'semantic_similarity_to_relevant': 0.0,
                    'semantic_isolation_score': 1.0,
                    'max_semantic_similarity': 0.0,
                })
            
            # Citation pattern features
            citation_features = self._get_citation_features(row)
            doc_features.update(citation_features)
            
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def _get_semantic_features(self, doc_id: str, relevant_docs: set) -> Dict[str, float]:
        """Get semantic similarity features using embeddings."""
        features = {
            'semantic_similarity_to_relevant': 0.0,
            'semantic_isolation_score': 1.0,
            'max_semantic_similarity': 0.0,
        }
        
        if not self.embeddings_metadata or doc_id not in self.embeddings_metadata.get('openalex_id', []):
            return features
        
        try:
            # Get document embedding
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata['openalex_id'])}
            doc_idx = id_to_idx[doc_id]
            doc_embedding = self.embeddings[doc_idx:doc_idx+1]
            
            # Get relevant document embeddings
            relevant_with_embeddings = [doc for doc in relevant_docs if doc in id_to_idx and doc != doc_id]
            
            if relevant_with_embeddings:
                relevant_indices = [id_to_idx[doc] for doc in relevant_with_embeddings]
                relevant_embeddings = self.embeddings[relevant_indices]
                
                # Compute similarities
                similarities = cosine_similarity(doc_embedding, relevant_embeddings)[0]
                
                features.update({
                    'semantic_similarity_to_relevant': float(np.mean(similarities)),
                    'semantic_isolation_score': float(1.0 - np.mean(similarities)),
                    'max_semantic_similarity': float(np.max(similarities)),
                })
        except Exception as e:
            logger.debug(f"Failed to compute semantic features for {doc_id}: {e}")
        
        return features
    
    def _get_citation_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract citation-based features from paper metadata."""
        features = {
            'publication_year': float(row.get('year', 0)),
            'title_length': float(len(str(row.get('title', '')))),
            'has_references': 0.0,
            'num_references_estimate': 0.0,
        }
        
        # Count references if available
        references = self._extract_references(row)
        if references:
            features['has_references'] = 1.0
            features['num_references_estimate'] = float(len(references))
        
        return features
    
    def _compute_dbscan_scores(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute outlier scores for DBSCAN (distance to nearest cluster center)."""
        scores = np.zeros(len(features))
        
        # For noise points (label = -1), compute distance to nearest cluster center
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if not unique_labels:
            # If no clusters found, return uniform scores
            return np.ones(len(features))
        
        # Compute cluster centers
        cluster_centers = {}
        for label in unique_labels:
            cluster_points = features[labels == label]
            cluster_centers[label] = np.mean(cluster_points, axis=0)
        
        # Compute scores
        for i, (point, label) in enumerate(zip(features, labels)):
            if label == -1:  # Noise point
                # Distance to nearest cluster center
                min_distance = min(np.linalg.norm(point - center) for center in cluster_centers.values())
                scores[i] = min_distance
            else:  # Cluster point
                # Distance to own cluster center
                center = cluster_centers[label]
                scores[i] = np.linalg.norm(point - center)
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _compute_ensemble_scores(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute ensemble scores by combining multiple methods."""
        # Normalize individual scores
        lof_scores_norm = self._normalize_scores(results['lof_scores'])
        if_scores_norm = self._normalize_scores(results['isolation_forest_scores'])
        svm_scores_norm = self._normalize_scores(results['one_class_svm_scores'])
        dbscan_scores_norm = self._normalize_scores(results['dbscan_scores'])
        
        # Weighted ensemble (can be tuned based on method performance)
        weights = {
            'lof': 0.3,           # Good for local outliers
            'isolation_forest': 0.3,  # Good for global outliers
            'one_class_svm': 0.2,     # Good for boundary outliers
            'dbscan': 0.2,            # Good for cluster-based outliers
        }
        
        ensemble_scores = (
            weights['lof'] * lof_scores_norm +
            weights['isolation_forest'] * if_scores_norm +
            weights['one_class_svm'] * svm_scores_norm +
            weights['dbscan'] * dbscan_scores_norm
        )
        
        return ensemble_scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        scores = np.array(scores)
        if scores.max() - scores.min() > 1e-8:
            return (scores - scores.min()) / (scores.max() - scores.min())
        else:
            return np.zeros_like(scores)
    
    def get_outlier_documents(self, 
                            method: str = 'ensemble',
                            top_k: int = None) -> pd.DataFrame:
        """
        Get outlier documents with detailed information.
        
        Args:
            method: Which method to use ('ensemble', 'lof', 'isolation_forest', 'one_class_svm', 'dbscan')
            top_k: Number of top outliers to return (None = all outliers)
            
        Returns:
            DataFrame with outlier information sorted by score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting outlier documents")
        
        # Get scores and predictions for specified method
        score_key = f'{method}_scores'
        pred_key = f'{method}_predictions'
        
        if score_key not in self.outlier_results:
            raise ValueError(f"Unknown method: {method}")
        
        scores = self.outlier_results[score_key]
        predictions = self.outlier_results[pred_key]
        doc_ids = self.outlier_results['openalex_ids']
        
        # Create results DataFrame
        results = []
        for i, (doc_id, score, pred) in enumerate(zip(doc_ids, scores, predictions)):
            is_outlier = pred == -1 if method != 'ensemble' else pred == 1
            
            if top_k is None or is_outlier or len(results) < top_k:
                # Get additional features
                doc_features = self.features_df[self.features_df['openalex_id'] == doc_id].iloc[0]
                
                result = {
                    'document_id': doc_id,
                    'outlier_score': float(score),
                    'is_outlier': bool(is_outlier),
                    'degree': doc_features['degree'],
                    'relevant_neighbors': doc_features['relevant_neighbors'],
                    'relevant_ratio': doc_features['relevant_ratio'],
                    'clustering_coefficient': doc_features['clustering'],
                    'pagerank': doc_features['pagerank'],
                    'semantic_similarity': doc_features.get('semantic_similarity_to_relevant', 0.0),
                    'min_distance_to_relevant': doc_features['min_distance_to_relevant'],
                }
                results.append(result)
        
        # Sort by outlier score (descending)
        results_df = pd.DataFrame(results).sort_values('outlier_score', ascending=False)
        
        if top_k is not None:
            results_df = results_df.head(top_k)
        
        return results_df
    
    def get_method_comparison(self) -> pd.DataFrame:
        """Compare results across all outlier detection methods."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before comparing methods")
        
        methods = ['lof', 'isolation_forest', 'one_class_svm', 'dbscan', 'ensemble']
        comparison = []
        
        for method in methods:
            pred_key = f'{method}_predictions'
            score_key = f'{method}_scores'
            
            predictions = self.outlier_results[pred_key]
            scores = self.outlier_results[score_key]
            
            if method == 'ensemble':
                num_outliers = sum(predictions == 1)
            else:
                num_outliers = sum(predictions == -1)
            
            comparison.append({
                'method': method,
                'num_outliers': num_outliers,
                'outlier_percentage': num_outliers / len(predictions) * 100,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
            })
        
        return pd.DataFrame(comparison)
    
    def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Provide detailed analysis of a specific document.
        
        Args:
            doc_id: Document ID to analyze
            
        Returns:
            Dictionary with comprehensive analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing documents")
        
        # Find document index
        doc_indices = np.where(self.outlier_results['openalex_ids'] == doc_id)[0]
        if len(doc_indices) == 0:
            raise ValueError(f"Document {doc_id} not found in results")
        
        idx = doc_indices[0]
        
        # Get scores from all methods
        analysis = {
            'document_id': doc_id,
            'scores': {
                'lof': float(self.outlier_results['lof_scores'][idx]),
                'isolation_forest': float(self.outlier_results['isolation_forest_scores'][idx]),
                'one_class_svm': float(self.outlier_results['one_class_svm_scores'][idx]),
                'dbscan': float(self.outlier_results['dbscan_scores'][idx]),
                'ensemble': float(self.outlier_results['ensemble_scores'][idx]),
            },
            'predictions': {
                'lof': bool(self.outlier_results['lof_predictions'][idx] == -1),
                'isolation_forest': bool(self.outlier_results['isolation_forest_predictions'][idx] == -1),
                'one_class_svm': bool(self.outlier_results['one_class_svm_predictions'][idx] == -1),
                'dbscan': bool(self.outlier_results['dbscan_predictions'][idx] == -1),
                'ensemble': bool(self.outlier_results['ensemble_predictions'][idx] == 1),
            }
        }
        
        # Get document features
        doc_features = self.features_df[self.features_df['openalex_id'] == doc_id]
        if not doc_features.empty:
            features = doc_features.iloc[0].to_dict()
            features.pop('openalex_id', None)
            analysis['features'] = features
        
        # Network analysis
        if hasattr(self, 'graph') and doc_id in self.graph.nodes:
            neighbors = list(self.graph.neighbors(doc_id))
            analysis['network'] = {
                'in_network': True,
                'num_neighbors': len(neighbors),
                'neighbors_sample': neighbors[:5],  # Show first 5 neighbors
            }
        else:
            analysis['network'] = {
                'in_network': False,
                'num_neighbors': 0,
                'neighbors_sample': [],
            }
        
        return analysis


def main():
    """Example usage of the CitationNetworkOutlierDetector."""
    print("Citation Network Outlier Detection - Example Usage")
    print("=" * 50)
    
    # This would typically load your actual dataset
    # For demonstration, we'll show the expected interface
    
    # Load your simulation data
    # simulation_df = pd.read_csv('path/to/your/simulation_data.csv')
    
    print("Expected usage:")
    print("""
    # Initialize detector
    detector = CitationNetworkOutlierDetector(
        contamination=0.1,  # Expect 10% outliers
        enable_semantic=True
    )
    
    # Run outlier detection (everything happens here)
    results = detector.fit_predict_outliers(simulation_df, dataset_name='your_dataset')
    
    # Get outlier documents
    outliers = detector.get_outlier_documents(method='ensemble', top_k=20)
    print(outliers)
    
    # Compare methods
    comparison = detector.get_method_comparison()
    print(comparison)
    
    # Analyze specific document
    analysis = detector.analyze_document('https://openalex.org/W12345')
    print(analysis)
    """)
    
    print("\nFeatures extracted:")
    print("- Network features: degree, clustering, centrality measures")
    print("- Citation features: references, publication patterns")
    print("- Semantic features: similarity to relevant papers (if embeddings available)")
    print("- Path features: distance to relevant papers in citation network")
    
    print("\nOutlier detection methods applied:")
    print("- LOF: Local density-based outlier detection")
    print("- Isolation Forest: Global anomaly detection")
    print("- One-Class SVM: Boundary-based anomaly detection") 
    print("- DBSCAN: Cluster-based outlier identification")
    print("- Ensemble: Weighted combination of all methods")


if __name__ == "__main__":
    main()