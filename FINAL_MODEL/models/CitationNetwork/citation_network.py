"""
Citation Network Outlier Detection Model

This module provides a streamlined outlier detection system for citation networks
using LOF on embeddings, Isolation Forest, and DBSCAN with automatic eps tuning.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Core libraries for outlier detection
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import umap

# Graph analysis
import networkx as nx
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Automatic knee detection for DBSCAN eps tuning
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    logging.warning("kneed library not available - using fallback eps detection")

# Import utility functions from utils.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import (
    get_project_root, load_datasets_config, load_simulation_data, 
    load_embeddings, load_synergy_dataset, prompt_dataset_selection,
    evaluate_outlier_ranking, normalize_scores, compute_ensemble_weights
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CitationNetworkOutlierDetector:
    """
    Streamlined Citation Network Outlier Detection using:
    - LOF on embeddings for semantic outlier detection
    - Isolation Forest for global anomaly detection
    - DBSCAN with automatic eps tuning for density-based clustering
    """
    
    def __init__(self, 
                 use_umap: bool = False,
                 umap_components: int = 50,
                 random_state: int = 42):
        """
        Initialize the outlier detector.
        
        Args:
            use_umap: Whether to use UMAP for dimensionality reduction of embeddings
            umap_components: Number of UMAP components to reduce embeddings to
            random_state: Random state for reproducibility
        """
        self.use_umap = use_umap
        self.umap_components = umap_components
        self.random_state = random_state
        
        # Initialize scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # State variables
        self.is_fitted = False
        self.embeddings = None
        self.embeddings_metadata = None
        self.umap_reducer = None
        self.reduced_embeddings = None
        
    def fit_predict_outliers(self, 
                           simulation_df: pd.DataFrame,
                           dataset_name: str = None) -> Dict[str, np.ndarray]:
        """
        Main method: takes dataset and immediately returns outlier scores.
        
        Args:
            simulation_df: DataFrame with paper data and citation information
            dataset_name: Optional dataset name for loading embeddings and synergy data
            
        Returns:
            Dictionary with outlier scores and predictions from all methods
        """
        logger.info(f"Starting outlier detection for {len(simulation_df)} documents")
        start_time = time.time()
        
        # Get dataset size for method configuration
        n_docs = len(simulation_df)
        logger.info(f"Processing {n_docs} documents for outlier scoring")
        
        # Build citation network
        G = self._build_citation_network(simulation_df, dataset_name)
        logger.info(f"Built citation network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Load embeddings if available
        if dataset_name:
            self._load_and_prepare_embeddings(dataset_name)
        
        # Extract comprehensive features
        features_df = self._extract_network_features(G, simulation_df)
        logger.info(f"Extracted features: {features_df.shape[1]-1} features for {features_df.shape[0]} documents")
        
        # Get feature matrix (excluding document ID)
        feature_matrix = features_df.drop('openalex_id', axis=1).values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        features_standard = self.standard_scaler.fit_transform(feature_matrix)
        
        # Apply outlier detection methods
        outlier_results = {}
        
        # 1. LOF on embeddings for semantic outlier detection
        logger.info("Applying LOF on embeddings for semantic outlier detection...")
        lof_results = self._apply_lof_to_embeddings(simulation_df)
        outlier_results['lof_scores'] = lof_results['scores']
        
        # 2. Isolation Forest
        logger.info("Applying Isolation Forest...")
        isolation_forest = IsolationForest(
            n_estimators=100,
            contamination='auto',  # Use auto for scoring without predefined thresholds
            random_state=self.random_state,
            n_jobs=-1
        )
        isolation_forest.fit(features_standard)
        if_scores = isolation_forest.decision_function(features_standard)
        outlier_results['isolation_forest_scores'] = -if_scores  # Convert to anomaly scores
        
        # 3. DBSCAN with automatic eps tuning
        logger.info("Applying DBSCAN with automatic eps tuning...")
        dbscan_results = self._apply_dbscan_with_auto_eps(features_standard, n_docs)
        outlier_results['dbscan_scores'] = dbscan_results['scores']
        
        # 4. Ensemble scoring
        logger.info("Computing ensemble scores...")
        ensemble_scores = self._compute_ensemble_scores(outlier_results)
        outlier_results['ensemble_scores'] = ensemble_scores
        
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
        logger.info(f"Outlier scoring completed in {total_time:.2f}s")
        logger.info(f"Generated outlier scores for all {len(simulation_df)} documents using ensemble method")
        
        return outlier_results
        
    def _build_citation_network(self, simulation_df: pd.DataFrame, dataset_name: str = None) -> nx.DiGraph:
        """Build citation network from simulation data using Synergy dataset for citations."""
        G = nx.DiGraph()
        
        # Add all documents as nodes
        for _, row in simulation_df.iterrows():
            G.add_node(row['openalex_id'], 
                      title=row.get('title', ''),
                      year=row.get('year', 0),
                      label=row.get('label_included', 0))
        
        logger.info(f"Created graph with {len(G.nodes)} nodes")
        
        # Add citation edges from synergy dataset
        if dataset_name:
            synergy_data = load_synergy_dataset(dataset_name)
            citation_count = self._add_citation_edges(G, simulation_df, synergy_data)
            logger.info(f"Added {citation_count} citation edges")
        
        return G
    
    def _add_citation_edges(self, G: nx.DiGraph, simulation_df: pd.DataFrame, synergy_data: Optional[Dict]) -> int:
        """Add direct citation edges from synergy dataset."""
        if synergy_data is None:
            logger.warning("No synergy data available - cannot add citation edges")
            return 0
        
        simulation_ids = set(simulation_df['openalex_id'].tolist())
        citation_count = 0
        
        for _, row in tqdm(simulation_df.iterrows(), desc="Building citation network", total=len(simulation_df)):
            citing_paper = row['openalex_id']
            
            if citing_paper in synergy_data:
                referenced_works = synergy_data[citing_paper].get('referenced_works', [])
                
                if referenced_works and isinstance(referenced_works, list):
                    for ref_id in referenced_works:
                        if ref_id in simulation_ids and ref_id != citing_paper and ref_id in G.nodes:
                            if G.has_edge(citing_paper, ref_id):
                                G[citing_paper][ref_id]['weight'] += 1
                            else:
                                G.add_edge(citing_paper, ref_id, 
                                         edge_type='citation', 
                                         weight=2.0)
                            citation_count += 1
        
        return citation_count
    
    def _load_and_prepare_embeddings(self, dataset_name: str):
        """Load and prepare embeddings with optional UMAP reduction."""
        self.embeddings, self.embeddings_metadata = load_embeddings(dataset_name)
        
        if self.embeddings is not None:
            logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            
            # Apply UMAP dimensionality reduction if enabled
            if self.use_umap and self.embeddings.shape[1] > self.umap_components:
                try:
                    logger.info(f"Reducing embeddings from {self.embeddings.shape[1]} to {self.umap_components} dimensions using UMAP")
                    
                    self.umap_reducer = umap.UMAP(
                        n_components=self.umap_components,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='cosine',
                        random_state=self.random_state,
                        verbose=False
                    )
                    
                    self.reduced_embeddings = self.umap_reducer.fit_transform(self.embeddings)
                    logger.info(f"UMAP reduction completed: {self.reduced_embeddings.shape}")
                    
                except Exception as e:
                    logger.warning(f"UMAP reduction failed: {e}. Using original embeddings.")
                    self.reduced_embeddings = self.embeddings
            else:
                self.reduced_embeddings = self.embeddings
    
    def _apply_lof_to_embeddings(self, simulation_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Apply LOF directly to embeddings for semantic outlier detection."""
        
        if self.reduced_embeddings is None or self.embeddings_metadata is None:
            logger.warning("No embeddings available for LOF analysis")
            n_docs = len(simulation_df)
            return {
                'scores': np.zeros(n_docs)
            }
        
        # Create mapping from openalex_id to embedding index
        if 'documents' in self.embeddings_metadata:
            id_to_idx = {doc.get('openalex_id', ''): idx for idx, doc in enumerate(self.embeddings_metadata['documents'])}
        else:
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata.get('openalex_id', []))}
        
        # Get embeddings for documents in simulation_df
        doc_embeddings = []
        doc_indices = []
        
        for idx, row in simulation_df.iterrows():
            doc_id = row['openalex_id']
            if doc_id in id_to_idx:
                embedding_idx = id_to_idx[doc_id]
                doc_embeddings.append(self.reduced_embeddings[embedding_idx])
                doc_indices.append(idx)
        
        if len(doc_embeddings) == 0:
            logger.warning("No embeddings found for simulation documents")
            n_docs = len(simulation_df)
            return {
                'scores': np.zeros(n_docs)
            }
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Configure LOF for embedding space
        n_neighbors = min(20, max(5, len(doc_embeddings) // 10))
        metric = 'euclidean' if self.use_umap else 'cosine'
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=False,
            metric=metric,
            n_jobs=-1
        )
        
        logger.info(f"Applying LOF to embeddings: {doc_embeddings.shape} with {n_neighbors} neighbors, metric={metric}")
        
        # Apply LOF to get scores only (no predictions with contamination)
        lof.fit_predict(doc_embeddings)
        scores = -lof.negative_outlier_factor_
        
        # Map results back to full simulation dataframe
        n_docs = len(simulation_df)
        full_scores = np.zeros(n_docs)
        
        for i, sim_idx in enumerate(doc_indices):
            full_scores[sim_idx] = scores[i]
        
        return {
            'scores': full_scores
        }
    
    def _apply_dbscan_with_auto_eps(self, features_standard: np.ndarray, n_docs: int) -> Dict[str, np.ndarray]:
        """Apply DBSCAN with automatic eps tuning using k-distance and knee detection."""
        
        # Determine min_samples based on dimensionality and dataset size
        n_features = features_standard.shape[1]
        
        if n_docs < 100:
            min_samples = max(2, int(1.5 * n_features))
        elif n_docs < 500:
            min_samples = max(8, int(1.8 * n_features))
        elif n_docs < 2000:
            min_samples = max(40, int(2.0 * n_features))
        else:
            min_samples = max(50, int(2.2 * n_features))
        
        # Find optimal eps using k-distance with automatic knee detection
        optimal_eps = self._find_optimal_eps_auto(features_standard, min_samples)
        
        # Apply DBSCAN
        dbscan = DBSCAN(
            eps=optimal_eps,
            min_samples=min_samples,
            metric='euclidean',
            n_jobs=-1
        )
        
        logger.info(f"DBSCAN parameters: eps={optimal_eps:.4f}, min_samples={min_samples}")
        
        labels = dbscan.fit_predict(features_standard)
        scores = self._compute_dbscan_scores(features_standard, labels)
        
        # Log DBSCAN performance
        outlier_pct = (labels == -1).sum() / len(labels) * 100
        logger.info(f"DBSCAN flagged {outlier_pct:.1f}% as outliers")
        
        return {
            'scores': scores
        }
    
    def _find_optimal_eps_auto(self, features: np.ndarray, k: int) -> float:
        """Find optimal eps parameter using k-distance with automatic knee detection."""
        
        # Compute k-nearest neighbors
        logger.info(f"Computing {k}-distance graph for eps optimization...")
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
        nbrs.fit(features)
        
        # Get distances to k-th nearest neighbor for all points
        distances, _ = nbrs.kneighbors(features)
        k_distances = distances[:, k-1]  # Distance to k-th neighbor (0-indexed)
        k_distances_sorted = np.sort(k_distances)
        
        # Use kneed library for automatic knee detection if available
        if KNEED_AVAILABLE:
            try:
                # Create x-axis (point indices)
                x = np.arange(len(k_distances_sorted))
                
                # Use KneeLocator to find the knee point
                knee_locator = KneeLocator(
                    x, k_distances_sorted,
                    curve='convex',  # k-distance curve is typically convex
                    direction='increasing',  # distances are sorted in ascending order
                    S=1.0  # sensitivity parameter
                )
                
                if knee_locator.knee is not None:
                    optimal_eps = k_distances_sorted[knee_locator.knee]
                    logger.info(f"Knee detection successful: eps={optimal_eps:.4f} at index {knee_locator.knee}")
                else:
                    # Fallback to percentile method
                    optimal_eps = np.percentile(k_distances_sorted, 95)
                    logger.warning(f"Knee detection failed, using 95th percentile: eps={optimal_eps:.4f}")
                    
            except Exception as e:
                logger.warning(f"Kneed library failed: {e}. Using fallback method.")
                optimal_eps = self._detect_knee_fallback(k_distances_sorted)
        else:
            optimal_eps = self._detect_knee_fallback(k_distances_sorted)
        
        # Apply bounds check
        min_eps = np.percentile(k_distances_sorted, 1)
        max_eps = np.percentile(k_distances_sorted, 95)
        optimal_eps = max(min_eps, min(optimal_eps, max_eps))
        
        return optimal_eps
    
    def _detect_knee_fallback(self, distances: np.ndarray) -> float:
        """Fallback knee detection method when kneed library is not available."""
        n_points = len(distances)
        
        if n_points < 10:
            return np.median(distances)
        
        # Create x-axis (point indices normalized to [0,1])
        x = np.arange(n_points) / (n_points - 1)
        y_norm = (distances - distances.min()) / (distances.max() - distances.min()) if distances.max() > distances.min() else distances
        
        # Find point with maximum distance from line connecting start and end points
        start_point = np.array([x[0], y_norm[0]])
        end_point = np.array([x[-1], y_norm[-1]])
        
        max_distance = 0
        knee_idx = 0
        
        for i in range(1, n_points - 1):
            point = np.array([x[i], y_norm[i]])
            distance = self._point_to_line_distance(point, start_point, end_point)
            
            if distance > max_distance:
                max_distance = distance
                knee_idx = i
        
        return distances[knee_idx]
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj = proj_length * line_unitvec
        perp_vec = point_vec - proj
        
        return np.linalg.norm(perp_vec)
    
    def _extract_network_features(self, G: nx.DiGraph, simulation_df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive network and semantic features."""
        features = []
        relevant_docs = set(simulation_df[simulation_df['label_included'] == 1]['openalex_id'].tolist())
        
        # Pre-compute centrality measures for efficiency
        logger.info("Computing graph centrality measures...")
        if len(G.nodes) > 0:
            try:
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
                in_degree = G.in_degree(doc_id)
                out_degree = G.out_degree(doc_id)
                total_degree = in_degree + out_degree
                
                # Get neighbors
                neighbors = set()
                neighbors.update(G.predecessors(doc_id))
                neighbors.update(G.successors(doc_id))
                neighbors = list(neighbors)
                
                relevant_neighbors = len([n for n in neighbors if n in relevant_docs])
                
                doc_features.update({
                    'degree': float(total_degree),
                    'in_degree': float(in_degree),
                    'out_degree': float(out_degree),
                    'relevant_neighbors': float(relevant_neighbors),
                    'relevant_ratio': float(relevant_neighbors / max(1, len(neighbors))),
                    'clustering': float(nx.clustering(G.to_undirected(), doc_id)),
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
                    'degree': 0.0, 'in_degree': 0.0, 'out_degree': 0.0,
                    'relevant_neighbors': 0.0, 'relevant_ratio': 0.0,
                    'clustering': 0.0, 'pagerank': 0.0, 'betweenness': 0.0,
                    'closeness': 0.0, 'eigenvector': 0.0,
                    'mean_neighbor_degree': 0.0, 'std_neighbor_degree': 0.0,
                    'max_neighbor_degree': 0.0, 'min_distance_to_relevant': 10.0,
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
        
        # Create mapping from openalex_id to embedding index
        if 'documents' in self.embeddings_metadata:
            id_to_idx = {doc.get('openalex_id', ''): idx for idx, doc in enumerate(self.embeddings_metadata['documents'])}
        else:
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata.get('openalex_id', []))}
        
        if doc_id not in id_to_idx:
            return features
        
        try:
            # Get document embedding
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
        return {
            'publication_year': float(row.get('year', 0)),
            'title_length': float(len(str(row.get('title', '')))),
        }
    
    def _compute_dbscan_scores(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute outlier scores for DBSCAN (distance to nearest cluster center)."""
        scores = np.zeros(len(features))
        
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if not unique_labels:
            return np.ones(len(features))
        
        # Compute cluster centers
        cluster_centers = {}
        for label in unique_labels:
            cluster_points = features[labels == label]
            cluster_centers[label] = np.mean(cluster_points, axis=0)
        
        # Compute scores
        for i, (point, label) in enumerate(zip(features, labels)):
            if label == -1:  # Noise point
                min_distance = min(np.linalg.norm(point - center) for center in cluster_centers.values())
                scores[i] = min_distance
            else:  # Cluster point
                center = cluster_centers[label]
                scores[i] = np.linalg.norm(point - center)
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _compute_ensemble_scores(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute ensemble scores with data-driven weighting."""
        # Normalize scores
        lof_scores_norm = normalize_scores(results['lof_scores'])
        if_scores_norm = normalize_scores(results['isolation_forest_scores'])
        dbscan_scores_norm = normalize_scores(results['dbscan_scores'])
        
        # Calculate variance-based weights
        score_arrays = {
            'lof': lof_scores_norm,
            'isolation_forest': if_scores_norm,
            'dbscan': dbscan_scores_norm,
        }
        
        weights = compute_ensemble_weights(score_arrays, method='variance')
        
        # Ensure LOF gets significant weight for semantic detection
        if weights['lof'] < 0.4:
            weights['lof'] = 0.5
            remaining = 0.5
            weights['isolation_forest'] = remaining * 0.7
            weights['dbscan'] = remaining * 0.3
        
        logger.info(f"Ensemble weights: LOF={weights['lof']:.3f}, IF={weights['isolation_forest']:.3f}, DBSCAN={weights['dbscan']:.3f}")
        
        # Compute weighted ensemble
        ensemble_scores = (
            weights['lof'] * lof_scores_norm +
            weights['isolation_forest'] * if_scores_norm +
            weights['dbscan'] * dbscan_scores_norm
        )
        
        return ensemble_scores
    
    def get_outlier_documents(self, 
                            method: str = 'ensemble',
                            top_k: int = None) -> pd.DataFrame:
        """Get outlier documents with detailed information including individual sub-model scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting outlier documents")
        
        score_key = f'{method}_scores'
        
        if score_key not in self.outlier_results:
            raise ValueError(f"Unknown method: {method}")
        
        scores = self.outlier_results[score_key]
        doc_ids = self.outlier_results['openalex_ids']
        
        # Get individual sub-model scores for all documents
        lof_scores = self.outlier_results.get('lof_scores', np.zeros(len(doc_ids)))
        isolation_forest_scores = self.outlier_results.get('isolation_forest_scores', np.zeros(len(doc_ids)))
        dbscan_scores = self.outlier_results.get('dbscan_scores', np.zeros(len(doc_ids)))
        ensemble_scores = self.outlier_results.get('ensemble_scores', np.zeros(len(doc_ids)))
        
        # Create results DataFrame - rank by score for top_k selection
        score_rank_pairs = list(zip(scores, range(len(scores))))
        score_rank_pairs.sort(key=lambda x: x[0], reverse=True)  # Sort by score, highest first
        
        results = []
        for rank, (score, i) in enumerate(score_rank_pairs):
            if top_k is None or rank < top_k:
                doc_id = doc_ids[i]
                doc_features = self.features_df[self.features_df['openalex_id'] == doc_id].iloc[0]
                
                result = {
                    'document_id': doc_id,
                    'outlier_score': float(score),
                    'rank': rank + 1,
                    # Individual sub-model scores
                    'lof_score': float(lof_scores[i]),
                    'isolation_forest_score': float(isolation_forest_scores[i]),
                    'dbscan_score': float(dbscan_scores[i]),
                    'ensemble_score': float(ensemble_scores[i]),
                    # Graph features
                    'degree': doc_features['degree'],
                    'relevant_neighbors': doc_features['relevant_neighbors'],
                    'relevant_ratio': doc_features['relevant_ratio'],
                    'clustering_coefficient': doc_features['clustering'],
                    'pagerank': doc_features['pagerank'],
                    'semantic_similarity': doc_features.get('semantic_similarity_to_relevant', 0.0),
                    'min_distance_to_relevant': doc_features['min_distance_to_relevant'],
                }
                results.append(result)
        
        results_df = pd.DataFrame(results).sort_values('outlier_score', ascending=False)
        
        if top_k is not None:
            results_df = results_df.head(top_k)
        
        return results_df
    
    def get_method_comparison(self) -> pd.DataFrame:
        """Compare results across all outlier detection methods."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before comparing methods")
        
        methods = ['lof', 'isolation_forest', 'dbscan', 'ensemble']
        comparison = []
        
        for method in methods:
            score_key = f'{method}_scores'
            
            if score_key not in self.outlier_results:
                continue
                
            scores = self.outlier_results[score_key]
            
            method_names = {
                'lof': 'LOF (Embeddings)',
                'isolation_forest': 'Isolation Forest',
                'dbscan': 'DBSCAN',
                'ensemble': 'Ensemble'
            }
            
            comparison.append({
                'method': method_names.get(method, method),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'score_range': np.max(scores) - np.min(scores),
            })
        
        return pd.DataFrame(comparison)
    
    def get_detailed_outlier_breakdown(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get a detailed breakdown of outlier scores showing all sub-model contributions.
        
        Args:
            top_k: Number of top outliers to show
            
        Returns:
            DataFrame with detailed score breakdown
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting outlier breakdown")
        
        doc_ids = self.outlier_results['openalex_ids']
        lof_scores = self.outlier_results.get('lof_scores', np.zeros(len(doc_ids)))
        isolation_forest_scores = self.outlier_results.get('isolation_forest_scores', np.zeros(len(doc_ids)))
        dbscan_scores = self.outlier_results.get('dbscan_scores', np.zeros(len(doc_ids)))
        ensemble_scores = self.outlier_results.get('ensemble_scores', np.zeros(len(doc_ids)))
        
        # Create detailed breakdown
        breakdown = []
        for i, doc_id in enumerate(doc_ids):
            breakdown.append({
                'document_id': doc_id,
                'lof_score': float(lof_scores[i]),
                'isolation_forest_score': float(isolation_forest_scores[i]),
                'dbscan_score': float(dbscan_scores[i]),
                'ensemble_score': float(ensemble_scores[i]),
                'rank_by_ensemble': 0  # Will be filled after sorting
            })
        
        # Convert to DataFrame and sort by ensemble score
        breakdown_df = pd.DataFrame(breakdown)
        breakdown_df = breakdown_df.sort_values('ensemble_score', ascending=False)
        breakdown_df['rank_by_ensemble'] = range(1, len(breakdown_df) + 1)
        
        # Return top_k results
        return breakdown_df.head(top_k)
    
    def print_outlier_score_summary(self, top_k: int = 10):
        """
        Print a formatted summary of outlier scores for easy reading.
        
        Args:
            top_k: Number of top outliers to display
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before printing summary")
        
        breakdown_df = self.get_detailed_outlier_breakdown(top_k)
        
        print(f"\n" + "=" * 80)
        print(f"TOP {top_k} OUTLIERS - DETAILED SCORE BREAKDOWN")
        print("=" * 80)
        print(f"{'Rank':<5} {'Document ID':<20} {'LOF':<8} {'IsolFor':<8} {'DBSCAN':<8} {'Ensemble':<10}")
        print("-" * 80)
        
        for _, row in breakdown_df.iterrows():
            print(f"{row['rank_by_ensemble']:<5} "
                  f"{row['document_id']:<20} "
                  f"{row['lof_score']:<8.4f} "
                  f"{row['isolation_forest_score']:<8.4f} "
                  f"{row['dbscan_score']:<8.4f} "
                  f"{row['ensemble_score']:<10.4f}")
        
        print("=" * 80)
        print("LOF = Local Outlier Factor (Embeddings)")
        print("IsolFor = Isolation Forest") 
        print("DBSCAN = Density-Based Clustering")
        print("Ensemble = Weighted combination of all methods")
        print("=" * 80)


def main():
    """Standalone Citation Network Outlier Detection."""
    print("=" * 60)
    print("STREAMLINED CITATION NETWORK OUTLIER DETECTION")
    print("=" * 60)
    
    try:
        # Dataset Selection
        print("\nStep 1: Dataset Selection")
        dataset_name = prompt_dataset_selection()
        
        # Load Data
        print(f"\nStep 2: Loading dataset '{dataset_name}'...")
        simulation_df = load_simulation_data(dataset_name)
        print(f"Loaded {len(simulation_df)} documents")
        
        # Initialize and Run Model
        print(f"\nStep 3: Running Citation Network Outlier Detection...")
        print("Methods: LOF (embeddings), Isolation Forest, DBSCAN with auto eps tuning")
        
        detector = CitationNetworkOutlierDetector(
            use_umap=False,
            umap_components=50,
            random_state=42
        )
        
        start_time = time.time()
        results = detector.fit_predict_outliers(simulation_df, dataset_name=dataset_name)
        runtime = time.time() - start_time
        
        print(f"Model completed in {runtime:.2f} seconds")
        
        # Evaluate Known Outliers
        print(f"\n" + "=" * 50)
        print("OUTLIER RANKING PERFORMANCE")
        print("=" * 50)
        
        # Create score dictionary for ranking
        doc_ids = results['openalex_ids']
        ensemble_scores = results['ensemble_scores']
        scores_dict = dict(zip(doc_ids, ensemble_scores))
        
        # Load datasets config for outlier evaluation
        datasets_config = load_datasets_config()
        outlier_ranking_results = evaluate_outlier_ranking(scores_dict, dataset_name, datasets_config)
        
        if outlier_ranking_results:
            for result in outlier_ranking_results:
                print(f"\nKnown Outlier: {result['outlier_id']}")
                print(f"  Rank: {result['rank']} out of {result['total_documents']}")
                print(f"  Ensemble Score: {result['score']:.4f}")
                print(f"  Percentile: {result['percentile']:.1f}%")
                
                if result['percentile'] >= 95:
                    performance = "Excellent ✓"
                elif result['percentile'] >= 90:
                    performance = "Very Good ✓"
                elif result['percentile'] >= 80:
                    performance = "Good"
                else:
                    performance = "Fair/Poor"
                print(f"  Performance: {performance}")
        else:
            print("No known outliers defined for this dataset.")
        
        # Show Top Documents with Detailed Scores
        print(f"\n" + "=" * 50)
        print("TOP OUTLIER DOCUMENTS - DETAILED BREAKDOWN")
        print("=" * 50)
        
        # Use the new detailed breakdown method
        detector.print_outlier_score_summary(top_k=10)
        
        # Method Comparison
        print(f"\n" + "=" * 50)
        print("METHOD COMPARISON")
        print("=" * 50)
        
        comparison = detector.get_method_comparison()
        print(f"{'Method':<20} {'Mean Score':<12} {'Std Score':<12} {'Score Range':<12}")
        print("-" * 60)
        for _, row in comparison.iterrows():
            print(f"{row['method']:<20} {row['mean_score']:<12.4f} {row['std_score']:<12.4f} {row['score_range']:<12.4f}")
        
        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 