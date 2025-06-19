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
import umap

# Graph analysis
import networkx as nx
from scipy.spatial.distance import cosine
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Add models directory to path to import synergy_dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
models_dir = os.path.join(project_root, 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

try:
    from synergy_dataset import Dataset
except ImportError:
    logger.warning("synergy_dataset not available - citation edges will not be added")
    Dataset = None

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
                 use_umap: bool = False,
                 umap_components: int = 50,
                 random_state: int = 42):
        """
        Initialize the outlier detector.
        
        Args:
            contamination: Initial contamination value (will be dynamically calculated as 1/n_docs during fit_predict_outliers)
            enable_semantic: Whether to use semantic embeddings if available
            use_umap: Whether to use UMAP for dimensionality reduction of embeddings
            umap_components: Number of UMAP components to reduce embeddings to
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.enable_semantic = enable_semantic
        self.use_umap = use_umap
        self.umap_components = umap_components
        self.random_state = random_state
        
        # Initialize outlier detection algorithms
        self.lof = LocalOutlierFactor(
            n_neighbors=50,  # Increased for more stable density estimation
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
        
        # DBSCAN parameters will be set dynamically based on dataset size
        self.dbscan = None
        
        # Scalers for different feature types
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # State variables
        self.is_fitted = False
        self.feature_names = []
        self.embeddings = None
        self.embeddings_metadata = None
        self.umap_reducer = None
        self.reduced_embeddings = None
        
    def fit_predict_outliers(self, 
                           simulation_df: pd.DataFrame,
                           dataset_name: str = None) -> Dict[str, np.ndarray]:
        """
        Main method: takes dataset and immediately returns outlier scores.
        No training phase - everything happens on-the-fly.
        
        Args:
            simulation_df: DataFrame with paper data and citation information
            dataset_name: Optional dataset name for loading embeddings and synergy data
            
        Returns:
            Dictionary with outlier scores and predictions from all methods
        """
        logger.info(f"Starting on-the-fly outlier detection for {len(simulation_df)} documents")
        start_time = time.time()
        
        # Store dataset name for synergy data loading
        self._current_dataset_name = dataset_name
        
        # Calculate dynamic contamination for RANKING - more liberal than binary classification
        n_docs = len(simulation_df)
        # For ranking: use higher contamination to ensure good recall at top ranks
        dynamic_contamination = max(0.15, min(0.25, 10.0 / n_docs))  # 15-25% range for ranking
        logger.info(f"Using dynamic contamination for ranking: {dynamic_contamination:.4f}")
        
        # Update contamination for all algorithms
        self.lof.contamination = dynamic_contamination
        self.isolation_forest.contamination = dynamic_contamination
        self.one_class_svm.nu = dynamic_contamination
        
        # FIXED: Better DBSCAN parameter selection using k-distance method principles
        # Use much smaller eps values and data-driven min_samples
        if n_docs < 100:
            eps, min_samples = 0.5, max(3, int(np.log2(n_docs)))
        elif n_docs < 500:
            eps, min_samples = 0.7, max(5, int(np.log2(n_docs)))
        elif n_docs < 2000:
            eps, min_samples = 0.9, max(8, int(np.log2(n_docs)))
        else:
            # For large datasets: much more conservative eps, higher min_samples
            eps, min_samples = 1.2, max(12, int(np.log2(n_docs) * 1.5))
        
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean',
            n_jobs=-1
        )
        logger.info(f"DBSCAN parameters: eps={eps}, min_samples={min_samples}")
        
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
        
        # 1. Local Outlier Factor (LOF) on mixed features
        logger.info("Applying Local Outlier Factor (LOF) on mixed features...")
        lof_scores = self.lof.fit_predict(features_standard)
        lof_negative_scores = self.lof.negative_outlier_factor_
        outlier_results['lof_mixed_predictions'] = lof_scores
        outlier_results['lof_mixed_scores'] = -lof_negative_scores  # Convert to positive scores
        
        # 2. Local Outlier Factor (LOF) on embeddings for subtopic detection
        logger.info("Applying Local Outlier Factor (LOF) directly on embeddings...")
        embedding_lof_results = self._apply_lof_to_embeddings(simulation_df, dynamic_contamination)
        outlier_results['lof_predictions'] = embedding_lof_results['embedding_lof_predictions']
        outlier_results['lof_scores'] = embedding_lof_results['embedding_lof_scores']
        
        # 3. Isolation Forest
        logger.info("Applying Isolation Forest...")
        if_predictions = self.isolation_forest.fit_predict(features_standard)
        if_scores = self.isolation_forest.decision_function(features_standard)
        outlier_results['isolation_forest_predictions'] = if_predictions
        outlier_results['isolation_forest_scores'] = -if_scores  # Convert to anomaly scores
        
        # 4. One-Class SVM
        logger.info("Applying One-Class SVM...")
        svm_predictions = self.one_class_svm.fit_predict(features_robust)
        svm_scores = self.one_class_svm.decision_function(features_robust)
        outlier_results['one_class_svm_predictions'] = svm_predictions
        outlier_results['one_class_svm_scores'] = -svm_scores  # Convert to anomaly scores
        
        # 5. DBSCAN (cluster-based outliers)
        logger.info("Applying DBSCAN clustering...")
        dbscan_labels = self.dbscan.fit_predict(features_standard)
        dbscan_predictions = (dbscan_labels == -1).astype(int) * 2 - 1  # Convert noise to -1, clusters to 1
        # For DBSCAN scores, use distance to nearest cluster center
        dbscan_scores = self._compute_dbscan_scores(features_standard, dbscan_labels)
        outlier_results['dbscan_predictions'] = dbscan_predictions
        outlier_results['dbscan_scores'] = dbscan_scores
        
        # Log DBSCAN performance
        dbscan_outlier_pct = (dbscan_labels == -1).sum() / len(dbscan_labels) * 100
        logger.info(f"DBSCAN flagged {dbscan_outlier_pct:.1f}% as outliers")
        
        # 6. Ensemble scoring with improved methods
        logger.info("Computing ensemble scores...")
        ensemble_scores = self._compute_ensemble_scores(outlier_results)
        outlier_results['ensemble_scores'] = ensemble_scores
        outlier_results['ensemble_predictions'] = (ensemble_scores > np.percentile(ensemble_scores, 100 * (1 - dynamic_contamination))).astype(int) * 2 - 1
        
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
        """Build citation network from simulation data using Synergy dataset for citations."""
        G = nx.DiGraph()  # Use directed graph for proper citation representation
        
        # Add all documents as nodes
        for _, row in simulation_df.iterrows():
            G.add_node(row['openalex_id'], 
                      title=row.get('title', ''),
                      year=row.get('year', 0),
                      label=row.get('label_included', 0))
        
        logger.info(f"Created graph with {len(G.nodes)} nodes")
        
        # Load synergy dataset for citation information
        synergy_data = self._load_synergy_dataset()
        
        # Add citation edges from synergy dataset
        citation_count = self._add_citation_edges(G, simulation_df, synergy_data)
        
        logger.info(f"Added {citation_count} citation edges")
        return G
    
    def _load_synergy_dataset(self) -> Optional[Dict]:
        """Load synergy dataset configuration and data."""
        if Dataset is None:
            logger.warning("synergy_dataset package not available")
            return None
            
        try:
            # Load datasets configuration to get synergy dataset name
            project_root = self._get_project_root()
            config_path = os.path.join(project_root, 'data', 'datasets.json')
            
            if not os.path.exists(config_path):
                logger.error(f"Datasets configuration not found: {config_path}")
                return None
                
            with open(config_path, 'r') as f:
                datasets_config = json.load(f)
            
            # Get the dataset name that was passed to fit_predict_outliers
            dataset_name = getattr(self, '_current_dataset_name', None)
            if not dataset_name:
                logger.warning("No dataset name available for synergy data loading")
                return None
                
            if dataset_name not in datasets_config:
                logger.warning(f"Dataset '{dataset_name}' not found in configuration")
                return None
                
            synergy_name = datasets_config[dataset_name]['synergy_dataset_name']
            logger.info(f"Loading Synergy dataset: {synergy_name}")
            
            # Load the synergy dataset
            dataset = Dataset(synergy_name)
            synergy_data = dataset.to_dict(["id", "title", "referenced_works"])
            
            logger.info(f"Loaded {len(synergy_data)} documents from Synergy dataset")
            return synergy_data
            
        except Exception as e:
            logger.error(f"Error loading synergy dataset: {e}")
            return None
    
    def _add_citation_edges(self, G: nx.DiGraph, simulation_df: pd.DataFrame, synergy_data: Optional[Dict]) -> int:
        """Add direct citation edges from synergy dataset."""
        if synergy_data is None:
            logger.warning("No synergy data available - cannot add citation edges")
            return 0
        
        # Create mapping from openalex_id in simulation to check what documents we have
        simulation_ids = set(simulation_df['openalex_id'].tolist())
        
        citation_count = 0
        total_papers = len(simulation_df)
        processed_papers = 0
        
        for _, row in tqdm(simulation_df.iterrows(), desc="Building citation network", total=len(simulation_df)):
            citing_paper = row['openalex_id']
            
            # Get referenced works from synergy data
            if citing_paper in synergy_data:
                referenced_works = synergy_data[citing_paper].get('referenced_works', [])
                
                if referenced_works and isinstance(referenced_works, list):
                    for ref_id in referenced_works:
                        # Check if the referenced paper is in our simulation dataset
                        if ref_id in simulation_ids and ref_id != citing_paper and ref_id in G.nodes:
                            # Add directed edge from citing to cited paper
                            if G.has_edge(citing_paper, ref_id):
                                # Increase weight if edge already exists
                                G[citing_paper][ref_id]['weight'] += 1
                                G[citing_paper][ref_id]['citation_count'] += 1
                            else:
                                # Add new citation edge
                                G.add_edge(citing_paper, ref_id, 
                                         edge_type='citation', 
                                         weight=2.0,  # Higher weight for direct citations
                                         citation_count=1)
                            citation_count += 1
            
            processed_papers += 1
            if processed_papers % 500 == 0:
                logger.info(f"Processed {processed_papers}/{total_papers} papers, found {citation_count} citation edges")
        
        return citation_count
    
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
                
                # Apply UMAP dimensionality reduction if enabled
                self._prepare_embeddings_for_outlier_detection()
                
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self.embeddings = None
                self.embeddings_metadata = None
        else:
            logger.info(f"No embeddings found for {dataset_name}")
    
    def _prepare_embeddings_for_outlier_detection(self):
        """Prepare embeddings for outlier detection with optional UMAP reduction."""
        if self.embeddings is None:
            return
            
        if self.use_umap and self.embeddings.shape[1] > self.umap_components:
            try:
                logger.info(f"Reducing embeddings from {self.embeddings.shape[1]} to {self.umap_components} dimensions using UMAP")
                
                # Configure UMAP for outlier detection
                self.umap_reducer = umap.UMAP(
                    n_components=self.umap_components,
                    n_neighbors=15,  # Preserve local structure
                    min_dist=0.1,   # Allow for tighter clusters
                    metric='cosine', # Better for high-dimensional embeddings
                    random_state=self.random_state,
                    verbose=False
                )
                
                self.reduced_embeddings = self.umap_reducer.fit_transform(self.embeddings)
                logger.info(f"UMAP reduction completed: {self.reduced_embeddings.shape}")
                
            except Exception as e:
                logger.warning(f"UMAP reduction failed: {e}. Using original embeddings.")
                self.reduced_embeddings = self.embeddings
                self.umap_reducer = None
        else:
            logger.info("Using original embeddings (UMAP disabled or unnecessary)")
            self.reduced_embeddings = self.embeddings
    
    def _apply_lof_to_embeddings(self, simulation_df: pd.DataFrame, dynamic_contamination: float) -> Dict[str, np.ndarray]:
        """Apply LOF directly to embeddings for subtopic outlier detection."""
        
        if self.reduced_embeddings is None or self.embeddings_metadata is None:
            logger.warning("No embeddings available for LOF analysis")
            n_docs = len(simulation_df)
            return {
                'embedding_lof_scores': np.zeros(n_docs),
                'embedding_lof_predictions': np.ones(n_docs)  # All normal
            }
        
        # Create mapping from openalex_id to embedding index
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata['openalex_id'])}
        
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
                'embedding_lof_scores': np.zeros(n_docs),
                'embedding_lof_predictions': np.ones(n_docs)
            }
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Configure LOF for embedding space
        n_neighbors = min(50, max(10, len(doc_embeddings) // 20))  # More conservative neighborhood
        
        # Use cosine similarity for high-dimensional embeddings unless UMAP reduced
        metric = 'euclidean' if self.use_umap else 'cosine'
        
        embedding_lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=dynamic_contamination,
            metric=metric,
            novelty=False,
            n_jobs=-1
        )
        
        logger.info(f"Applying LOF to embeddings: {doc_embeddings.shape} with {n_neighbors} neighbors, metric={metric}")
        
        # Apply LOF
        embedding_predictions = embedding_lof.fit_predict(doc_embeddings)
        embedding_scores = -embedding_lof.negative_outlier_factor_
        
        # Map results back to full simulation dataframe
        n_docs = len(simulation_df)
        full_scores = np.zeros(n_docs)
        full_predictions = np.ones(n_docs)
        
        for i, sim_idx in enumerate(doc_indices):
            full_scores[sim_idx] = embedding_scores[i]
            full_predictions[sim_idx] = embedding_predictions[i]
        
        return {
            'embedding_lof_scores': full_scores,
            'embedding_lof_predictions': full_predictions
        }
    
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
                # Basic connectivity features for directed graph
                in_degree = G.in_degree(doc_id) if hasattr(G, 'in_degree') else G.degree(doc_id)
                out_degree = G.out_degree(doc_id) if hasattr(G, 'out_degree') else G.degree(doc_id)
                total_degree = in_degree + out_degree
                
                # Get all neighbors (both incoming and outgoing)
                neighbors = set()
                if hasattr(G, 'predecessors'):
                    neighbors.update(G.predecessors(doc_id))
                if hasattr(G, 'successors'):
                    neighbors.update(G.successors(doc_id))
                neighbors = list(neighbors)
                
                relevant_neighbors = len([n for n in neighbors if n in relevant_docs])
                
                doc_features.update({
                    'degree': float(total_degree),
                    'in_degree': float(in_degree),
                    'out_degree': float(out_degree),
                    'relevant_neighbors': float(relevant_neighbors),
                    'relevant_ratio': float(relevant_neighbors / max(1, len(neighbors))),
                    'clustering': float(nx.clustering(G.to_undirected(), doc_id) if hasattr(G, 'to_undirected') else nx.clustering(G, doc_id)),
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
                    'in_degree': 0.0,
                    'out_degree': 0.0,
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
    
    def _extract_references(self, row: pd.Series) -> List[str]:
        """Extract reference information from paper metadata."""
        references = []
        
        # Check common fields that might contain reference information
        possible_ref_fields = [
            'referenced_works', 'references', 'cited_by', 'bibliography',
            'reference_list', 'citations', 'refs'
        ]
        
        for field in possible_ref_fields:
            if field in row and pd.notna(row[field]):
                ref_data = row[field]
                
                if isinstance(ref_data, str):
                    # Try to parse as JSON if it looks like a JSON string
                    if ref_data.startswith('[') or ref_data.startswith('{'):
                        try:
                            import json
                            parsed_refs = json.loads(ref_data)
                            if isinstance(parsed_refs, list):
                                references.extend([str(ref) for ref in parsed_refs if ref])
                            elif isinstance(parsed_refs, dict):
                                references.extend([str(v) for v in parsed_refs.values() if v])
                        except (json.JSONDecodeError, ValueError):
                            # If JSON parsing fails, split by common delimiters
                            references.extend([ref.strip() for ref in ref_data.split(',') if ref.strip()])
                    else:
                        # Split by common delimiters
                        references.extend([ref.strip() for ref in ref_data.split(',') if ref.strip()])
                        
                elif isinstance(ref_data, list):
                    references.extend([str(ref) for ref in ref_data if ref])
                    
                elif isinstance(ref_data, (int, float)) and ref_data > 0:
                    # If it's a number, assume it's a count of references
                    references.extend([f"ref_{i}" for i in range(int(ref_data))])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_references = []
        for ref in references:
            if ref not in seen:
                seen.add(ref)
                unique_references.append(ref)
        
        return unique_references
    
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
        """Compute ensemble scores with rank-based normalization and data-driven weighting."""
        
        # Use rank-based normalization instead of min-max to avoid pollution from bad methods
        lof_embedding_scores_norm = self._rank_normalize_scores(results['lof_scores'])
        lof_mixed_scores_norm = self._rank_normalize_scores(results['lof_mixed_scores'])
        if_scores_norm = self._rank_normalize_scores(results['isolation_forest_scores'])
        svm_scores_norm = self._rank_normalize_scores(results['one_class_svm_scores'])
        dbscan_scores_norm = self._rank_normalize_scores(results['dbscan_scores'])
        
        # Calculate variance-based weights to emphasize methods with better discrimination
        score_arrays = {
            'lof_embedding': lof_embedding_scores_norm,
            'lof_mixed': lof_mixed_scores_norm,
            'isolation_forest': if_scores_norm,
            'one_class_svm': svm_scores_norm,
            'dbscan': dbscan_scores_norm,
        }
        
        # Compute weights based on score variance (higher variance = better discrimination)
        variances = {method: np.var(scores) for method, scores in score_arrays.items()}
        total_variance = sum(variances.values())
        
        if total_variance > 0:
            # Base weights on variance, but with minimum thresholds
            base_weights = {method: var / total_variance for method, var in variances.items()}
            
            # Apply constraints: LOF methods get at least 30% each, others get remaining
            lof_emb_weight = max(0.30, base_weights['lof_embedding'])
            lof_mixed_weight = max(0.25, base_weights['lof_mixed'])
            
            # Renormalize remaining weight for other methods
            remaining_weight = 1.0 - lof_emb_weight - lof_mixed_weight
            other_methods = ['isolation_forest', 'one_class_svm', 'dbscan']
            other_total_base = sum(base_weights[m] for m in other_methods)
            
            if other_total_base > 0:
                other_scale = remaining_weight / other_total_base
                weights = {
                    'lof_embedding': lof_emb_weight,
                    'lof_mixed': lof_mixed_weight,
                    'isolation_forest': base_weights['isolation_forest'] * other_scale,
                    'one_class_svm': base_weights['one_class_svm'] * other_scale,
                    'dbscan': base_weights['dbscan'] * other_scale,
                }
            else:
                # Fallback if other methods have no variance
                weights = {
                    'lof_embedding': 0.45,
                    'lof_mixed': 0.35,
                    'isolation_forest': 0.10,
                    'one_class_svm': 0.05,
                    'dbscan': 0.05,
                }
        else:
            # Fallback weights if no variance detected
            weights = {
                'lof_embedding': 0.40,
                'lof_mixed': 0.30,
                'isolation_forest': 0.15,
                'one_class_svm': 0.10,
                'dbscan': 0.05,
            }
        
        logger.info(f"Data-driven ensemble weights: LOF-EMB={weights['lof_embedding']:.3f}, "
                   f"LOF-MIX={weights['lof_mixed']:.3f}, IF={weights['isolation_forest']:.3f}, "
                   f"SVM={weights['one_class_svm']:.3f}, DBSCAN={weights['dbscan']:.3f}")
        
        # Compute weighted ensemble with rank-normalized scores
        ensemble_scores = (
            weights['lof_embedding'] * lof_embedding_scores_norm +
            weights['lof_mixed'] * lof_mixed_scores_norm +
            weights['isolation_forest'] * if_scores_norm +
            weights['one_class_svm'] * svm_scores_norm +
            weights['dbscan'] * dbscan_scores_norm
        )
        
        return ensemble_scores
    
    def _rank_normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores using rank-based method for better discrimination."""
        scores = np.array(scores)
        
        # Get ranks (higher score = higher rank)
        ranks = len(scores) - np.argsort(np.argsort(scores))  # Convert to ranks from 1 to n
        
        # Normalize ranks to [0, 1] range
        if len(scores) > 1:
            normalized = (ranks - 1) / (len(scores) - 1)
        else:
            normalized = np.zeros_like(ranks, dtype=float)
        
        return normalized
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max."""
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
        
        methods = ['lof', 'lof_mixed', 'isolation_forest', 'one_class_svm', 'dbscan', 'ensemble']
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
            
            # Create friendly method names
            method_names = {
                'lof': 'LOF (Embeddings)',
                'lof_mixed': 'LOF (Mixed Features)',
                'isolation_forest': 'Isolation Forest',
                'one_class_svm': 'One-Class SVM',
                'dbscan': 'DBSCAN',
                'ensemble': 'Ensemble'
            }
            
            comparison.append({
                'method': method_names.get(method, method),
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
                'lof_embeddings': float(self.outlier_results['lof_scores'][idx]),
                'lof_mixed': float(self.outlier_results['lof_mixed_scores'][idx]),
                'isolation_forest': float(self.outlier_results['isolation_forest_scores'][idx]),
                'one_class_svm': float(self.outlier_results['one_class_svm_scores'][idx]),
                'dbscan': float(self.outlier_results['dbscan_scores'][idx]),
                'ensemble': float(self.outlier_results['ensemble_scores'][idx]),
            },
            'predictions': {
                'lof_embeddings': bool(self.outlier_results['lof_predictions'][idx] == -1),
                'lof_mixed': bool(self.outlier_results['lof_mixed_predictions'][idx] == -1),
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


def _load_datasets_config() -> Dict[str, Any]:
    """Load datasets configuration from JSON file."""
    # Navigate up to project root from FINAL_MODEL/models/CitationNetwork/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    config_path = os.path.join(project_root, 'data', 'datasets.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Datasets configuration not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def _get_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    config = _load_datasets_config()
    return list(config.keys())


def _load_simulation_data(dataset_name: str) -> pd.DataFrame:
    """Load simulation data for a specific dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    simulation_path = os.path.join(project_root, 'data', 'simulations', f'{dataset_name}.csv')
    
    if not os.path.exists(simulation_path):
        raise FileNotFoundError(f"Simulation file not found: {simulation_path}")
    
    df = pd.read_csv(simulation_path)
    logger.info(f"Loaded simulation data for {dataset_name}: {len(df)} documents")
    
    return df


def _prompt_dataset_selection() -> str:
    """Prompt user to select a dataset from available options."""
    datasets = _get_available_datasets()
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    while True:
        try:
            selection = int(input("\nSelect dataset (enter number): "))
            if 1 <= selection <= len(datasets):
                return datasets[selection-1]
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Please enter a valid number")


def _evaluate_outlier_ranking(scores: Dict[str, float], dataset_name: str, datasets_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate how well known outliers are ranked by the model."""
    if dataset_name not in datasets_config:
        return []
    
    # Get known outlier record IDs
    outlier_record_ids = datasets_config[dataset_name].get('outlier_ids', [])
    
    # Load simulation data to map record_id to openalex_id
    simulation_df = _load_simulation_data(dataset_name)
    record_to_openalex = dict(zip(simulation_df['record_id'], simulation_df['openalex_id']))
    
    # Convert record IDs to OpenAlex IDs
    outlier_openalex_ids = []
    for record_id in outlier_record_ids:
        if record_id in record_to_openalex:
            outlier_openalex_ids.append(record_to_openalex[record_id])
    
    # Sort all documents by score (highest first)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create ranking lookup
    doc_to_rank = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_docs)}
    
    results = []
    for outlier_id in outlier_openalex_ids:
        if outlier_id in scores:
            rank = doc_to_rank[outlier_id]
            score = scores[outlier_id]
            total_docs = len(scores)
            percentile = ((total_docs - rank + 1) / total_docs) * 100
            
            results.append({
                'outlier_id': outlier_id,
                'rank': rank,
                'score': score,
                'total_documents': total_docs,
                'percentile': percentile
            })
    
    return results


def main():
    """Standalone Citation Network Outlier Detection with Dataset Selection and Ranking Analysis."""
    print("=" * 60)
    print("CITATION NETWORK OUTLIER DETECTION - STANDALONE MODE")
    print("=" * 60)
    
    try:
        # Step 1: Dataset Selection
        print("\nStep 1: Dataset Selection")
        dataset_name = _prompt_dataset_selection()
        
        # Load datasets configuration
        datasets_config = _load_datasets_config()
        
        # Step 2: Load Data
        print(f"\nStep 2: Loading dataset '{dataset_name}'...")
        simulation_df = _load_simulation_data(dataset_name)
        print(f"Loaded {len(simulation_df)} documents")
        
        # Step 3: Initialize and Run Model
        print(f"\nStep 3: Running Citation Network Outlier Detection...")
        print("Configuration:")
        print("  - UMAP dimensionality reduction: Enabled (50 components)")
        print("  - LOF on embeddings: Primary method for subtopic detection")
        print("  - Ensemble weighting: 50% embedding-LOF, 50% other methods")
        
        detector = CitationNetworkOutlierDetector(
            contamination=0.1,
            enable_semantic=True,
            use_umap=False,  # Enable UMAP for dimensionality reduction
            umap_components=50,  # Reduce to 50 dimensions
            random_state=42
        )
        
        start_time = time.time()
        results = detector.fit_predict_outliers(simulation_df, dataset_name=dataset_name)
        runtime = time.time() - start_time
        
        print(f"Model completed in {runtime:.2f} seconds")
        
        # Step 4: Create Score Dictionary for Ranking
        doc_ids = results['openalex_ids']
        ensemble_scores = results['ensemble_scores']
        scores_dict = dict(zip(doc_ids, ensemble_scores))
        
        # Step 5: Evaluate Known Outliers
        print(f"\n" + "=" * 50)
        print("OUTLIER RANKING PERFORMANCE")
        print("=" * 50)
        
        outlier_ranking_results = _evaluate_outlier_ranking(scores_dict, dataset_name, datasets_config)
        
        if outlier_ranking_results:
            for result in outlier_ranking_results:
                print(f"\nKnown Outlier: {result['outlier_id']}")
                print(f"  Rank: {result['rank']} out of {result['total_documents']}")
                print(f"  Ensemble Score: {result['score']:.4f}")
                print(f"  Percentile: {result['percentile']:.1f}% (higher is better)")
                
                # Performance assessment
                if result['percentile'] >= 95:
                    performance = "Excellent ✓"
                elif result['percentile'] >= 90:
                    performance = "Very Good ✓"
                elif result['percentile'] >= 80:
                    performance = "Good"
                elif result['percentile'] >= 70:
                    performance = "Fair"
                else:
                    performance = "Poor"
                print(f"  Performance: {performance}")
                
                # Show detailed score breakdown for the known outlier
                print(f"\n  DETAILED SCORE BREAKDOWN:")
                analysis = detector.analyze_document(result['outlier_id'])
                
                print(f"    Individual Method Scores:")
                method_display_names = {
                    'lof_embeddings': 'LOF-EMBEDDINGS',
                    'lof_mixed': 'LOF-MIXED',
                    'isolation_forest': 'ISOLATION-FOREST',
                    'one_class_svm': 'ONE-CLASS-SVM',
                    'dbscan': 'DBSCAN',
                    'ensemble': 'ENSEMBLE'
                }
                for method, score in analysis['scores'].items():
                    prediction = analysis['predictions'][method]
                    status = "OUTLIER" if prediction else "normal"
                    display_name = method_display_names.get(method, method.upper())
                    print(f"      {display_name}: {score:.4f} ({status})")
                
                print(f"\n    Network Features:")
                features = analysis.get('features', {})
                print(f"      Degree: {features.get('degree', 0)}")
                print(f"      PageRank: {features.get('pagerank', 0):.6f}")
                print(f"      Clustering Coefficient: {features.get('clustering', 0):.4f}")
                print(f"      Relevant Neighbors: {features.get('relevant_neighbors', 0)}")
                print(f"      Relevant Ratio: {features.get('relevant_ratio', 0):.4f}")
                print(f"      Min Distance to Relevant: {features.get('min_distance_to_relevant', 0):.4f}")
                if 'semantic_similarity_to_relevant' in features:
                    print(f"      Semantic Similarity: {features['semantic_similarity_to_relevant']:.4f}")
        else:
            print("No known outliers defined for this dataset.")
        
        # Step 6: Show Top Documents by Ensemble Score
        print(f"\n" + "=" * 50)
        print("TOP SCORING DOCUMENTS (by Ensemble Score)")
        print("=" * 50)
        
        top_docs = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (doc_id, score) in enumerate(top_docs, 1):
            print(f"\n{i}. Document: {doc_id}")
            print(f"   Ensemble Score: {score:.4f}")
            
            # Show abbreviated breakdown for top 5
            if i <= 5:
                analysis = detector.analyze_document(doc_id)
                print(f"   Method Scores: ", end="")
                method_scores = []
                method_abbrev = {
                    'lof_embeddings': 'LOF-EMB',
                    'lof_mixed': 'LOF-MIX',
                    'isolation_forest': 'IF',
                    'one_class_svm': 'SVM',
                    'dbscan': 'DBSCAN'
                }
                for method, score_val in analysis['scores'].items():
                    if method != 'ensemble':
                        abbrev = method_abbrev.get(method, method.upper())
                        method_scores.append(f"{abbrev}: {score_val:.3f}")
                print(" | ".join(method_scores))
        
        # Step 7: Method Comparison
        print(f"\n" + "=" * 50)
        print("METHOD COMPARISON")
        print("=" * 50)
        
        comparison = detector.get_method_comparison()
        print(f"{'Method':<20} {'Outliers':<10} {'Percentage':<12} {'Mean Score':<12} {'Max Score':<10}")
        print("-" * 70)
        for _, row in comparison.iterrows():
            print(f"{row['method']:<20} {row['num_outliers']:<10} {row['outlier_percentage']:<12.1f} {row['mean_score']:<12.4f} {row['max_score']:<10.4f}")
        
        print(f"\n" + "=" * 60)
        print("CITATION NETWORK ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()