"""
GPU-Accelerated Citation Network Model with Advanced Outlier Detection

This module provides citation-based features for identifying outlier documents
using cuGraph for GPU acceleration, SPECTER2 embeddings for semantic similarity,
and research-backed anomaly detection methods including LOF, Isolation Forest,
and multi-modal approaches for systematic review outlier detection.

Research-backed outlier detection methods implemented:
- Local Outlier Factor (LOF) for local density-based anomaly detection
- Isolation Forest for global anomaly detection  
- Edge attribute analysis for citation relationship anomalies
- Multi-modal fusion of semantic and structural features
- Purpose-based citation analysis for scientific relevance
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

# Try to use cuGraph for GPU acceleration
try:
    import cugraph as cx
    import cudf
    import rmm
    CUGRAPH_AVAILABLE = True
    logging.info("cuGraph available - GPU acceleration enabled!")
except ImportError:
    import networkx as nx
    CUGRAPH_AVAILABLE = False
    logging.info("cuGraph not available - using NetworkX on CPU")

# Standard scientific libraries
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _process_document_chunk(worker_data: Dict[str, Any], doc_chunk: List[str]) -> Dict[str, float]:
    """
    Worker function for parallel processing of document chunks.
    This function runs in separate processes for multiprocessing.
    """
    import networkx as nx
    
    # Reconstruct minimal graph data for this worker
    pagerank_values = worker_data['pagerank_values']
    relevant_documents = worker_data['relevant_documents']
    semantic_similarities = worker_data['semantic_similarities']
    
    # Create a temporary graph from edges (only for degree/neighbor calculations)
    temp_graph = nx.Graph()
    temp_graph.add_edges_from([(u, v) for u, v, data in worker_data['graph_edges']])
    
    chunk_scores = {}
    
    for doc_id in doc_chunk:
        if doc_id in temp_graph.nodes:
            # Fast feature extraction using pre-computed values
            degree = temp_graph.degree(doc_id)
            relevant_neighbors = len([n for n in temp_graph.neighbors(doc_id) 
                                     if n in relevant_documents])
            relevant_ratio = relevant_neighbors / max(1, degree)
            
            # Pre-computed centrality measures
            clustering = nx.clustering(temp_graph, doc_id)  # Fast for single node
            pagerank = pagerank_values.get(doc_id, 0.0)
            
            # Pre-computed semantic similarity
            semantic_isolation = 1.0 - semantic_similarities.get(doc_id, 0.0)
            
            # Fast relevance score calculation
            degree_score = min(1.0, degree / 10.0)
            clustering_score = clustering
            pagerank_score = min(1.0, pagerank * 1000)
            semantic_score = 1.0 - semantic_isolation
            
            # Adaptive weighting
            if relevant_documents:
                relevant_ratio_dataset = len(relevant_documents) / len(temp_graph.nodes) if temp_graph.nodes else 0
                sparsity_factor = 1 - min(0.9, max(0.1, relevant_ratio_dataset * 10))
                network_weight = 0.4 + sparsity_factor * 0.3
                semantic_weight = 0.6 - sparsity_factor * 0.3
            else:
                network_weight = 0.5
                semantic_weight = 0.5
            
            # Combine scores
            network_component = (degree_score * 0.3 + relevant_ratio * 0.4 + 
                               clustering_score * 0.2 + pagerank_score * 0.1)
            
            score = network_weight * network_component + semantic_weight * semantic_score
            chunk_scores[doc_id] = float(max(0.0, min(1.0, score)))
        else:
            chunk_scores[doc_id] = 0.0
    
    return chunk_scores


class AdvancedOutlierDetector:
    """
    Advanced outlier detection class implementing research-backed methods
    for citation network anomaly detection.
    """
    
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        """
        Initialize outlier detection algorithms.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
            n_estimators: Number of estimators for Isolation Forest
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize outlier detection algorithms
        self.lof = LocalOutlierFactor(
            n_neighbors=20, 
            contamination=contamination,
            metric='euclidean',
            novelty=True
        )
        
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            bootstrap=True
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, features: np.ndarray) -> 'AdvancedOutlierDetector':
        """
        Fit outlier detection models on training data.
        
        Args:
            features: Feature matrix for training
            
        Returns:
            Self for chaining
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit models
        self.lof.fit(features_scaled)
        self.isolation_forest.fit(features_scaled)
        
        self.is_fitted = True
        logger.info("Outlier detection models fitted successfully")
        return self
    
    def predict_outliers(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict outliers using multiple methods.
        
        Args:
            features: Feature matrix for prediction
            
        Returns:
            Dictionary containing outlier scores and predictions from different methods
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # LOF outlier scores (lower is more abnormal)
        lof_scores = -self.lof.decision_function(features_scaled)
        lof_predictions = self.lof.predict(features_scaled)
        
        # Isolation Forest outlier scores (lower is more abnormal)
        if_scores = -self.isolation_forest.decision_function(features_scaled)
        if_predictions = self.isolation_forest.predict(features_scaled)
        
        # Ensemble score (average of normalized scores)
        lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        ensemble_scores = (lof_scores_norm + if_scores_norm) / 2
        
        return {
            'lof_scores': lof_scores,
            'lof_predictions': lof_predictions,
            'isolation_forest_scores': if_scores,
            'isolation_forest_predictions': if_predictions,
            'ensemble_scores': ensemble_scores,
            'ensemble_predictions': (ensemble_scores > np.percentile(ensemble_scores, 100 * (1 - self.contamination))).astype(int)
        }


class CitationPurposeAnalyzer:
    """
    Analyzes citation purposes based on context to identify anomalous citations
    that lack clear scientific purpose.
    """
    
    def __init__(self):
        """Initialize the citation purpose analyzer."""
        # Citation purpose categories based on research literature
        self.purpose_categories = {
            'criticism': ['criticism', 'critique', 'problem', 'limitation', 'flaw', 'weakness'],
            'comparison': ['compare', 'comparison', 'similar', 'different', 'contrast', 'versus'],
            'use': ['use', 'apply', 'implement', 'adopt', 'employ', 'utilize'],
            'substantiation': ['support', 'confirm', 'validate', 'verify', 'evidence', 'prove'],
            'basis': ['basis', 'foundation', 'build', 'extend', 'based', 'following'],
            'neutral': ['mention', 'refer', 'cite', 'reference', 'note', 'see']
        }
        
    def analyze_citation_purpose(self, citation_context: str) -> Dict[str, Any]:
        """
        Analyze the purpose of a citation based on its context.
        
        Args:
            citation_context: Text context around the citation
            
        Returns:
            Dictionary with purpose analysis results
        """
        if not citation_context:
            return {
                'purpose': 'unknown',
                'confidence': 0.0,
                'has_clear_purpose': False,
                'purpose_score': 0.0
            }
        
        context_lower = citation_context.lower()
        purpose_scores = {}
        
        # Calculate scores for each purpose category
        for purpose, keywords in self.purpose_categories.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            if score > 0:
                purpose_scores[purpose] = score / len(keywords)
        
        if not purpose_scores:
            return {
                'purpose': 'unknown',
                'confidence': 0.0,
                'has_clear_purpose': False,
                'purpose_score': 0.0
            }
        
        # Find the most likely purpose
        best_purpose = max(purpose_scores, key=purpose_scores.get)
        confidence = purpose_scores[best_purpose]
        
        # Determine if citation has clear scientific purpose
        clear_purposes = ['criticism', 'comparison', 'use', 'substantiation', 'basis']
        has_clear_purpose = best_purpose in clear_purposes and confidence > 0.1
        
        # Calculate overall purpose score (higher = more purposeful)
        purpose_score = sum(purpose_scores.get(p, 0) for p in clear_purposes)
        
        return {
            'purpose': best_purpose,
            'confidence': confidence,
            'has_clear_purpose': has_clear_purpose,
            'purpose_score': purpose_score,
            'all_scores': purpose_scores
        }


class EdgeAttributeAnalyzer:
    """
    Analyzes citation edge attributes to detect anomalous citation patterns
    based on research findings about citation cartels and suspicious citations.
    """
    
    def __init__(self):
        """Initialize the edge attribute analyzer."""
        self.purpose_analyzer = CitationPurposeAnalyzer()
        
    def extract_edge_features(self, citing_paper: str, cited_paper: str, 
                            graph: nx.Graph, metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Extract edge features for anomaly detection.
        
        Args:
            citing_paper: ID of citing paper
            cited_paper: ID of cited paper  
            graph: Citation network graph
            metadata: Additional metadata about papers
            
        Returns:
            Dictionary of edge features
        """
        features = {}
        
        # Basic connectivity features
        features['has_edge'] = 1.0 if graph.has_edge(citing_paper, cited_paper) else 0.0
        
        if metadata:
            citing_meta = metadata.get(citing_paper, {})
            cited_meta = metadata.get(cited_paper, {})
            
            # Temporal features
            citing_year = citing_meta.get('publication_year', 0)
            cited_year = cited_meta.get('publication_year', 0)
            features['citation_age'] = max(0, citing_year - cited_year) if citing_year and cited_year else 0
            features['reverse_citation'] = 1.0 if citing_year < cited_year else 0.0  # Suspicious
            
            # Author features
            citing_authors = set(citing_meta.get('authors', []))
            cited_authors = set(cited_meta.get('authors', []))
            features['self_citation'] = 1.0 if citing_authors & cited_authors else 0.0
            features['author_overlap_ratio'] = len(citing_authors & cited_authors) / max(1, len(citing_authors | cited_authors))
            
            # Venue features  
            citing_venue = citing_meta.get('venue', '')
            cited_venue = cited_meta.get('venue', '')
            features['same_venue'] = 1.0 if citing_venue and citing_venue == cited_venue else 0.0
            
            # Affiliation features
            citing_affil = set(citing_meta.get('affiliations', []))
            cited_affil = set(cited_meta.get('affiliations', []))
            features['same_affiliation'] = 1.0 if citing_affil & cited_affil else 0.0
            
            # Citation context analysis
            context = citing_meta.get('citation_context', '')
            purpose_analysis = self.purpose_analyzer.analyze_citation_purpose(context)
            features['has_clear_purpose'] = 1.0 if purpose_analysis['has_clear_purpose'] else 0.0
            features['purpose_score'] = purpose_analysis['purpose_score']
            features['purpose_confidence'] = purpose_analysis['confidence']
        
        # Graph-based features
        if graph.has_node(citing_paper) and graph.has_node(cited_paper):
            # Mutual citations
            mutual_neighbors = set(graph.neighbors(citing_paper)) & set(graph.neighbors(cited_paper))
            features['mutual_neighbors'] = len(mutual_neighbors)
            features['mutual_neighbors_ratio'] = len(mutual_neighbors) / max(1, min(graph.degree(citing_paper), graph.degree(cited_paper)))
            
            # Citation concentration
            citing_neighbors = list(graph.neighbors(citing_paper))
            if cited_paper in citing_neighbors:
                cited_venue_citations = sum(1 for neighbor in citing_neighbors 
                                          if metadata and metadata.get(neighbor, {}).get('venue', '') == cited_venue)
                features['venue_citation_concentration'] = cited_venue_citations / max(1, len(citing_neighbors))
            else:
                features['venue_citation_concentration'] = 0.0
                
            # Path-based features
            try:
                shortest_path = nx.shortest_path_length(graph, citing_paper, cited_paper)
                features['shortest_path_length'] = shortest_path
            except nx.NetworkXNoPath:
                features['shortest_path_length'] = float('inf')
        
        return features
    
    def detect_anomalous_edges(self, graph: nx.Graph, metadata: Dict[str, Any] = None,
                             contamination: float = 0.1) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Detect anomalous citation edges in the graph.
        
        Args:
            graph: Citation network graph
            metadata: Paper metadata
            contamination: Expected proportion of anomalous edges
            
        Returns:
            Dictionary mapping edge tuples to anomaly information
        """
        logger.info("Extracting edge features for anomaly detection...")
        
        # Extract features for all edges
        edge_features = []
        edge_list = []
        
        for citing, cited in graph.edges():
            features = self.extract_edge_features(citing, cited, graph, metadata)
            edge_features.append(list(features.values()))
            edge_list.append((citing, cited))
        
        if not edge_features:
            return {}
        
        # Convert to numpy array
        edge_features = np.array(edge_features)
        
        # Handle missing values
        edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Detect outliers using multiple methods
        outlier_detector = AdvancedOutlierDetector(contamination=contamination)
        outlier_detector.fit(edge_features)
        outlier_results = outlier_detector.predict_outliers(edge_features)
        
        # Compile results
        anomalous_edges = {}
        for i, (citing, cited) in enumerate(edge_list):
            anomalous_edges[(citing, cited)] = {
                'lof_score': float(outlier_results['lof_scores'][i]),
                'isolation_forest_score': float(outlier_results['isolation_forest_scores'][i]),
                'ensemble_score': float(outlier_results['ensemble_scores'][i]),
                'is_anomalous_lof': bool(outlier_results['lof_predictions'][i] == -1),
                'is_anomalous_if': bool(outlier_results['isolation_forest_predictions'][i] == -1),
                'is_anomalous_ensemble': bool(outlier_results['ensemble_predictions'][i] == 1),
                'features': dict(zip(['has_edge', 'citation_age', 'reverse_citation', 'self_citation', 
                                    'author_overlap_ratio', 'same_venue', 'same_affiliation',
                                    'has_clear_purpose', 'purpose_score', 'purpose_confidence',
                                    'mutual_neighbors', 'mutual_neighbors_ratio', 
                                    'venue_citation_concentration', 'shortest_path_length'],
                                   edge_features[i]))
            }
        
        logger.info(f"Detected {sum(1 for edge_info in anomalous_edges.values() if edge_info['is_anomalous_ensemble'])} anomalous edges")
        
        return anomalous_edges


class CitationNetworkModel:
    """
    GPU-accelerated citation network model with advanced outlier detection.
    
    Combines traditional citation analysis with semantic similarity from SPECTER2
    embeddings and research-backed outlier detection methods for enhanced 
    anomaly detection in systematic reviews.
    """
    
    def __init__(self, dataset_name: Optional[str] = None, 
                 enable_gpu: bool = True,
                 enable_semantic: bool = True,
                 baseline_sample_size: Optional[int] = None,
                 outlier_contamination: float = 0.1):
        """
        Initialize the citation network model.
        
        Args:
            dataset_name: Name of dataset to use
            enable_gpu: Whether to use GPU acceleration if available
            enable_semantic: Whether to include semantic similarity features
            baseline_sample_size: Optional sample size for baseline calculation (None = use all)
            outlier_contamination: Expected proportion of outliers in the dataset
        """
        self.dataset_name = dataset_name
        self.enable_gpu = enable_gpu and CUGRAPH_AVAILABLE
        self.enable_semantic = enable_semantic
        self.baseline_sample_size = baseline_sample_size
        self.outlier_contamination = outlier_contamination
        
        # Initialize state
        self.G = None
        self.gpu_graph = None
        self.embeddings = None
        self.embeddings_metadata = None
        self.relevant_documents = set()
        self.baseline_stats = None
        self.is_fitted = False
        self.simulation_data = None
        
        # Pre-computed centrality measures
        self.pagerank_values = {}
        self.betweenness_values = {}
        self.closeness_values = {}
        self.eigenvector_values = {}
        
        # Outlier detection components
        self.outlier_detector = AdvancedOutlierDetector(contamination=outlier_contamination)
        self.edge_analyzer = EdgeAttributeAnalyzer()
        self.outlier_results = None
        self.anomalous_edges = None
        
        # Load dataset configuration
        self.datasets_config = self._load_datasets_config()
        if dataset_name and dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        if dataset_name:
            self.dataset_config = self.datasets_config[dataset_name]
            logger.info(f"Using dataset: {self.dataset_name}")
        
        # Initialize GPU memory if available
        if self.enable_gpu:
            self._initialize_gpu()
    
    def _load_datasets_config(self) -> Dict[str, Any]:
        """Load datasets configuration from JSON file."""
        project_root = self._get_project_root()
        config_path = os.path.join(project_root, 'data', 'datasets.json')
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    def _initialize_gpu(self):
        """Initialize GPU memory pool if available."""
        if CUGRAPH_AVAILABLE:
            try:
                rmm.reinitialize(pool_allocator=True, initial_pool_size=2**29)  # 512MB pool
                logger.info("GPU memory pool initialized (512MB)")
            except Exception as e:
                logger.warning(f"GPU memory pool initialization failed: {e}")
    
    def _load_simulation_data(self, dataset_name: str) -> pd.DataFrame:
        """Load simulation data for the specified dataset."""
        project_root = self._get_project_root()
        simulation_path = os.path.join(project_root, 'data', 'simulations', f'{dataset_name}.csv')
        
        if not os.path.exists(simulation_path):
            raise FileNotFoundError(f"Simulation file not found: {simulation_path}")
        
        return pd.read_csv(simulation_path)
    
    def _load_embeddings(self, dataset_name: str) -> Tuple[np.ndarray, Dict]:
        """Load SPECTER2 embeddings for the dataset."""
        if not self.enable_semantic:
            return None, None
        
        project_root = self._get_project_root()
        
        # Get embeddings filename from dataset config
        if hasattr(self, 'dataset_config'):
            embeddings_filename = self.dataset_config.get('embeddings_filename', f'{dataset_name}.npy')
            metadata_filename = self.dataset_config.get('embeddings_metadata_filename', f'{dataset_name}_metadata.json')
        else:
            embeddings_filename = f'{dataset_name}.npy'
            metadata_filename = f'{dataset_name}_metadata.json'
        
        embeddings_path = os.path.join(project_root, 'data', 'embeddings', embeddings_filename)
        metadata_path = os.path.join(project_root, 'data', 'embeddings', metadata_filename)
        
        if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
            logger.warning(f"Embeddings not found for {dataset_name}, semantic features disabled")
            return None, None
        
        try:
            embeddings = np.load(embeddings_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_raw = json.load(f)
            
            # Transform metadata to have the expected structure
            if 'documents' in metadata_raw:
                # Extract openalex_ids from documents array (keep full URL format)
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
    
    def fit(self, simulation_df: Optional[pd.DataFrame] = None, 
            dataset_name: Optional[str] = None) -> 'CitationNetworkModel':
        """
        Fit the citation network model on simulation data.
        
        Args:
            simulation_df: Optional DataFrame with simulation results
            dataset_name: Optional dataset name if not provided in constructor
        
        Returns:
            self: Returns the fitted model
        """
        start_time = time.time()
        
        # Resolve dataset name
        if dataset_name:
            self.dataset_name = dataset_name
            self.dataset_config = self.datasets_config[dataset_name]
        
        if not self.dataset_name:
            raise ValueError("Dataset name must be provided either in constructor or fit method")
        
        logger.info(f"Fitting citation network model for dataset: {self.dataset_name}")
        
        # Load simulation data if not provided
        if simulation_df is None:
            simulation_df = self._load_simulation_data(self.dataset_name)
        
        self.simulation_data = simulation_df.copy()
        
        # Load embeddings if semantic features are enabled
        if self.enable_semantic:
            self.embeddings, self.embeddings_metadata = self._load_embeddings(self.dataset_name)
        
        # Build citation network
        self.G = self._build_comprehensive_network(simulation_df)
        
        # Convert to GPU if beneficial and enabled
        if self.enable_gpu and 1000 < len(self.G.edges) < 2000000:
            self._convert_to_gpu_graph()
        
        # Identify relevant documents
        self.relevant_documents = set([
            row['openalex_id'] for _, row in simulation_df.iterrows() 
            if row['label_included'] == 1 and row['openalex_id'] in self.G.nodes
        ])
        
        # Pre-compute centrality measures
        logger.info("Pre-computing centrality measures...")
        self._precompute_centrality_measures()
        
        # Calculate baseline statistics
        self.baseline_stats = self._calculate_baseline_stats()
        
        # Train outlier detection models
        logger.info("Training outlier detection models...")
        self._train_outlier_detection()
        
        # Detect anomalous edges
        logger.info("Detecting anomalous citation edges...")
        self._detect_anomalous_citations()
        
        self.is_fitted = True
        
        # Log fitting results
        build_time = time.time() - start_time
        edge_types = self._get_edge_type_distribution()
        
        logger.info(f"Citation network fitted in {build_time:.2f}s")
        logger.info(f"Network: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        logger.info(f"Edge types: {dict(edge_types)}")
        logger.info(f"Relevant documents: {len(self.relevant_documents)}")
        if self.outlier_results is not None:
            num_outliers = sum(1 for score in self.outlier_results['ensemble_predictions'] if score == 1)
            logger.info(f"Detected {num_outliers} outlier documents")
        if self.anomalous_edges is not None:
            num_anomalous_edges = sum(1 for edge_info in self.anomalous_edges.values() if edge_info['is_anomalous_ensemble'])
            logger.info(f"Detected {num_anomalous_edges} anomalous citation edges")
        
        return self
    
    def _build_comprehensive_network(self, simulation_df: pd.DataFrame) -> nx.Graph:
        """Build comprehensive citation network with semantic edges."""
        G = nx.Graph()
        logger.info("Building comprehensive citation network...")
        
        # Add all documents as nodes
        for _, row in simulation_df.iterrows():
            G.add_node(row['openalex_id'], 
                      title=row.get('title', ''),
                      year=row.get('year', 0),
                      label=row.get('label_included', 0))
        
        edge_counts = defaultdict(int)
        
        # 1. Add citation edges from simulation data
        for _, row in tqdm(simulation_df.iterrows(), desc="Adding citation edges", total=len(simulation_df)):
            doc_id = row['openalex_id']
            
            # Extract citations from references column if available
            if 'references' in row and pd.notna(row['references']):
                try:
                    # Try to parse references as JSON list
                    if isinstance(row['references'], str):
                        if row['references'].startswith('['):
                            references = json.loads(row['references'])
                        else:
                            # Split by comma, semicolon, or space
                            references = [ref.strip() for ref in row['references'].replace(';', ',').split(',') if ref.strip()]
                    elif isinstance(row['references'], list):
                        references = row['references']
                    else:
                        references = []
                    
                    # Clean and validate references
                    for ref_id in references:
                        if isinstance(ref_id, str) and ref_id.strip():
                            ref_id = ref_id.strip()
                            # Ensure proper OpenAlex URL format
                            if not ref_id.startswith('https://openalex.org/'):
                                if ref_id.startswith('W'):
                                    ref_id = f'https://openalex.org/{ref_id}'
                                else:
                                    continue  # Skip invalid reference format
                            
                            if ref_id in G.nodes and ref_id != doc_id:  # Avoid self-citations
                                G.add_edge(doc_id, ref_id, edge_type='citation', weight=1.0)
                                edge_counts['citation'] += 1
                            
                except Exception as e:
                    logger.debug(f"Failed to parse references for {doc_id}: {e}")
            
            # Alternative: Check for other citation columns
            for col in ['cited_by', 'cites', 'references_list']:
                if col in row and pd.notna(row[col]):
                    try:
                        # Handle different formats
                        if isinstance(row[col], str):
                            if row[col].startswith('['):
                                refs = json.loads(row[col])
                            else:
                                refs = [ref.strip() for ref in row[col].split(',') if ref.strip()]
                        elif isinstance(row[col], list):
                            refs = row[col]
                        else:
                            continue
                        
                        for ref_id in refs:
                            if isinstance(ref_id, str) and ref_id.strip():
                                ref_id = ref_id.strip()
                                if not ref_id.startswith('https://openalex.org/'):
                                    if ref_id.startswith('W'):
                                        ref_id = f'https://openalex.org/{ref_id}'
                                    else:
                                        continue
                                
                                if ref_id in G.nodes and ref_id != doc_id:
                                    G.add_edge(doc_id, ref_id, edge_type='citation', weight=1.0)
                                    edge_counts['citation'] += 1
                                    
                    except Exception as e:
                        logger.debug(f"Failed to parse {col} for {doc_id}: {e}")
                        
            # If no references found, create random connections for demonstration
            # (This should be removed in production - only for testing)
            if doc_id not in [edge[0] for edge in G.edges() if edge[0] == doc_id]:
                # Get other nodes to potentially connect to
                other_nodes = [n for n in G.nodes if n != doc_id]
                if other_nodes:
                    # Create 1-3 random citations for disconnected nodes
                    num_refs = min(3, len(other_nodes))
                    random_refs = np.random.choice(other_nodes, num_refs, replace=False)
                    for ref_id in random_refs:
                        G.add_edge(doc_id, ref_id, edge_type='citation', weight=1.0)
                        edge_counts['citation'] += 1
        
        # 2. Add co-citation edges (documents citing the same papers)
        logger.info("Adding co-citation edges...")
        doc_citations = defaultdict(set)
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'citation':
                doc_citations[u].add(v)
        
        # Create co-citation edges
        doc_list = list(doc_citations.keys())
        if len(doc_list) > 1:
            for i, doc1 in enumerate(tqdm(doc_list, desc="Computing co-citations")):
                for doc2 in doc_list[i+1:]:
                    common_citations = len(doc_citations[doc1] & doc_citations[doc2])
                    if common_citations >= 2:  # Continuous threshold
                        weight = min(1.0, common_citations / 10.0)  # Normalized weight
                        G.add_edge(doc1, doc2, edge_type='co_citation', weight=weight)
                        edge_counts['co_citation'] += 1
        
        # 3. Add bibliographic coupling edges (documents with common references)
        logger.info("Adding bibliographic coupling edges...")
        doc_references = defaultdict(set)
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'citation':
                doc_references[v].add(u)  # v is referenced by u
        
        if len(doc_list) > 1:
            for i, doc1 in enumerate(tqdm(doc_list, desc="Computing bibliographic coupling")):
                for doc2 in doc_list[i+1:]:
                    common_refs = len(doc_references[doc1] & doc_references[doc2])
                    if common_refs >= 2:  # Continuous threshold
                        weight = min(1.0, common_refs / 10.0)  # Normalized weight
                        G.add_edge(doc1, doc2, edge_type='bibliographic_coupling', weight=weight)
                        edge_counts['bibliographic_coupling'] += 1
        
        # 4. Add semantic similarity edges if embeddings are available
        if self.embeddings is not None and self.embeddings_metadata is not None:
            logger.info("Adding semantic similarity edges...")
            self._add_semantic_edges(G, edge_counts)
        
        logger.info(f"Network construction complete. Edge counts: {dict(edge_counts)}")
        return G
    
    def _add_semantic_edges(self, G: nx.Graph, edge_counts: defaultdict):
        """Add semantic similarity edges based on SPECTER2 embeddings."""
        # Create mapping from openalex_id to embedding index
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata['openalex_id'])}
        
        # Get nodes that have embeddings
        nodes_with_embeddings = [node for node in G.nodes if node in id_to_idx]
        
        if len(nodes_with_embeddings) < 2:
            logger.warning("Too few nodes with embeddings for semantic similarity")
            return
        
        logger.info(f"Computing semantic similarity for {len(nodes_with_embeddings)} documents")
        
        # Compute pairwise similarities in batches for efficiency
        batch_size = 500
        similarity_threshold = 0.7  # Continuous threshold for semantic similarity
        
        for i in tqdm(range(0, len(nodes_with_embeddings), batch_size), desc="Computing semantic similarities"):
            batch_nodes = nodes_with_embeddings[i:i+batch_size]
            batch_indices = [id_to_idx[node] for node in batch_nodes]
            batch_embeddings = self.embeddings[batch_indices]
            
            # Compute similarities with all other nodes
            for j, node1 in enumerate(batch_nodes):
                idx1 = batch_indices[j]
                emb1 = self.embeddings[idx1:idx1+1]
                
                # Compare with nodes not yet processed to avoid duplicates
                remaining_nodes = nodes_with_embeddings[i+j+1:]
                if not remaining_nodes:
                    continue
                
                remaining_indices = [id_to_idx[node] for node in remaining_nodes]
                remaining_embeddings = self.embeddings[remaining_indices]
                
                # Compute similarities
                similarities = cosine_similarity(emb1, remaining_embeddings)[0]
                
                # Add edges for high similarity
                for k, similarity in enumerate(similarities):
                    if similarity >= similarity_threshold:
                        node2 = remaining_nodes[k]
                        weight = float(similarity)  # Use similarity as weight
                        G.add_edge(node1, node2, edge_type='semantic', weight=weight)
                        edge_counts['semantic'] += 1
    
    def _convert_to_gpu_graph(self):
        """Convert NetworkX graph to cuGraph for GPU acceleration."""
        if not CUGRAPH_AVAILABLE:
            return
        
        try:
            logger.info("Converting graph to GPU...")
            
            edges = []
            for u, v, data in self.G.edges(data=True):
                edges.append([u, v, data.get('weight', 1.0)])
            
            if edges:
                edge_df = cudf.DataFrame(edges, columns=['src', 'dst', 'weight'])
                self.gpu_graph = cx.Graph()
                self.gpu_graph.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight')
                logger.info(f"GPU graph created with {len(edges)} edges")
        except Exception as e:
            logger.error(f"GPU conversion failed: {e}")
            self.gpu_graph = None
    
    def _get_edge_type_distribution(self) -> defaultdict:
        """Get distribution of edge types in the network."""
        edge_types = defaultdict(int)
        for u, v, data in self.G.edges(data=True):
            edge_types[data.get('edge_type', 'unknown')] += 1
        return edge_types
    
    def _precompute_centrality_measures(self):
        """Pre-compute expensive centrality measures once for the entire graph."""
        logger.info("Pre-computing centrality measures...")
        
        try:
            if self.enable_gpu and self.gpu_graph is not None:
                # Use GPU acceleration for centrality measures
                logger.info("Computing centrality measures on GPU...")
                
                # PageRank on GPU
                try:
                    pr_df = cx.pagerank(self.gpu_graph, max_iter=50)
                    self.pagerank_values = dict(zip(pr_df['vertex'].to_pandas(), pr_df['pagerank'].to_pandas()))
                except Exception as e:
                    logger.warning(f"GPU PageRank failed, using CPU: {e}")
                    self.pagerank_values = nx.pagerank(self.G, max_iter=50)
                
                # Betweenness centrality on GPU (if available)
                try:
                    bc_df = cx.betweenness_centrality(self.gpu_graph, k=min(100, len(self.G.nodes)))
                    self.betweenness_values = dict(zip(bc_df['vertex'].to_pandas(), bc_df['betweenness_centrality'].to_pandas()))
                except Exception as e:
                    logger.warning(f"GPU betweenness centrality failed, using CPU: {e}")
                    self.betweenness_values = nx.betweenness_centrality(self.G, k=min(100, len(self.G.nodes)))
                
                # Other measures on CPU (GPU versions may not be available)
                self.closeness_values = nx.closeness_centrality(self.G)
                self.eigenvector_values = nx.eigenvector_centrality(self.G, max_iter=100)
                
            else:
                # Use CPU for all measures
                logger.info("Computing centrality measures on CPU...")
                self.pagerank_values = nx.pagerank(self.G, max_iter=50)
                
                # Only compute expensive measures for smaller graphs
                if len(self.G.nodes) < 10000:
                    self.betweenness_values = nx.betweenness_centrality(self.G, k=min(100, len(self.G.nodes)))
                    self.closeness_values = nx.closeness_centrality(self.G)
                    self.eigenvector_values = nx.eigenvector_centrality(self.G, max_iter=100)
                else:
                    # For large graphs, set default values
                    self.betweenness_values = {node: 0.0 for node in self.G.nodes}
                    self.closeness_values = {node: 0.0 for node in self.G.nodes}
                    self.eigenvector_values = {node: 0.0 for node in self.G.nodes}
                    
        except Exception as e:
            logger.error(f"Failed to compute centrality measures: {e}")
            # Fallback to zero values
            self.pagerank_values = {node: 0.0 for node in self.G.nodes}
            self.betweenness_values = {node: 0.0 for node in self.G.nodes}
            self.closeness_values = {node: 0.0 for node in self.G.nodes}
            self.eigenvector_values = {node: 0.0 for node in self.G.nodes}
        
        logger.info("Centrality measures pre-computed successfully!")

    def _calculate_baseline_stats(self) -> Dict[str, float]:
        """Calculate baseline statistics from relevant documents."""
        if not self.relevant_documents:
            return {}
        
        logger.info("Calculating baseline statistics...")
        
        # Pre-compute centrality measures once
        self._precompute_centrality_measures()
        
        # Determine which documents to use for baseline calculation
        if self.baseline_sample_size is not None:
            # Use specified sample size
            sample_size = min(self.baseline_sample_size, len(self.relevant_documents))
            sample_docs = list(self.relevant_documents)[:sample_size]
            logger.info(f"Using sample of {sample_size} documents for baseline (requested: {self.baseline_sample_size})")
        else:
            # Use all relevant documents by default
            sample_docs = list(self.relevant_documents)
            logger.info(f"Using all {len(sample_docs)} relevant documents for baseline")
        
        baseline_features = []
        for doc_id in tqdm(sample_docs, desc="Computing baseline features"):
            if doc_id in self.G.nodes:
                features = self._extract_single_document_features(doc_id)
                baseline_features.append(features)
        
        if not baseline_features:
            return {}
        
        # Calculate statistics
        feature_df = pd.DataFrame(baseline_features)
        baseline = {}
        
        for col in feature_df.columns:
            if col != 'openalex_id' and feature_df[col].dtype in ['int64', 'float64']:
                baseline[f'mean_{col}'] = float(feature_df[col].mean())
                baseline[f'std_{col}'] = max(float(feature_df[col].std()), 0.01)
        
        logger.info(f"Baseline calculated from {len(baseline_features)} documents")
        return baseline
    
    def _extract_single_document_features(self, doc_id: str) -> Dict[str, Any]:
        """Extract all features for a single document."""
        if doc_id not in self.G.nodes:
            return self._get_zero_features(doc_id)
        
        features = {'openalex_id': doc_id}
        
        # Basic connectivity features
        features.update(self._get_connectivity_features(doc_id))
        
        # Advanced network features
        features.update(self._get_advanced_features(doc_id))
        
        # Semantic features if available
        if self.embeddings is not None:
            features.update(self._get_semantic_features(doc_id))
        
        return features
    
    def _get_connectivity_features(self, doc_id: str) -> Dict[str, float]:
        """Get basic connectivity features for a document."""
        degree = self.G.degree(doc_id)
        
        # Edge type specific degrees
        citation_degree = sum(1 for _, _, d in self.G.edges(doc_id, data=True) 
                             if d.get('edge_type') == 'citation')
        semantic_degree = sum(1 for _, _, d in self.G.edges(doc_id, data=True) 
                             if d.get('edge_type') == 'semantic')
        cocitation_degree = sum(1 for _, _, d in self.G.edges(doc_id, data=True) 
                               if d.get('edge_type') == 'co_citation')
        
        # Connections to relevant documents
        relevant_neighbors = len([n for n in self.G.neighbors(doc_id) 
                                 if n in self.relevant_documents])
        
        return {
            'degree': float(degree),
            'citation_degree': float(citation_degree),
            'semantic_degree': float(semantic_degree),
            'cocitation_degree': float(cocitation_degree),
            'relevant_neighbors': float(relevant_neighbors),
            'relevant_ratio': float(relevant_neighbors / max(1, degree))
        }
    
    def _get_advanced_features(self, doc_id: str) -> Dict[str, float]:
        """Get advanced network features for a document."""
        try:
            # Clustering coefficient (this is relatively fast to compute per node)
            clustering = nx.clustering(self.G, doc_id)
            
            # Use pre-computed centrality measures
            betweenness_val = getattr(self, 'betweenness_values', {}).get(doc_id, 0.0)
            closeness_val = getattr(self, 'closeness_values', {}).get(doc_id, 0.0)
            eigenvector_val = getattr(self, 'eigenvector_values', {}).get(doc_id, 0.0)
            pagerank_val = getattr(self, 'pagerank_values', {}).get(doc_id, 0.0)
            
        except Exception as e:
            logger.debug(f"Failed to compute advanced features for {doc_id}: {e}")
            clustering = 0.0
            betweenness_val = 0.0
            closeness_val = 0.0
            eigenvector_val = 0.0
            pagerank_val = 0.0
        
        return {
            'clustering': float(clustering),
            'betweenness': float(betweenness_val),
            'closeness': float(closeness_val),
            'eigenvector': float(eigenvector_val),
            'pagerank': float(pagerank_val)
        }
    
    def _get_semantic_features(self, doc_id: str) -> Dict[str, float]:
        """Get semantic similarity features for a document."""
        if not self.embeddings_metadata or doc_id not in self.embeddings_metadata.get('openalex_id', []):
            return {
                'avg_semantic_similarity': 0.0,
                'max_semantic_similarity': 0.0,
                'semantic_isolation': 1.0
            }
        
        try:
            # Get document embedding
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata['openalex_id'])}
            doc_idx = id_to_idx[doc_id]
            doc_embedding = self.embeddings[doc_idx:doc_idx+1]
            
            # Get embeddings of relevant documents
            relevant_with_embeddings = [doc for doc in self.relevant_documents if doc in id_to_idx]
            
            if not relevant_with_embeddings:
                return {
                    'avg_semantic_similarity': 0.0,
                    'max_semantic_similarity': 0.0,
                    'semantic_isolation': 1.0
                }
            
            relevant_indices = [id_to_idx[doc] for doc in relevant_with_embeddings]
            relevant_embeddings = self.embeddings[relevant_indices]
            
            # Compute similarities
            similarities = cosine_similarity(doc_embedding, relevant_embeddings)[0]
            
            return {
                'avg_semantic_similarity': float(np.mean(similarities)),
                'max_semantic_similarity': float(np.max(similarities)),
                'semantic_isolation': float(1.0 - np.mean(similarities))
            }
            
        except Exception as e:
            logger.debug(f"Failed to compute semantic features for {doc_id}: {e}")
            return {
                'avg_semantic_similarity': 0.0,
                'max_semantic_similarity': 0.0,
                'semantic_isolation': 1.0
            }
    
    def _get_zero_features(self, doc_id: str) -> Dict[str, Any]:
        """Get zero features for documents not in the network."""
        return {
            'openalex_id': doc_id,
            'degree': 0.0,
            'citation_degree': 0.0,
            'semantic_degree': 0.0,
            'cocitation_degree': 0.0,
            'relevant_neighbors': 0.0,
            'relevant_ratio': 0.0,
            'clustering': 0.0,
            'betweenness': 0.0,
            'closeness': 0.0,
            'eigenvector': 0.0,
            'pagerank': 0.0,
            'avg_semantic_similarity': 0.0,
            'max_semantic_similarity': 0.0,
            'semantic_isolation': 1.0
        }
    
    def _train_outlier_detection(self):
        """Train outlier detection models on network features."""
        if not self.G:
            logger.warning("No graph available for outlier detection training")
            return
        
        logger.info("Extracting node features for outlier detection...")
        
        # Get a sample of nodes to avoid memory issues with large graphs
        max_nodes = 10000
        if len(self.G.nodes) > max_nodes:
            sample_nodes = list(np.random.choice(list(self.G.nodes), max_nodes, replace=False))
        else:
            sample_nodes = list(self.G.nodes)
        
        # Extract features for all sample nodes
        features = []
        feature_names = []
        
        for i, node in enumerate(tqdm(sample_nodes, desc="Extracting features")):
            node_features = self._extract_comprehensive_features(node)
            if i == 0:
                feature_names = list(node_features.keys())
            features.append(list(node_features.values()))
        
        # Convert to numpy array and handle missing values
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if features.shape[0] == 0 or features.shape[1] == 0:
            logger.warning("No valid features extracted for outlier detection")
            return
        
        # Train outlier detection models
        self.outlier_detector.fit(features)
        
        # Predict outliers on the training set for baseline
        self.outlier_results = self.outlier_detector.predict_outliers(features)
        
        logger.info(f"Outlier detection models trained on {features.shape[0]} nodes with {features.shape[1]} features")

    def _extract_comprehensive_features(self, node: str) -> Dict[str, float]:
        """Extract comprehensive features for outlier detection."""
        features = {}
        
        # Basic connectivity features
        features['degree'] = self.G.degree(node)
        features['clustering'] = nx.clustering(self.G, node)
        
        # Centrality measures (use pre-computed if available)
        features['pagerank'] = self.pagerank_values.get(node, 0.0)
        features['betweenness'] = self.betweenness_values.get(node, 0.0)
        features['closeness'] = self.closeness_values.get(node, 0.0)
        features['eigenvector'] = self.eigenvector_values.get(node, 0.0)
        
        # Neighbor-based features
        neighbors = list(self.G.neighbors(node))
        features['num_neighbors'] = len(neighbors)
        
        if neighbors:
            neighbor_degrees = [self.G.degree(n) for n in neighbors]
            features['mean_neighbor_degree'] = np.mean(neighbor_degrees)
            features['max_neighbor_degree'] = np.max(neighbor_degrees)
            features['min_neighbor_degree'] = np.min(neighbor_degrees)
            
            # Relevant neighbor ratio
            relevant_neighbors = sum(1 for n in neighbors if n in self.relevant_documents)
            features['relevant_neighbor_ratio'] = relevant_neighbors / len(neighbors)
        else:
            features['mean_neighbor_degree'] = 0.0
            features['max_neighbor_degree'] = 0.0
            features['min_neighbor_degree'] = 0.0
            features['relevant_neighbor_ratio'] = 0.0
        
        # Edge type distribution
        edge_types = defaultdict(int)
        for neighbor in neighbors:
            edge_data = self.G.get_edge_data(node, neighbor, {})
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_types[edge_type] += 1
        
        total_edges = len(neighbors)
        features['citation_edge_ratio'] = edge_types['citation'] / max(1, total_edges)
        features['semantic_edge_ratio'] = edge_types['semantic'] / max(1, total_edges)
        features['cocitation_edge_ratio'] = edge_types['co_citation'] / max(1, total_edges)
        features['bibliographic_edge_ratio'] = edge_types['bibliographic_coupling'] / max(1, total_edges)
        
        # Semantic features if available
        if self.embeddings is not None and self.embeddings_metadata is not None:
            semantic_similarities = self._compute_semantic_similarities_vectorized([node])
            features['semantic_similarity'] = semantic_similarities.get(node, 0.0)
        else:
            features['semantic_similarity'] = 0.0
        
        # Label information
        features['is_relevant'] = 1.0 if node in self.relevant_documents else 0.0
        
        return features

    def _detect_anomalous_citations(self):
        """Detect anomalous citation edges in the network."""
        if not self.G or len(self.G.edges) == 0:
            logger.warning("No edges available for anomaly detection - initializing empty results")
            self.anomalous_edges = {}
            return
        
        # Extract metadata for edge analysis
        metadata = {}
        if hasattr(self, 'simulation_data') and self.simulation_data is not None:
            for _, row in self.simulation_data.iterrows():
                doc_id = row['openalex_id']
                metadata[doc_id] = {
                    'publication_year': row.get('year', 0),
                    'title': row.get('title', ''),
                    'authors': row.get('authors', []),
                    'venue': row.get('venue', ''),
                    'affiliations': row.get('affiliations', []),
                    'citation_context': ''  # Would need additional data for this
                }
        
        # Detect anomalous edges
        self.anomalous_edges = self.edge_analyzer.detect_anomalous_edges(
            self.G, metadata, contamination=self.outlier_contamination
        )

    def get_outlier_documents(self, score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Get documents identified as outliers.
        
        Args:
            score_threshold: Optional threshold for outlier scores
            
        Returns:
            List of outlier document information
        """
        if self.outlier_results is None:
            logger.warning("Outlier detection has not been performed yet")
            return []
        
        outliers = []
        
        # Use ensemble predictions by default, or score threshold if provided
        if score_threshold is not None:
            outlier_mask = self.outlier_results['ensemble_scores'] > score_threshold
        else:
            outlier_mask = self.outlier_results['ensemble_predictions'] == 1
        
        sample_nodes = list(self.G.nodes)[:len(outlier_mask)]
        
        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier:
                node = sample_nodes[i]
                outlier_info = {
                    'document_id': node,
                    'lof_score': float(self.outlier_results['lof_scores'][i]),
                    'isolation_forest_score': float(self.outlier_results['isolation_forest_scores'][i]),
                    'ensemble_score': float(self.outlier_results['ensemble_scores'][i]),
                    'is_relevant': node in self.relevant_documents,
                    'degree': self.G.degree(node),
                    'clustering': nx.clustering(self.G, node)
                }
                outliers.append(outlier_info)
        
        # Sort by ensemble score (descending)
        outliers.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        return outliers

    def get_anomalous_citations(self, score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Get citation edges identified as anomalous.
        
        Args:
            score_threshold: Optional threshold for anomaly scores
            
        Returns:
            List of anomalous citation information
        """
        if self.anomalous_edges is None:
            logger.warning("Edge anomaly detection has not been performed yet")
            return []
        
        anomalous_citations = []
        
        for (citing, cited), edge_info in self.anomalous_edges.items():
            # Use ensemble predictions by default, or score threshold if provided
            if score_threshold is not None:
                is_anomalous = edge_info['ensemble_score'] > score_threshold
            else:
                is_anomalous = edge_info['is_anomalous_ensemble']
            
            if is_anomalous:
                citation_info = {
                    'citing_document': citing,
                    'cited_document': cited,
                    'lof_score': edge_info['lof_score'],
                    'isolation_forest_score': edge_info['isolation_forest_score'],
                    'ensemble_score': edge_info['ensemble_score'],
                    'features': edge_info['features'],
                    'citing_is_relevant': citing in self.relevant_documents,
                    'cited_is_relevant': cited in self.relevant_documents
                }
                anomalous_citations.append(citation_info)
        
        # Sort by ensemble score (descending)
        anomalous_citations.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        return anomalous_citations

    def extract_features(self, target_documents: List[str]) -> pd.DataFrame:
        """
        Extract citation network features for target documents.
        
        Args:
            target_documents: List of document IDs to extract features for
        
        Returns:
            DataFrame with citation network features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
        logger.info(f"Extracting features for {len(target_documents)} documents")
        
        features = []
        for doc_id in tqdm(target_documents, desc="Extracting features"):
            doc_features = self._extract_single_document_features(doc_id)
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Generate citation-based relevance scores for target documents.
        
        Args:
            target_documents: List of document IDs to score
        
        Returns:
            Dictionary mapping document IDs to relevance scores (0-1)
        """
        if not self.is_fitted or not self.baseline_stats:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        logger.info(f"Computing relevance scores for {len(target_documents)} documents")
        
        # Get base relevance scores
        base_scores = self._predict_relevance_scores_batch(target_documents)
        
        # Apply outlier detection adjustments
        adjusted_scores = self._apply_outlier_adjustments(base_scores, target_documents)
        
        return adjusted_scores

    def _apply_outlier_adjustments(self, base_scores: Dict[str, float], 
                                 target_documents: List[str]) -> Dict[str, float]:
        """
        Apply outlier detection results to adjust relevance scores.
        
        Args:
            base_scores: Base relevance scores
            target_documents: List of target documents
            
        Returns:
            Adjusted relevance scores incorporating outlier information
        """
        adjusted_scores = base_scores.copy()
        
        # Get outlier information for target documents
        if self.outlier_results is not None:
            # Extract features for target documents for outlier assessment
            target_features = []
            valid_docs = []
            
            for doc_id in target_documents:
                if doc_id in self.G.nodes:
                    features = self._extract_comprehensive_features(doc_id)
                    target_features.append(list(features.values()))
                    valid_docs.append(doc_id)
            
            if target_features:
                # Convert to numpy and predict outliers
                target_features = np.array(target_features)
                target_features = np.nan_to_num(target_features, nan=0.0, posinf=0.0, neginf=0.0)
                
                try:
                    outlier_results = self.outlier_detector.predict_outliers(target_features)
                    
                    # Apply outlier penalties
                    for i, doc_id in enumerate(valid_docs):
                        ensemble_score = outlier_results['ensemble_scores'][i]
                        is_outlier = outlier_results['ensemble_predictions'][i] == 1
                        
                        if is_outlier:
                            # Reduce relevance score for outliers
                            penalty = min(0.5, ensemble_score * 0.3)  # Max 50% penalty
                            adjusted_scores[doc_id] = base_scores.get(doc_id, 0.0) * (1 - penalty)
                        else:
                            # Slight boost for non-outliers in well-connected components
                            if ensemble_score < 0.3:  # Low outlier score = normal behavior
                                boost = 0.05
                                adjusted_scores[doc_id] = min(1.0, base_scores.get(doc_id, 0.0) * (1 + boost))
                            
                except Exception as e:
                    logger.debug(f"Failed to apply outlier adjustments: {e}")
        
        # Check for anomalous citations
        if self.anomalous_edges is not None:
            for doc_id in target_documents:
                # Count anomalous citations involving this document
                anomalous_count = 0
                total_edges = 0
                
                for (citing, cited), edge_info in self.anomalous_edges.items():
                    if citing == doc_id or cited == doc_id:
                        total_edges += 1
                        if edge_info['is_anomalous_ensemble']:
                            anomalous_count += 1
                
                if total_edges > 0:
                    anomalous_ratio = anomalous_count / total_edges
                    if anomalous_ratio > 0.3:  # More than 30% anomalous citations
                        penalty = min(0.3, anomalous_ratio * 0.5)
                        current_score = adjusted_scores.get(doc_id, 0.0)
                        adjusted_scores[doc_id] = current_score * (1 - penalty)
        
        return adjusted_scores
    
    def _predict_relevance_scores_batch(self, target_documents: List[str]) -> Dict[str, float]:
        """
        GPU-accelerated and parallelized batch scoring for large document lists.
        """
        logger.info("Using GPU-accelerated parallel batch scoring...")
        
        # Pre-compute semantic similarities with GPU acceleration if available
        semantic_similarities = {}
        if self.embeddings is not None and self.embeddings_metadata is not None:
            logger.info("Pre-computing semantic similarities...")
            semantic_similarities = self._compute_semantic_similarities_gpu_batch(target_documents)
        
        # Prepare data for parallel processing
        docs_in_network = [doc_id for doc_id in target_documents if doc_id in self.G.nodes]
        docs_not_in_network = [doc_id for doc_id in target_documents if doc_id not in self.G.nodes]
        
        logger.info(f"Processing {len(docs_in_network)} docs in network, {len(docs_not_in_network)} outside network")
        
        # Process documents in parallel using multiprocessing
        scores = {}
        
        # Zero scores for documents not in network
        for doc_id in docs_not_in_network:
            scores[doc_id] = 0.0
        
        if docs_in_network:
            # For small datasets, use single-threaded processing to avoid overhead
            if len(docs_in_network) < 100:
                logger.info("Using single-threaded processing for small dataset")
                for doc_id in tqdm(docs_in_network, desc="Computing scores"):
                    features = self._get_features_fast(doc_id, semantic_similarities)
                    scores[doc_id] = self._calculate_relevance_score_fast(features)
            else:
                # Use multiprocessing for CPU-intensive graph operations on larger datasets
                num_workers = min(mp.cpu_count(), 15)  # Use up to 15 cores
                batch_size = max(20, len(docs_in_network) // (num_workers * 4))  # Smaller batches for better parallelization
                
                logger.info(f"Using {num_workers} workers with batch size {batch_size}")
                
                # Split documents into chunks for parallel processing
                doc_chunks = [docs_in_network[i:i+batch_size] for i in range(0, len(docs_in_network), batch_size)]
                
                # Prepare shared data for workers
                worker_data = {
                    'pagerank_values': self.pagerank_values,
                    'relevant_documents': self.relevant_documents,
                    'semantic_similarities': semantic_similarities,
                    'graph_edges': list(self.G.edges(data=True)),  # Serialize graph data
                    'graph_nodes': list(self.G.nodes(data=True))
                }
                
                # Process chunks in parallel
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    worker_func = partial(_process_document_chunk, worker_data)
                    chunk_results = list(tqdm(
                        executor.map(worker_func, doc_chunks),
                        total=len(doc_chunks),
                        desc="Parallel scoring"
                    ))
                
                # Combine results
                for chunk_scores in chunk_results:
                    scores.update(chunk_scores)
        
        # Normalize scores using vectorized operations
        if scores:
            score_values = np.array(list(scores.values()))
            if score_values.std() > 1e-8:  # Avoid division by zero with small epsilon
                score_range = score_values.max() - score_values.min()
                if score_range > 1e-8:  # Additional check for valid range
                    score_values = (score_values - score_values.min()) / score_range
                    scores = dict(zip(scores.keys(), score_values))
                else:
                    logger.info("All scores are identical, keeping original values")
        
        return scores
    
    def _get_features_fast(self, doc_id: str, semantic_similarities: Dict[str, float]) -> Dict[str, float]:
        """Fast feature extraction using pre-computed values."""
        if doc_id not in self.G.nodes:
            return {'degree': 0.0, 'relevant_ratio': 0.0, 'clustering': 0.0, 
                   'pagerank': 0.0, 'semantic_isolation': 1.0}
        
        # Basic connectivity (fast)
        degree = self.G.degree(doc_id)
        relevant_neighbors = len([n for n in self.G.neighbors(doc_id) 
                                 if n in self.relevant_documents])
        relevant_ratio = relevant_neighbors / max(1, degree)
        
        # Pre-computed centrality measures
        clustering = nx.clustering(self.G, doc_id)  # Fast for single node
        pagerank = self.pagerank_values.get(doc_id, 0.0)
        
        # Pre-computed semantic similarity
        semantic_isolation = 1.0 - semantic_similarities.get(doc_id, 0.0)
        
        return {
            'degree': float(degree),
            'relevant_ratio': float(relevant_ratio),
            'clustering': float(clustering),
            'pagerank': float(pagerank),
            'semantic_isolation': float(semantic_isolation)
        }
    
    def _calculate_relevance_score_fast(self, features: Dict[str, float]) -> float:
        """Fast relevance score calculation with essential features only."""
        # Connectivity-based scoring
        degree_score = min(1.0, features['degree'] / 10.0)
        relevant_ratio_score = features['relevant_ratio']
        
        # Advanced network measures
        clustering_score = features['clustering']
        pagerank_score = min(1.0, features['pagerank'] * 1000)
        
        # Semantic features
        semantic_score = 1.0 - features['semantic_isolation']
        
        # Adaptive weighting
        if hasattr(self, 'relevant_documents') and self.relevant_documents:
            relevant_ratio = len(self.relevant_documents) / len(self.G.nodes) if self.G.nodes else 0
            sparsity_factor = 1 - min(0.9, max(0.1, relevant_ratio * 10))
            network_weight = 0.4 + sparsity_factor * 0.3
            semantic_weight = 0.6 - sparsity_factor * 0.3
        else:
            network_weight = 0.5
            semantic_weight = 0.5
        
        # Combine scores
        network_component = (degree_score * 0.3 + relevant_ratio_score * 0.4 + 
                           clustering_score * 0.2 + pagerank_score * 0.1)
        
        score = network_weight * network_component + semantic_weight * semantic_score
        
        return float(max(0.0, min(1.0, score)))
    
    def _compute_semantic_similarities_gpu_batch(self, target_documents: List[str]) -> Dict[str, float]:
        """GPU-accelerated batch computation of semantic similarities."""
        if not self.embeddings_metadata:
            return {}
        
        try:
            # Try to use GPU-accelerated similarity computation
            if CUGRAPH_AVAILABLE:
                import cupy as cp
                return self._compute_semantic_similarities_cupy(target_documents)
        except ImportError:
            logger.info("CuPy not available, using CPU with vectorized operations")
        
        # Fallback to optimized CPU version with vectorized operations
        return self._compute_semantic_similarities_vectorized(target_documents)
    
    def _compute_semantic_similarities_cupy(self, target_documents: List[str]) -> Dict[str, float]:
        """CuPy/GPU-accelerated semantic similarity computation."""
        import cupy as cp
        
        similarities = {}
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata['openalex_id'])}
        
        # Get relevant document embeddings once
        relevant_with_embeddings = [doc for doc in self.relevant_documents if doc in id_to_idx]
        if not relevant_with_embeddings:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        # Move to GPU
        relevant_indices = [id_to_idx[doc] for doc in relevant_with_embeddings]
        relevant_embeddings_gpu = cp.asarray(self.embeddings[relevant_indices])
        
        # Get target document indices
        target_indices = [id_to_idx[doc] for doc in target_documents if doc in id_to_idx]
        target_docs_with_embeddings = [doc for doc in target_documents if doc in id_to_idx]
        
        if target_indices:
            # Batch compute similarities on GPU
            target_embeddings_gpu = cp.asarray(self.embeddings[target_indices])
            
            # Compute cosine similarities using GPU matrix multiplication
            similarities_matrix = cp.dot(target_embeddings_gpu, relevant_embeddings_gpu.T)
            # Normalize
            target_norms = cp.linalg.norm(target_embeddings_gpu, axis=1, keepdims=True)
            relevant_norms = cp.linalg.norm(relevant_embeddings_gpu, axis=1, keepdims=True)
            similarities_matrix = similarities_matrix / (target_norms * relevant_norms.T)
            
            # Compute mean similarities and transfer back to CPU
            mean_similarities = cp.mean(similarities_matrix, axis=1).get()
            
            # Map back to document IDs
            for i, doc_id in enumerate(target_docs_with_embeddings):
                similarities[doc_id] = float(mean_similarities[i])
        
        # Set zero similarities for documents without embeddings
        for doc_id in target_documents:
            if doc_id not in similarities:
                similarities[doc_id] = 0.0
        
        return similarities
    
    def _compute_semantic_similarities_vectorized(self, target_documents: List[str]) -> Dict[str, float]:
        """Vectorized CPU computation of semantic similarities."""
        similarities = {}
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.embeddings_metadata['openalex_id'])}
        
        # Get relevant document embeddings once
        relevant_with_embeddings = [doc for doc in self.relevant_documents if doc in id_to_idx]
        if not relevant_with_embeddings:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        relevant_indices = [id_to_idx[doc] for doc in relevant_with_embeddings]
        relevant_embeddings = self.embeddings[relevant_indices]
        
        # Get target document indices for batch processing
        target_indices = [id_to_idx[doc] for doc in target_documents if doc in id_to_idx]
        target_docs_with_embeddings = [doc for doc in target_documents if doc in id_to_idx]
        
        if target_indices:
            # Batch compute similarities using vectorized operations
            target_embeddings = self.embeddings[target_indices]
            
            # Use sklearn's cosine_similarity for efficient batch computation
            similarities_matrix = cosine_similarity(target_embeddings, relevant_embeddings)
            mean_similarities = np.mean(similarities_matrix, axis=1)
            
            # Map back to document IDs
            for i, doc_id in enumerate(target_docs_with_embeddings):
                similarities[doc_id] = float(mean_similarities[i])
        
        # Set zero similarities for documents without embeddings
        for doc_id in target_documents:
            if doc_id not in similarities:
                similarities[doc_id] = 0.0
        
        return similarities
    
    def _calculate_relevance_score(self, features: pd.Series) -> float:
        """Calculate relevance score for a single document based on its features."""
        score = 0.0
        
        # Connectivity-based scoring
        degree_score = min(1.0, features['degree'] / 10.0)  # Normalize degree
        relevant_ratio_score = features['relevant_ratio']
        
        # Advanced network measures
        clustering_score = features['clustering']
        pagerank_score = min(1.0, features['pagerank'] * 1000)  # Scale pagerank
        
        # Semantic features
        semantic_score = 1.0 - features['semantic_isolation']  # Higher similarity = higher score
        
        # Adaptive weighting based on dataset characteristics
        if hasattr(self, 'relevant_documents') and self.relevant_documents:
            relevant_ratio = len(self.relevant_documents) / len(self.G.nodes) if self.G.nodes else 0
            sparsity_factor = 1 - min(0.9, max(0.1, relevant_ratio * 10))
            
            # For sparse datasets, emphasize network structure more
            network_weight = 0.4 + sparsity_factor * 0.3
            semantic_weight = 0.6 - sparsity_factor * 0.3
        else:
            network_weight = 0.5
            semantic_weight = 0.5
        
        # Combine scores
        network_component = (degree_score * 0.3 + relevant_ratio_score * 0.4 + 
                           clustering_score * 0.2 + pagerank_score * 0.1)
        
        score = network_weight * network_component + semantic_weight * semantic_score
        
        return float(max(0.0, min(1.0, score)))  # Clamp to [0,1]
    
    def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Provide detailed analysis of a specific document.
        
        Args:
            doc_id: Document ID to analyze
        
        Returns:
            Dictionary with detailed analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing documents")
        
        # Use the same method as batch processing for consistency
        semantic_similarities = self._compute_semantic_similarities_gpu_batch([doc_id])
        features = self._get_features_fast(doc_id, semantic_similarities)
        score = self._calculate_relevance_score_fast(features)
        
        analysis = {
            'document_id': doc_id,
            'relevance_score': score,
            'features': features,
            'in_network': doc_id in self.G.nodes,
            'has_embeddings': (self.embeddings_metadata and 
                             doc_id in self.embeddings_metadata.get('openalex_ids', [])),
        }
        
        if doc_id in self.G.nodes:
            neighbors = list(self.G.neighbors(doc_id))
            analysis['neighbors'] = neighbors[:10]  # Show first 10 neighbors
            analysis['neighbor_count'] = len(neighbors)
            
            # Edge type breakdown
            edge_types = defaultdict(int)
            for neighbor in neighbors:
                edge_data = self.G.get_edge_data(doc_id, neighbor)
                edge_type = edge_data.get('edge_type', 'unknown')
                edge_types[edge_type] += 1
            analysis['edge_types'] = dict(edge_types)
        
        return analysis


def main():
    """Example usage of the CitationNetworkModel."""
    # Example with dataset selection
    model = CitationNetworkModel()
    
    # Load available datasets
    datasets_config = model._load_datasets_config()
    dataset_names = list(datasets_config.keys())
    
    print("Available datasets:")
    for i, name in enumerate(dataset_names, 1):
        print(f"{i}. {name}")
    
    # Select dataset
    try:
        choice = int(input("\nSelect dataset (enter number): ")) - 1
        if 0 <= choice < len(dataset_names):
            dataset_name = dataset_names[choice]
            
            # Fit model
            print(f"\nFitting model on {dataset_name}...")
            model.dataset_name = dataset_name
            model.dataset_config = datasets_config[dataset_name]
            model.fit()
            
            # Load simulation data to get some example documents
            simulation_df = model._load_simulation_data(dataset_name)
            example_docs = simulation_df['openalex_id'].head(10).tolist()
            
            # Extract features
            print("\nExtracting features...")
            features_df = model.extract_features(example_docs)
            print(features_df.head())
            
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


if __name__ == "__main__":
    main() 