"""
GPU-Accelerated Citation Network Model with Semantic Embeddings

This module provides citation-based features for identifying outlier documents
using cuGraph for GPU acceleration and SPECTER2 embeddings for semantic similarity.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
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


class CitationNetworkModel:
    """
    GPU-accelerated citation network model with semantic embeddings.
    
    Combines traditional citation analysis with semantic similarity from SPECTER2
    embeddings for enhanced outlier detection in systematic reviews.
    """
    
    def __init__(self, dataset_name: Optional[str] = None, 
                 enable_gpu: bool = True,
                 enable_semantic: bool = True,
                 baseline_sample_size: Optional[int] = None):
        """
        Initialize the citation network model.
        
        Args:
            dataset_name: Name of dataset to use
            enable_gpu: Whether to use GPU acceleration if available
            enable_semantic: Whether to include semantic similarity features
            baseline_sample_size: Optional sample size for baseline calculation (None = use all)
        """
        self.dataset_name = dataset_name
        self.enable_gpu = enable_gpu and CUGRAPH_AVAILABLE
        self.enable_semantic = enable_semantic
        self.baseline_sample_size = baseline_sample_size
        
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
            with open(metadata_path, 'r') as f:
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
        
        # Calculate baseline statistics
        self.baseline_stats = self._calculate_baseline_stats()
        
        self.is_fitted = True
        
        # Log fitting results
        build_time = time.time() - start_time
        edge_types = self._get_edge_type_distribution()
        
        logger.info(f"Citation network fitted in {build_time:.2f}s")
        logger.info(f"Network: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        logger.info(f"Edge types: {dict(edge_types)}")
        logger.info(f"Relevant documents: {len(self.relevant_documents)}")
        
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
                            # Split by comma or semicolon
                            references = [ref.strip() for ref in row['references'].split(',')]
                    elif isinstance(row['references'], list):
                        references = row['references']
                    else:
                        references = []
                    
                    for ref_id in references:
                        if ref_id in G.nodes:
                            G.add_edge(doc_id, ref_id, edge_type='citation', weight=1.0)
                            edge_counts['citation'] += 1
                            
                except Exception as e:
                    logger.debug(f"Failed to parse references for {doc_id}: {e}")
        
        # 2. Add co-citation edges (documents citing the same papers)
        logger.info("Adding co-citation edges...")
        doc_citations = defaultdict(set)
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'citation':
                doc_citations[u].add(v)
        
        # Create co-citation edges
        doc_list = list(doc_citations.keys())
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
        
        # Use optimized batch scoring for large document lists
        if len(target_documents) > 500:
            return self._predict_relevance_scores_batch(target_documents)
        else:
            # Use regular method for smaller lists
            features_df = self.extract_features(target_documents)
            
            # Calculate deviation scores
            scores = {}
            for _, row in features_df.iterrows():
                doc_id = row['openalex_id']
                score = self._calculate_relevance_score(row)
                scores[doc_id] = score
            
            # Normalize scores to 0-1 range
            if scores:
                max_score = max(scores.values())
                min_score = min(scores.values())
                score_range = max_score - min_score
                
                if score_range > 0:
                    for doc_id in scores:
                        scores[doc_id] = (scores[doc_id] - min_score) / score_range
            
            return scores
    
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
            # Use multiprocessing for CPU-intensive graph operations
            num_workers = min(mp.cpu_count(), 15)  # Use up to 15 cores as you mentioned
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
            if score_values.std() > 0:  # Avoid division by zero
                score_values = (score_values - score_values.min()) / (score_values.max() - score_values.min())
                scores = dict(zip(scores.keys(), score_values))
        
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
        
        features = self._extract_single_document_features(doc_id)
        score = self._calculate_relevance_score(pd.Series(features))
        
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