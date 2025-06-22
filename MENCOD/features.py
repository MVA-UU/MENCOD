"""
Feature Extraction Module

Handles extraction of network features, semantic features, and citation features
for outlier detection in citation networks.
"""

import pandas as pd
import numpy as np
import networkx as nx
import logging
from typing import Dict, Set, Optional
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Handles feature extraction for citation network analysis."""
    
    def __init__(self, embeddings: Optional[np.ndarray] = None, 
                 embeddings_metadata: Optional[Dict] = None):
        """
        Initialize feature extractor.
        
        Args:
            embeddings: Optional SPECTER2 embeddings array
            embeddings_metadata: Optional embeddings metadata
        """
        self.embeddings = embeddings
        self.embeddings_metadata = embeddings_metadata
    
    def extract_network_features(self, G: nx.DiGraph, 
                               simulation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive network and semantic features for simulation-eligible papers only.
        
        This method now works with the full citation network but only extracts features
        for papers in the simulation dataset that are eligible for outlier detection.
        
        Args:
            G: Citation network graph (includes full dataset with marked eligibility)
            simulation_df: DataFrame with simulation data (eligible papers only)
            
        Returns:
            DataFrame with extracted features for simulation papers only
        """
        features = []
        relevant_docs = set(
            simulation_df[simulation_df['label_included'] == 1]['openalex_id'].tolist()
        )
        
        # Get simulation-eligible nodes from the graph
        simulation_eligible_nodes = set(self._get_simulation_eligible_nodes(G))
        logger.info(f"Found {len(simulation_eligible_nodes)} simulation-eligible nodes in the network")
        
        # Verify that simulation_df papers are in the eligible set
        simulation_df_ids = set(simulation_df['openalex_id'].tolist())
        missing_ids = simulation_df_ids - simulation_eligible_nodes
        if missing_ids:
            logger.warning(f"Found {len(missing_ids)} simulation papers not marked as eligible in network")
        
        # Pre-compute centrality measures on the full network for better connectivity
        centrality_measures = self._compute_centrality_measures(G)
        
        # Extract features for each simulation document
        for _, row in tqdm(simulation_df.iterrows(), 
                          desc="Extracting features for simulation papers", 
                          total=len(simulation_df)):
            doc_id = row['openalex_id']
            doc_features = {'openalex_id': doc_id}
            
            # Network features (leveraging full network connectivity)
            network_features = self._get_network_features(
                G, doc_id, relevant_docs, centrality_measures
            )
            doc_features.update(network_features)
            
            # Semantic features (if embeddings available)
            semantic_features = self._get_semantic_features(doc_id, relevant_docs)
            doc_features.update(semantic_features)
            
            # Citation pattern features
            citation_features = self._get_citation_features(row)
            doc_features.update(citation_features)
            
            features.append(doc_features)
        
        logger.info(f"Extracted features for {len(features)} simulation papers using full network connectivity")
        return pd.DataFrame(features)
    
    def _compute_centrality_measures(self, G: nx.DiGraph) -> Dict[str, Dict]:
        """Compute centrality measures for the graph."""
        logger.info("Computing graph centrality measures...")
        
        if len(G.nodes) == 0:
            return {
                'pagerank': {}, 'betweenness': {}, 
                'closeness': {}, 'eigenvector': {}
            }
        
        # Dynamic parameters based on graph size
        n_nodes = len(G.nodes)
        
        # Scale max_iter dynamically with graph size for better convergence
        base_iter = 200
        max_iter = max(base_iter, int(np.log(n_nodes + 1) * 50))
        
        # Dynamic sample size for betweenness (use full computation for better accuracy)
        # Only use sampling if absolutely necessary for very large graphs
        betweenness_k = n_nodes if n_nodes <= 1000 else max(500, int(n_nodes * 0.1))
        
        logger.info(f"Computing centralities for {n_nodes} nodes with max_iter={max_iter}")
        
        # Always compute full centralities - no shortcuts or fallbacks
        pagerank = nx.pagerank(G, max_iter=max_iter, tol=1e-6)
        betweenness = nx.betweenness_centrality(G, k=betweenness_k, normalized=True)
        closeness = nx.closeness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=max_iter, tol=1e-6)
        
        logger.info("Successfully computed all centrality measures")
        
        return {
            'pagerank': pagerank,
            'betweenness': betweenness,
            'closeness': closeness,
            'eigenvector': eigenvector
        }
    
    def _get_network_features(self, G: nx.DiGraph, doc_id: str, 
                            relevant_docs: Set[str], 
                            centrality_measures: Dict[str, Dict]) -> Dict[str, float]:
        """Extract network-based features for a document."""
        if doc_id not in G.nodes:
            return self._get_zero_network_features()
        
        # Basic connectivity
        in_degree = G.in_degree(doc_id)
        out_degree = G.out_degree(doc_id)
        total_degree = in_degree + out_degree
        
        # Get neighbors
        neighbors = list(set(list(G.predecessors(doc_id)) + list(G.successors(doc_id))))
        relevant_neighbors = len([n for n in neighbors if n in relevant_docs])
        
        features = {
            'degree': float(total_degree),
            'in_degree': float(in_degree),
            'out_degree': float(out_degree),
            'relevant_neighbors': float(relevant_neighbors),
            'relevant_ratio': float(relevant_neighbors / max(1, len(neighbors))),
            'clustering': float(nx.clustering(G.to_undirected(), doc_id)),
            'pagerank': float(centrality_measures['pagerank'].get(doc_id, 0.0)),
            'betweenness': float(centrality_measures['betweenness'].get(doc_id, 0.0)),
            'closeness': float(centrality_measures['closeness'].get(doc_id, 0.0)),
            'eigenvector': float(centrality_measures['eigenvector'].get(doc_id, 0.0)),
        }
        
        # Neighborhood diversity
        if neighbors:
            neighbor_degrees = [G.degree(n) for n in neighbors]
            features.update({
                'mean_neighbor_degree': float(np.mean(neighbor_degrees)),
                'std_neighbor_degree': float(np.std(neighbor_degrees)),
                'max_neighbor_degree': float(np.max(neighbor_degrees)),
            })
        else:
            features.update({
                'mean_neighbor_degree': 0.0,
                'std_neighbor_degree': 0.0,
                'max_neighbor_degree': 0.0,
            })
        
        # Distance to relevant documents
        features['min_distance_to_relevant'] = self._compute_min_distance_to_relevant(
            G, doc_id, relevant_docs
        )
        
        return features
    
    def _get_zero_network_features(self) -> Dict[str, float]:
        """Return zero features for documents not in the graph."""
        return {
            'degree': 0.0, 'in_degree': 0.0, 'out_degree': 0.0,
            'relevant_neighbors': 0.0, 'relevant_ratio': 0.0,
            'clustering': 0.0, 'pagerank': 0.0, 'betweenness': 0.0,
            'closeness': 0.0, 'eigenvector': 0.0,
            'mean_neighbor_degree': 0.0, 'std_neighbor_degree': 0.0,
            'max_neighbor_degree': 0.0, 'min_distance_to_relevant': 10.0,
        }
    
    def _compute_min_distance_to_relevant(self, G: nx.DiGraph, doc_id: str, 
                                        relevant_docs: Set[str]) -> float:
        """Compute minimum distance to relevant documents."""
        if not relevant_docs:
            return 10.0
        
        min_distance = float('inf')
        
        # NO SUBSAMPLING - compute distances to ALL relevant documents for accuracy
        for rel_doc in relevant_docs:
            if rel_doc in G.nodes and rel_doc != doc_id:
                try:
                    dist = nx.shortest_path_length(G, doc_id, rel_doc)
                    min_distance = min(min_distance, dist)
                    # Early termination if we find distance 1 (direct neighbor)
                    if min_distance == 1:
                        break
                except nx.NetworkXNoPath:
                    continue
        
        return float(min_distance) if min_distance != float('inf') else 10.0
    
    def _get_semantic_features(self, doc_id: str, relevant_docs: Set[str]) -> Dict[str, float]:
        """Extract semantic similarity features using embeddings."""
        default_features = {
            'semantic_similarity_to_relevant': 0.0,
            'semantic_isolation_score': 1.0,
            'max_semantic_similarity': 0.0,
        }
        
        if self.embeddings is None or self.embeddings_metadata is None:
            return default_features
        
        # Create mapping from openalex_id to embedding index
        if 'documents' in self.embeddings_metadata:
            id_to_idx = {
                doc.get('openalex_id', ''): idx 
                for idx, doc in enumerate(self.embeddings_metadata['documents'])
            }
        else:
            id_to_idx = {
                doc_id: idx 
                for idx, doc_id in enumerate(self.embeddings_metadata.get('openalex_id', []))
            }
        
        if doc_id not in id_to_idx:
            return default_features
        
        try:
            # Get document embedding
            doc_idx = id_to_idx[doc_id]
            doc_embedding = self.embeddings[doc_idx:doc_idx+1]
            
            # Get relevant document embeddings
            relevant_with_embeddings = [
                doc for doc in relevant_docs 
                if doc in id_to_idx and doc != doc_id
            ]
            
            if relevant_with_embeddings:
                relevant_indices = [id_to_idx[doc] for doc in relevant_with_embeddings]
                relevant_embeddings = self.embeddings[relevant_indices]
                
                # Compute similarities
                similarities = cosine_similarity(doc_embedding, relevant_embeddings)[0]
                
                return {
                    'semantic_similarity_to_relevant': float(np.mean(similarities)),
                    'semantic_isolation_score': float(1.0 - np.mean(similarities)),
                    'max_semantic_similarity': float(np.max(similarities)),
                }
        except Exception as e:
            logger.debug(f"Failed to compute semantic features for {doc_id}: {e}")
        
        return default_features
    
    def _get_citation_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract citation-based features from paper metadata."""
        return {
            'publication_year': float(row.get('year', 0)),
            'title_length': float(len(str(row.get('title', '')))),
        }
    
    def _get_simulation_eligible_nodes(self, G: nx.DiGraph) -> list:
        """Get list of node IDs that are eligible for outlier detection."""
        return [n for n in G.nodes if G.nodes[n].get('simulation_eligible', False)] 