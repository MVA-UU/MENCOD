"""
Citation Network Construction Module

Handles building and managing citation networks from simulation data
using Synergy datasets for citation relationships.
"""

import pandas as pd
import networkx as nx
import logging
from typing import Dict, Optional
from tqdm import tqdm

from utils import load_synergy_dataset

logger = logging.getLogger(__name__)


class NetworkBuilder:
    """Handles citation network construction and management."""
    
    def __init__(self):
        """Initialize the network builder."""
        pass
    
    def build_citation_network(self, simulation_df: pd.DataFrame, 
                             dataset_name: str = None) -> nx.DiGraph:
        """
        Build citation network from simulation data.
        
        Args:
            simulation_df: DataFrame with paper data and citation information
            dataset_name: Optional dataset name for loading synergy citation data
            
        Returns:
            NetworkX directed graph representing the citation network
        """
        G = nx.DiGraph()
        
        # Add all documents as nodes with metadata
        self._add_document_nodes(G, simulation_df)
        logger.info(f"Created graph with {len(G.nodes)} nodes")
        
        # Add citation edges from synergy dataset if available
        citation_count = 0
        if dataset_name:
            synergy_data = load_synergy_dataset(dataset_name)
            citation_count = self._add_citation_edges(G, simulation_df, synergy_data)
            logger.info(f"Added {citation_count} citation edges")
        
        return G
    
    def _add_document_nodes(self, G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
        """Add document nodes to the graph with metadata."""
        for _, row in simulation_df.iterrows():
            G.add_node(
                row['openalex_id'],
                title=row.get('title', ''),
                year=row.get('year', 0),
                label=row.get('label_included', 0)
            )
    
    def _add_citation_edges(self, G: nx.DiGraph, simulation_df: pd.DataFrame, 
                          synergy_data: Optional[Dict]) -> int:
        """
        Add citation edges from synergy dataset.
        
        Args:
            G: NetworkX graph to add edges to
            simulation_df: DataFrame with simulation documents
            synergy_data: Dictionary with synergy dataset citation information
            
        Returns:
            Number of citation edges added
        """
        if synergy_data is None:
            logger.warning("No synergy data available - cannot add citation edges")
            return 0
        
        simulation_ids = set(simulation_df['openalex_id'].tolist())
        citation_count = 0
        
        for _, row in tqdm(simulation_df.iterrows(), 
                          desc="Building citation network", 
                          total=len(simulation_df)):
            citing_paper = row['openalex_id']
            
            if citing_paper in synergy_data:
                referenced_works = synergy_data[citing_paper].get('referenced_works', [])
                
                if referenced_works and isinstance(referenced_works, list):
                    citation_count += self._process_citations(
                        G, citing_paper, referenced_works, simulation_ids
                    )
        
        return citation_count
    
    def _process_citations(self, G: nx.DiGraph, citing_paper: str, 
                         referenced_works: list, simulation_ids: set) -> int:
        """Process citations for a single paper."""
        count = 0
        for ref_id in referenced_works:
            if (ref_id in simulation_ids and 
                ref_id != citing_paper and 
                ref_id in G.nodes):
                
                if G.has_edge(citing_paper, ref_id):
                    G[citing_paper][ref_id]['weight'] += 1
                else:
                    G.add_edge(citing_paper, ref_id, 
                             edge_type='citation', 
                             weight=2.0)
                count += 1
        return count
    
    def get_network_stats(self, G: nx.DiGraph) -> Dict[str, int]:
        """Get basic network statistics."""
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G.to_undirected()),
            'components': nx.number_connected_components(G.to_undirected())
        } 