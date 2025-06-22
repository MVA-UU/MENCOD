"""
Citation Network Construction Module

Handles building and managing citation networks from simulation data
using Synergy datasets for citation relationships.
"""

import pandas as pd
import networkx as nx
import logging
from typing import Dict, Optional, Set
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
        Build citation network from the FULL synergy dataset but mark simulation papers.
        
        This builds a complete citation network using all papers from the synergy dataset
        to achieve better connectivity, while marking which papers are eligible for 
        outlier detection (i.e., papers from the simulation dataset).
        
        Args:
            simulation_df: DataFrame with simulation papers (subset for outlier detection)
            dataset_name: Dataset name for loading the full synergy citation data
            
        Returns:
            NetworkX directed graph with full dataset but simulation papers marked
        """
        G = nx.DiGraph()
        
        # Load full synergy dataset for comprehensive network building
        if not dataset_name:
            logger.warning("No dataset name provided - falling back to simulation-only network")
            return self._build_simulation_only_network(simulation_df, dataset_name)
        
        synergy_data = load_synergy_dataset(dataset_name)
        if synergy_data is None:
            logger.warning("No synergy data available - falling back to simulation-only network")
            return self._build_simulation_only_network(simulation_df, dataset_name)
        
        # Get set of simulation paper IDs for marking eligibility
        simulation_ids = set(simulation_df['openalex_id'].tolist())
        logger.info(f"Simulation dataset contains {len(simulation_ids)} papers")
        logger.info(f"Full synergy dataset contains {len(synergy_data)} papers")
        
        # Add ALL documents from full synergy dataset as nodes
        self._add_full_dataset_nodes(G, synergy_data, simulation_ids)
        logger.info(f"Created graph with {len(G.nodes)} nodes from full dataset")
        
        # Add citation edges from full dataset
        citation_count = self._add_citation_edges_full_dataset(G, synergy_data)
        logger.info(f"Added {citation_count} citation edges from full dataset")
        
        # Log network connectivity stats
        self._log_network_stats(G, simulation_ids)
        
        return G
    
    def _build_simulation_only_network(self, simulation_df: pd.DataFrame, 
                                     dataset_name: str = None) -> nx.DiGraph:
        """Fallback method: build network only from simulation data (original behavior)."""
        logger.info("Building citation network from simulation data only")
        G = nx.DiGraph()
        
        # Add simulation documents as nodes with metadata
        self._add_document_nodes(G, simulation_df)
        logger.info(f"Created graph with {len(G.nodes)} nodes from simulation data")
        
        # Add citation edges from synergy dataset if available
        citation_count = 0
        if dataset_name:
            synergy_data = load_synergy_dataset(dataset_name)
            citation_count = self._add_citation_edges(G, simulation_df, synergy_data)
            logger.info(f"Added {citation_count} citation edges")
        
        return G
    
    def _add_full_dataset_nodes(self, G: nx.DiGraph, synergy_data: Dict, 
                              simulation_ids: Set[str]) -> None:
        """Add all documents from full synergy dataset as nodes with eligibility marking."""
        for paper_id, paper_data in synergy_data.items():
            # Determine if this paper is eligible for outlier detection
            is_simulation_eligible = paper_id in simulation_ids
            
            # Add node with comprehensive metadata
            G.add_node(
                paper_id,
                title=paper_data.get('title', ''),
                year=paper_data.get('year', 0),
                label=paper_data.get('label_included', 0),
                simulation_eligible=is_simulation_eligible,
                node_type='simulation' if is_simulation_eligible else 'background'
            )
    
    def _add_document_nodes(self, G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
        """Add document nodes to the graph with metadata (original method for fallback)."""
        for _, row in simulation_df.iterrows():
            G.add_node(
                row['openalex_id'],
                title=row.get('title', ''),
                year=row.get('year', 0),
                label=row.get('label_included', 0),
                simulation_eligible=True,  # All simulation papers are eligible
                node_type='simulation'
            )
    
    def _add_citation_edges_full_dataset(self, G: nx.DiGraph, synergy_data: Dict) -> int:
        """
        Add citation edges from full synergy dataset.
        
        Args:
            G: NetworkX graph to add edges to
            synergy_data: Dictionary with full synergy dataset citation information
            
        Returns:
            Number of citation edges added
        """
        citation_count = 0
        
        for paper_id, paper_data in tqdm(synergy_data.items(), 
                                       desc="Building full citation network", 
                                       total=len(synergy_data)):
            citing_paper = paper_id
            
            # Skip if citing paper is not in our graph
            if citing_paper not in G.nodes:
                continue
                
            referenced_works = paper_data.get('referenced_works', [])
            
            if referenced_works and isinstance(referenced_works, list):
                citation_count += self._process_citations_full_dataset(
                    G, citing_paper, referenced_works
                )
        
        return citation_count
    
    def _process_citations_full_dataset(self, G: nx.DiGraph, citing_paper: str, 
                                      referenced_works: list) -> int:
        """Process citations for a single paper in full dataset."""
        count = 0
        for ref_id in referenced_works:
            if (ref_id in G.nodes and 
                ref_id != citing_paper):
                
                if G.has_edge(citing_paper, ref_id):
                    G[citing_paper][ref_id]['weight'] += 1
                else:
                    G.add_edge(citing_paper, ref_id, 
                             edge_type='citation', 
                             weight=2.0)
                count += 1
        return count
    
    def _add_citation_edges(self, G: nx.DiGraph, simulation_df: pd.DataFrame, 
                          synergy_data: Optional[Dict]) -> int:
        """
        Add citation edges from synergy dataset (original method for fallback).
        
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
        """Process citations for a single paper (original method for fallback)."""
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
    
    def _log_network_stats(self, G: nx.DiGraph, simulation_ids: Set[str]) -> None:
        """Log network connectivity statistics."""
        simulation_nodes = [n for n in G.nodes if G.nodes[n].get('simulation_eligible', False)]
        background_nodes = [n for n in G.nodes if not G.nodes[n].get('simulation_eligible', False)]
        
        logger.info(f"Network composition:")
        logger.info(f"  - Simulation papers (eligible for ranking): {len(simulation_nodes)}")
        logger.info(f"  - Background papers (not eligible): {len(background_nodes)}")
        logger.info(f"  - Total papers: {len(G.nodes)}")
        logger.info(f"  - Total citations: {len(G.edges)}")
        
        # Check connectivity for simulation nodes
        if simulation_nodes:
            # Count edges involving simulation nodes
            sim_to_sim = sum(1 for u, v in G.edges() if u in simulation_ids and v in simulation_ids)
            sim_to_bg = sum(1 for u, v in G.edges() if u in simulation_ids and v not in simulation_ids)
            bg_to_sim = sum(1 for u, v in G.edges() if u not in simulation_ids and v in simulation_ids)
            
            logger.info(f"Citation patterns:")
            logger.info(f"  - Simulation → Simulation: {sim_to_sim}")
            logger.info(f"  - Simulation → Background: {sim_to_bg}")
            logger.info(f"  - Background → Simulation: {bg_to_sim}")
    
    def get_network_stats(self, G: nx.DiGraph) -> Dict[str, any]:
        """Get basic network statistics."""
        simulation_nodes = [n for n in G.nodes if G.nodes[n].get('simulation_eligible', False)]
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'simulation_eligible_nodes': len(simulation_nodes),
            'background_nodes': G.number_of_nodes() - len(simulation_nodes),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G.to_undirected()),
            'components': nx.number_connected_components(G.to_undirected())
        }
    
    def get_simulation_eligible_nodes(self, G: nx.DiGraph) -> list:
        """Get list of node IDs that are eligible for outlier detection."""
        return [n for n in G.nodes if G.nodes[n].get('simulation_eligible', False)] 