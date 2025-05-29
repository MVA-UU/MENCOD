"""
Citation Network Model for Hybrid Outlier Detection

This module provides citation-based features for identifying outlier documents
that are missed by content-based ranking methods.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from synergy_dataset import Dataset


class CitationNetworkModel:
    """Citation-based feature extractor for outlier detection."""
    
    def __init__(self, dataset_name: str = "Appenzeller-Herzog_2019"):
        self.dataset_name = dataset_name
        self.G = None
        self.relevant_documents = set()
        self.baseline_stats = None
        self.is_fitted = False
    
    def fit(self, simulation_df: pd.DataFrame) -> 'CitationNetworkModel':
        """Build citation network and identify relevant documents."""
        print("Building citation network...")
        
        # Load dataset and build network
        appenzeller = Dataset(self.dataset_name)
        appenzeller_data = appenzeller.to_dict(["id", "title", "abstract", "referenced_works"])
        self.G = self._build_network(appenzeller_data)
        
        # Map simulation data and identify relevant documents
        self._map_simulation_data(simulation_df)
        self.relevant_documents = set([
            node for node in self.G.nodes 
            if self.G.nodes[node].get('label_included') == 1
        ])
        
        self.is_fitted = True
        self.baseline_stats = self._calculate_baseline_stats()
        
        print(f"Citation network built: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        print(f"Relevant documents identified: {len(self.relevant_documents)}")
        print(f"Baseline citation patterns calculated from {len(self.relevant_documents)} relevant docs")
        
        return self
    
    def get_citation_features(self, target_documents: List[str]) -> pd.DataFrame:
        """Extract citation-based features for target documents."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
        features = []
        for doc_id in target_documents:
            if doc_id not in self.G.nodes:
                features.append(self._get_zero_features(doc_id))
                continue
            
            doc_features = {
                'openalex_id': doc_id,
                **self._calc_connectivity_features(doc_id),
                **self._calc_coupling_features(doc_id),
                **self._calc_neighborhood_features(doc_id)
            }
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """Generate citation-based outlier scores using relative deviation from baseline."""
        if not self.is_fitted or not self.baseline_stats:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        features_df = self.get_citation_features(target_documents)
        scores = {}
        
        for _, row in features_df.iterrows():
            # Calculate deviation scores
            iso_score = self._calc_isolation_deviation(row, self.baseline_stats)
            coup_score = self._calc_coupling_deviation(row, self.baseline_stats)
            neigh_score = self._calc_neighborhood_deviation(row, self.baseline_stats)
            
            # Handle completely isolated documents
            if row['total_citations'] == 0:
                iso_score = min(1.0, iso_score + 0.2)
            
            # Combine with weights
            outlier_score = 0.5 * iso_score + 0.3 * coup_score + 0.2 * neigh_score
            scores[row['openalex_id']] = min(1.0, max(0.0, outlier_score))
        
        return scores
    
    def _build_network(self, appenzeller_data: Dict) -> nx.Graph:
        """Build citation network from dataset."""
        nodes = [(k, {"title": v.get("title", ""), "abstract": v.get("abstract", "")}) 
                 for k, v in appenzeller_data.items()]
        edges = [(k, r) for k, v in appenzeller_data.items() 
                for r in v.get("referenced_works", [])]
        
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        G.remove_nodes_from(set(G.nodes) - set([n[0] for n in nodes]))
        return G
    
    def _map_simulation_data(self, simulation_df: pd.DataFrame) -> None:
        """Map simulation results to network nodes."""
        for _, row in simulation_df.iterrows():
            if row['openalex_id'] in self.G.nodes:
                node_data = self.G.nodes[row['openalex_id']]
                node_data.update({
                    'record_id': row['record_id'],
                    'asreview_ranking': row['asreview_ranking'],
                    'label_included': row['label_included'],
                    'asreview_prior': row.get('asreview_prior', 0)
                })
    
    def _calc_connectivity_features(self, doc_id: str) -> Dict[str, float]:
        """Calculate direct citation connectivity features."""
        neighbors = list(self.G.neighbors(doc_id))
        if not neighbors:
            return {'total_citations': 0, 'relevant_connections': 0, 'relevant_connections_ratio': 0.0}
        
        rel_conn = sum(1 for n in neighbors if n in self.relevant_documents)
        return {
            'total_citations': len(neighbors),
            'relevant_connections': rel_conn,
            'relevant_connections_ratio': rel_conn / len(neighbors)
        }
    
    def _calc_coupling_features(self, doc_id: str) -> Dict[str, float]:
        """Calculate bibliographic coupling features."""
        doc_refs = set(self.G.neighbors(doc_id))
        if not doc_refs:
            return {'max_coupling_strength': 0.0, 'avg_coupling_relevant': 0.0, 'coupling_above_threshold': 0}
        
        couplings = []
        for rel_doc in self.relevant_documents:
            if rel_doc == doc_id:
                continue
            rel_refs = set(self.G.neighbors(rel_doc))
            shared_refs = doc_refs.intersection(rel_refs)
            union_refs = doc_refs.union(rel_refs)
            if union_refs:
                jaccard = len(shared_refs) / len(union_refs)
                couplings.append(jaccard)
        
        if not couplings:
            return {'max_coupling_strength': 0.0, 'avg_coupling_relevant': 0.0, 'coupling_above_threshold': 0}
        
        return {
            'max_coupling_strength': max(couplings),
            'avg_coupling_relevant': np.mean(couplings),
            'coupling_above_threshold': sum(1 for c in couplings if c > 0.1)
        }
    
    def _calc_neighborhood_features(self, doc_id: str) -> Dict[str, float]:
        """Calculate citation neighborhood enrichment features."""
        hop1_neighbors = set(self.G.neighbors(doc_id))
        hop2_neighbors = set()
        for neighbor in hop1_neighbors:
            hop2_neighbors.update(self.G.neighbors(neighbor))
        hop2_neighbors -= {doc_id}
        hop2_neighbors -= hop1_neighbors
        
        hop1_rel = sum(1 for n in hop1_neighbors if n in self.relevant_documents)
        hop2_rel = sum(1 for n in hop2_neighbors if n in self.relevant_documents)
        
        return {
            'neighborhood_size_1hop': len(hop1_neighbors),
            'neighborhood_size_2hop': len(hop2_neighbors),
            'neighborhood_enrichment_1hop': hop1_rel / max(len(hop1_neighbors), 1),
            'neighborhood_enrichment_2hop': hop2_rel / max(len(hop2_neighbors), 1)
        }
    
    def _get_zero_features(self, doc_id: str) -> Dict[str, float]:
        """Return zero features for documents not in citation network."""
        return {
            'openalex_id': doc_id, 'total_citations': 0, 'relevant_connections': 0,
            'relevant_connections_ratio': 0.0, 'max_coupling_strength': 0.0,
            'avg_coupling_relevant': 0.0, 'coupling_above_threshold': 0,
            'neighborhood_size_1hop': 0, 'neighborhood_size_2hop': 0,
            'neighborhood_enrichment_1hop': 0.0, 'neighborhood_enrichment_2hop': 0.0
        }
    
    def _calculate_baseline_stats(self) -> Dict[str, float]:
        """Calculate baseline citation statistics from relevant documents."""
        if not self.relevant_documents:
            return {}
        
        rel_features = self.get_citation_features(list(self.relevant_documents))
        if rel_features.empty:
            return {}
        
        # Use pandas describe for efficient statistics calculation
        stats = rel_features[['total_citations', 'relevant_connections_ratio', 
                             'max_coupling_strength', 'neighborhood_enrichment_1hop']].describe()
        
        baseline = {}
        for col in stats.columns:
            baseline[f'mean_{col}'] = stats.loc['mean', col]
            baseline[f'std_{col}'] = max(stats.loc['std', col], 1.0)  # Avoid division by zero
        
        # Calculate distance statistics for adaptive distance scoring
        distance_stats = self._calculate_distance_baselines()
        baseline.update(distance_stats)
        
        return baseline
    
    def _calculate_distance_baselines(self) -> Dict[str, float]:
        """Calculate distance statistics from relevant documents for adaptive scoring."""
        if not self.relevant_documents:
            return {}
        
        # Calculate pairwise distances between relevant documents
        relevant_list = list(self.relevant_documents)
        distances = []
        
        for i, doc1 in enumerate(relevant_list):
            for doc2 in relevant_list[i+1:]:
                if doc1 in self.G.nodes and doc2 in self.G.nodes:
                    try:
                        dist = nx.shortest_path_length(self.G, doc1, doc2)
                        distances.append(dist)
                    except nx.NetworkXNoPath:
                        # If no path exists, consider this as max observable distance + 1
                        distances.append(10)  # High but finite value
        
        if not distances:
            # Fallback values if no distances can be calculated
            return {
                'mean_distance': 3.0,
                'std_distance': 1.5,
                'median_distance': 3.0,
                'p25_distance': 2.0,
                'p75_distance': 4.0,
                'max_distance': 6.0
            }
        
        distances = np.array(distances)
        return {
            'mean_distance': float(np.mean(distances)),
            'std_distance': max(float(np.std(distances)), 0.5),  # Avoid division by zero
            'median_distance': float(np.median(distances)),
            'p25_distance': float(np.percentile(distances, 25)),
            'p75_distance': float(np.percentile(distances, 75)),
            'max_distance': float(np.max(distances))
        }
    
    def _calc_distance_to_relevant(self, doc_id: str) -> float:
        """Calculate shortest path distance to nearest relevant paper."""
        if doc_id not in self.G.nodes or not self.relevant_documents:
            return float('inf')
        
        min_dist = float('inf')
        for rel_doc in self.relevant_documents:
            if rel_doc in self.G.nodes:
                try:
                    dist = nx.shortest_path_length(self.G, doc_id, rel_doc)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    continue
        return min_dist
    
    def _calc_isolation_deviation(self, features: pd.Series, baseline: Dict[str, float]) -> float:
        """Calculate how isolated this document is from the relevant paper ecosystem."""
        doc_id = features['openalex_id']
        dist = self._calc_distance_to_relevant(doc_id)
        
        # Distance to isolation score mapping
        dist_scores = {float('inf'): 1.0, 4: 0.95, 3: 0.8, 2: 0.65, 1: 0.2, 0: 0.0}
        dist_score = dist_scores.get(dist, dist_scores.get(min(k for k in dist_scores.keys() if k >= dist), 0.95))
        
        # Citation volume penalty
        cit_count = features['total_citations']
        penalties = [(0, 0.4), (2, 0.6), (5, 0.8)]
        penalty = next((p for threshold, p in penalties if cit_count <= threshold), 1.0)
        dist_score *= penalty
        
        # Citation volume bonus for distance=2 papers
        if dist != float('inf') and cit_count > 0 and dist == 2:
            cit_dev = (cit_count - baseline['mean_total_citations']) / baseline['std_total_citations']
            if cit_dev >= 0.5:
                bonus = min(0.20, cit_dev * 0.15)
            elif cit_dev >= 0:
                bonus = min(0.10, cit_dev * 0.20)
            elif cit_dev >= -0.5:
                bonus = min(0.05, abs(cit_dev) * 0.05)
            else:
                bonus = 0.0
            dist_score = min(1.0, dist_score + bonus)
        
        # Direct isolation score
        if features['relevant_connections'] == 0:
            # Base isolation with quality factors
            cit_factor = min(1.0, features['total_citations'] / 20.0)
            neigh_factor = 1.0 - features['neighborhood_enrichment_1hop']
            hop2_factor = 1.0 - min(1.0, features['neighborhood_enrichment_2hop'] * 10.0)
            
            # Citation quality bonus
            cit_bonus = 0.0
            if cit_count > 0:
                cit_dev = (cit_count - baseline['mean_total_citations']) / baseline['std_total_citations']
                if cit_dev >= 0:
                    cit_bonus = min(0.3, cit_dev * 0.15)
                elif cit_count >= baseline['mean_total_citations'] * 0.25:
                    cit_bonus = min(0.2, abs(cit_dev) * 0.1)
            
            # 2-hop isolation bonus
            hop2_enrich = features['neighborhood_enrichment_2hop']
            hop2_bonus = 0.15 if hop2_enrich <= 0.01 else (0.10 if hop2_enrich <= 0.03 else (0.05 if hop2_enrich <= 0.05 else 0.0))
            
            direct_iso = min(0.95, 0.5 + 0.3 * (0.4 * cit_factor + 0.4 * neigh_factor + 0.2 * hop2_factor) + cit_bonus + hop2_bonus)
        else:
            # Partial isolation
            conn_dev = (baseline['mean_relevant_connections_ratio'] - features['relevant_connections_ratio']) / baseline['std_relevant_connections_ratio']
            direct_iso = max(0, min(0.6, conn_dev / 1.5))
        
        # Neighborhood isolation
        neigh_dev = (baseline['mean_neighborhood_enrichment_1hop'] - features['neighborhood_enrichment_1hop']) / baseline['std_neighborhood_enrichment_1hop']
        neigh_iso = max(0, min(1.0, neigh_dev / 1.5))
        
        # Second-hop isolation
        if features['neighborhood_enrichment_2hop'] == 0:
            hop2_iso = 0.6
        elif features['neighborhood_enrichment_2hop'] < 0.02:
            hop2_iso = 0.4
        else:
            hop2_iso = max(0, min(0.3, (0.05 - features['neighborhood_enrichment_2hop']) / 0.05))
        
        # Combine with weights
        iso_score = 0.5 * dist_score + 0.3 * direct_iso + 0.1 * neigh_iso + 0.1 * hop2_iso
        return min(1.0, iso_score)
    
    def _calc_coupling_deviation(self, features: pd.Series, baseline: Dict[str, float]) -> float:
        """Calculate how different coupling patterns are from baseline."""
        coup_dev = (baseline['mean_max_coupling_strength'] - features['max_coupling_strength']) / baseline['std_max_coupling_strength']
        return max(0, min(1.0, coup_dev / 2.0))
    
    def _calc_neighborhood_deviation(self, features: pd.Series, baseline: Dict[str, float]) -> float:
        """Calculate how different neighborhood patterns are from baseline."""
        neigh_dev = (baseline['mean_neighborhood_enrichment_1hop'] - features['neighborhood_enrichment_1hop']) / baseline['std_neighborhood_enrichment_1hop']
        return max(0, min(1.0, neigh_dev / 2.0))


def main():
    """Test citation network model standalone and show outlier ranking."""
    print("=== CITATION NETWORK MODEL STANDALONE TEST ===")
    
    # Load and prepare data
    sim_df = pd.read_csv('../data/simulation.csv')
    train_data = sim_df.copy()
    train_data['label_included'] = train_data['asreview_ranking'].apply(lambda x: 1 if x <= 25 else 0)
    
    print(f"Training with {train_data['label_included'].sum()} relevant documents (ranks 1-25)")
    
    # Get outlier info
    outlier_row = sim_df[sim_df['record_id'] == 497]
    outlier_id = outlier_row.iloc[0]['openalex_id']
    outlier_rank = outlier_row.iloc[0]['asreview_ranking']
    print(f"Target outlier: record_id=497, ASReview rank={outlier_rank}, ID: {outlier_id}")
    
    # Fit model and test
    model = CitationNetworkModel()
    model.fit(train_data)
    
    # Test candidates
    irrelevant_docs = sim_df[sim_df['label_included'] == 0]['openalex_id'].tolist()
    test_candidates = [outlier_id] + irrelevant_docs
    
    # Get scores and features
    scores = model.predict_relevance_scores(test_candidates)
    outlier_score = scores.get(outlier_id, 0.0)
    outlier_features = model.get_citation_features([outlier_id])
    
    # Debug info
    if not outlier_features.empty:
        f_dict = outlier_features.iloc[0].to_dict()
        print(f"\nOutlier citation features:")
        for k, v in f_dict.items():
            if k != 'openalex_id':
                print(f"  {k}: {v}")
    
    # Show baseline stats
    print(f"\n=== BASELINE STATISTICS ===")
    for k, v in model.baseline_stats.items():
        print(f"  {k}: {v:.4f}")
    
    # Score breakdown
    if not outlier_features.empty:
        f_row = outlier_features.iloc[0]
        iso = model._calc_isolation_deviation(f_row, model.baseline_stats)
        coup = model._calc_coupling_deviation(f_row, model.baseline_stats)
        neigh = model._calc_neighborhood_deviation(f_row, model.baseline_stats)
        
        print(f"\n=== OUTLIER SCORE BREAKDOWN ===")
        print(f"Isolation: {iso:.4f} (50%), Coupling: {coup:.4f} (30%), Neighborhood: {neigh:.4f} (20%)")
        print(f"Calculated: {0.5*iso + 0.3*coup + 0.2*neigh:.4f}, Actual: {outlier_score:.4f}")
        
        # Network distance
        dist = model._calc_distance_to_relevant(outlier_id)
        print(f"Network distance to relevant papers: {dist}")
    
    # Ranking results
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    outlier_pos = next((i for i, (doc_id, _) in enumerate(sorted_cands, 1) if doc_id == outlier_id), None)
    
    print(f"\n=== RESULTS ===")
    print(f"Outlier score: {outlier_score:.4f}, Position: {outlier_pos}/{len(test_candidates)}")
    percentile = ((len(test_candidates) - outlier_pos) / len(test_candidates)) * 100
    print(f"Percentile: {percentile:.1f}th")
    
    # Top 10 scores
    print(f"\nTop 10 scores:")
    for i, (doc_id, score) in enumerate(sorted_cands[:10], 1):
        marker = " *** OUTLIER ***" if doc_id == outlier_id else ""
        print(f"  {i:2d}. Score: {score:.4f}{marker}")
    
    # Performance assessment
    assessments = [(10, "🟢 EXCELLENT"), (50, "🟡 GOOD"), (100, "🟠 FAIR")]
    assessment = next((msg for threshold, msg in assessments if outlier_pos <= threshold), "🔴 POOR")
    print(f"\nPerformance: {assessment}")
    
    # Score statistics
    all_scores = list(scores.values())
    print(f"\nScore stats: Mean={np.mean(all_scores):.4f}, Std={np.std(all_scores):.4f}, "
          f"Min={np.min(all_scores):.4f}, Max={np.max(all_scores):.4f}")
    
    # Warnings
    high_scores = sum(1 for s in all_scores if s > 0.7)
    if high_scores > len(all_scores) * 0.1:
        print(f"⚠️  WARNING: {high_scores} documents ({high_scores/len(all_scores)*100:.1f}%) have scores > 0.7")
    
    same_score = sum(1 for s in all_scores if abs(s - outlier_score) < 0.001)
    if same_score > 10:
        print(f"⚠️  WARNING: {same_score} documents have nearly identical scores to outlier")
    
    return {
        'outlier_position': outlier_pos, 'outlier_score': outlier_score,
        'total_candidates': len(test_candidates), 'percentile': percentile
    }


if __name__ == "__main__":
    main() 