import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os
from synergy_dataset import Dataset

# Create directories if they don't exist
os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)

def load_data():
    """Load both the simulation dataset and Appenzeller dataset with references"""
    # Load simulation results
    simulation_df = pd.read_csv('../data/simulation.csv')
    print(f"Loaded simulation dataset with {len(simulation_df)} records")
    
    # Load Appenzeller dataset with references
    appenzeller = Dataset("Appenzeller-Herzog_2019")
    appenzeller_data = appenzeller.to_dict(["id", "title", "abstract", "referenced_works"])
    print(f"Loaded Appenzeller dataset with {len(appenzeller_data)} records")
    
    return simulation_df, appenzeller_data

def build_citation_network(appenzeller_data):
    """Build a citation network from the Appenzeller dataset"""
    # Create nodes with metadata
    nodes = [(k, {"title": v.get("title", ""), 
                 "abstract": v.get("abstract", ""),
                 "label_included": v.get("label_included", 0)}) 
             for k, v in appenzeller_data.items()]
    
    # Create edges (citations)
    edges = [(k, r) for k, v in appenzeller_data.items() for r in v.get("referenced_works", [])]
    
    # Build the graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Remove nodes that aren't part of our original dataset
    G.remove_nodes_from(set(G.nodes) - set([n[0] for n in nodes]))
    
    print(f"Created citation network with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def map_simulation_to_network(simulation_df, appenzeller_data, G):
    """Map simulation results to the citation network"""
    record_to_openalex = dict(zip(simulation_df['record_id'], simulation_df['openalex_id']))
    
    for record_id, openalex_id in record_to_openalex.items():
        if openalex_id in G.nodes:
            G.nodes[openalex_id]['record_id'] = record_id
            sim_row = simulation_df[simulation_df['record_id'] == record_id].iloc[0]
            G.nodes[openalex_id]['asreview_ranking'] = sim_row['asreview_ranking']
            G.nodes[openalex_id]['label_included'] = sim_row['label_included']
            G.nodes[openalex_id]['asreview_prior'] = sim_row['asreview_prior']
            G.nodes[openalex_id]['is_outlier'] = (record_id == 497)
    
    mapped_nodes = sum(1 for n in G.nodes if G.nodes[n].get('record_id') is not None)
    print(f"Mapped {mapped_nodes} simulation records to the citation network")
    
    return G

def explore_citation_connectivity(G, simulation_df):
    """Explore how well citation connectivity predicts relevance"""
    print("\n=== Citation Connectivity Analysis ===")
    
    # Get all documents that have been through simulation
    sim_nodes = [n for n in G.nodes if G.nodes[n].get('record_id') is not None]
    
    # Calculate citation connectivity metrics for each document
    connectivity_data = []
    
    for node in sim_nodes:
        record_id = G.nodes[node]['record_id']
        label = G.nodes[node]['label_included']
        ranking = G.nodes[node]['asreview_ranking']
        is_outlier = G.nodes[node].get('is_outlier', False)
        
        # Get neighbors (direct citations)
        neighbors = list(G.neighbors(node))
        
        # Count connections to relevant documents
        relevant_connections = sum(1 for n in neighbors if G.nodes[n].get('label_included') == 1)
        
        # Count total connections within dataset
        internal_connections = sum(1 for n in neighbors if G.nodes[n].get('record_id') is not None)
        
        connectivity_data.append({
            'record_id': record_id,
            'label_included': label,
            'asreview_ranking': ranking,
            'is_outlier': is_outlier,
            'total_citations': len(neighbors),
            'relevant_connections': relevant_connections,
            'internal_connections': internal_connections,
            'connection_ratio': relevant_connections / max(internal_connections, 1)
        })
    
    connectivity_df = pd.DataFrame(connectivity_data)
    
    # Analyze patterns
    print("Citation connectivity by document type:")
    print("\nRelevant documents (excluding outlier):")
    relevant_no_outlier = connectivity_df[(connectivity_df['label_included'] == 1) & (~connectivity_df['is_outlier'])]
    print(f"Average total citations: {relevant_no_outlier['total_citations'].mean():.2f}")
    print(f"Average relevant connections: {relevant_no_outlier['relevant_connections'].mean():.2f}")
    print(f"Average connection ratio: {relevant_no_outlier['connection_ratio'].mean():.4f}")
    
    print("\nNon-relevant documents:")
    non_relevant = connectivity_df[connectivity_df['label_included'] == 0]
    print(f"Average total citations: {non_relevant['total_citations'].mean():.2f}")
    print(f"Average relevant connections: {non_relevant['relevant_connections'].mean():.2f}")
    print(f"Average connection ratio: {non_relevant['connection_ratio'].mean():.4f}")
    
    print("\nOutlier document:")
    outlier_data = connectivity_df[connectivity_df['is_outlier']]
    if not outlier_data.empty:
        outlier = outlier_data.iloc[0]
        print(f"Total citations: {outlier['total_citations']}")
        print(f"Relevant connections: {outlier['relevant_connections']}")
        print(f"Connection ratio: {outlier['connection_ratio']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Total citations by relevance
    axes[0].boxplot([
        connectivity_df[connectivity_df['label_included'] == 0]['total_citations'],
        connectivity_df[(connectivity_df['label_included'] == 1) & (~connectivity_df['is_outlier'])]['total_citations'],
        connectivity_df[connectivity_df['is_outlier']]['total_citations']
    ], labels=['Non-relevant', 'Relevant', 'Outlier'])
    axes[0].set_title('Total Citations by Document Type')
    axes[0].set_ylabel('Number of Citations')
    
    # Plot 2: Relevant connections by relevance
    axes[1].boxplot([
        connectivity_df[connectivity_df['label_included'] == 0]['relevant_connections'],
        connectivity_df[(connectivity_df['label_included'] == 1) & (~connectivity_df['is_outlier'])]['relevant_connections'],
        connectivity_df[connectivity_df['is_outlier']]['relevant_connections']
    ], labels=['Non-relevant', 'Relevant', 'Outlier'])
    axes[1].set_title('Connections to Relevant Documents')
    axes[1].set_ylabel('Number of Relevant Connections')
    
    # Plot 3: Connection ratio
    axes[2].boxplot([
        connectivity_df[connectivity_df['label_included'] == 0]['connection_ratio'],
        connectivity_df[(connectivity_df['label_included'] == 1) & (~connectivity_df['is_outlier'])]['connection_ratio'],
        connectivity_df[connectivity_df['is_outlier']]['connection_ratio']
    ], labels=['Non-relevant', 'Relevant', 'Outlier'])
    axes[2].set_title('Ratio of Relevant Connections')
    axes[2].set_ylabel('Relevant Connections / Total Internal Connections')
    
    plt.tight_layout()
    plt.savefig('output/figures/citation_connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return connectivity_df

def analyze_bibliographic_coupling_patterns(G, simulation_df):
    """Deep dive into bibliographic coupling patterns"""
    print("\n=== Bibliographic Coupling Pattern Analysis ===")
    
    # Get outlier and relevant documents
    outlier_record = simulation_df[simulation_df['record_id'] == 497]
    if outlier_record.empty:
        return None
    
    outlier_id = outlier_record.iloc[0]['openalex_id']
    if outlier_id not in G.nodes:
        return None
    
    relevant_docs = simulation_df[simulation_df['label_included'] == 1]
    relevant_ids = [simulation_df[simulation_df['record_id'] == rid]['openalex_id'].iloc[0] 
                   for rid in relevant_docs['record_id'] if 
                   len(simulation_df[simulation_df['record_id'] == rid]) > 0]
    relevant_ids = [rid for rid in relevant_ids if rid in G.nodes]
    
    # Calculate comprehensive bibliographic coupling
    coupling_data = []
    outlier_refs = set(G.neighbors(outlier_id))
    
    # Analyze coupling with all documents in the dataset
    sim_nodes = [n for n in G.nodes if G.nodes[n].get('record_id') is not None and n != outlier_id]
    
    for node in sim_nodes:
        node_refs = set(G.neighbors(node))
        shared_refs = outlier_refs.intersection(node_refs)
        union_refs = outlier_refs.union(node_refs)
        
        jaccard = len(shared_refs) / len(union_refs) if union_refs else 0
        overlap_count = len(shared_refs)
        
        # Handle NaN values in label_included - treat as irrelevant (0)
        label_included = G.nodes[node].get('label_included', 0)
        if pd.isna(label_included):
            label_included = 0
        
        coupling_data.append({
            'openalex_id': node,
            'record_id': G.nodes[node].get('record_id'),
            'label_included': label_included,
            'asreview_ranking': G.nodes[node].get('asreview_ranking'),
            'shared_refs_count': overlap_count,
            'jaccard_similarity': jaccard,
            'outlier_refs_total': len(outlier_refs),
            'node_refs_total': len(node_refs)
        })
    
    coupling_df = pd.DataFrame(coupling_data)
    coupling_df = coupling_df.sort_values('jaccard_similarity', ascending=False)
    
    # Analyze top coupled documents
    print("\nTop 15 documents by bibliographic coupling with outlier:")
    top_coupled = coupling_df.head(15)
    print(top_coupled[['record_id', 'label_included', 'asreview_ranking', 
                      'shared_refs_count', 'jaccard_similarity']].to_string(index=False))
    
    # Statistical analysis
    relevant_coupling = coupling_df[coupling_df['label_included'] == 1]
    non_relevant_coupling = coupling_df[coupling_df['label_included'] == 0]
    
    print(f"\nStatistical Analysis:")
    print(f"Mean Jaccard similarity with relevant docs: {relevant_coupling['jaccard_similarity'].mean():.4f}")
    print(f"Mean Jaccard similarity with non-relevant docs: {non_relevant_coupling['jaccard_similarity'].mean():.4f}")
    print(f"Relevant docs in top 10 coupled: {sum(coupling_df.head(10)['label_included'])}")
    print(f"Relevant docs in top 20 coupled: {sum(coupling_df.head(20)['label_included'])}")
    print(f"Relevant docs in top 50 coupled: {sum(coupling_df.head(50)['label_included'])}")
    
    # Visualization: Coupling strength vs ranking
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Coupling vs Ranking
    plt.subplot(2, 2, 1)
    relevant_mask = coupling_df['label_included'] == 1
    plt.scatter(coupling_df[~relevant_mask]['asreview_ranking'], 
                coupling_df[~relevant_mask]['jaccard_similarity'], 
                alpha=0.3, s=30, c='gray', label='Non-relevant')
    plt.scatter(coupling_df[relevant_mask]['asreview_ranking'], 
                coupling_df[relevant_mask]['jaccard_similarity'], 
                alpha=0.8, s=100, c='blue', label='Relevant')
    plt.xlabel('ASReview Ranking')
    plt.ylabel('Jaccard Similarity')
    plt.title('Bibliographic Coupling vs Ranking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribution of coupling strengths
    plt.subplot(2, 2, 2)
    plt.hist(coupling_df[coupling_df['label_included'] == 0]['jaccard_similarity'], 
             bins=30, alpha=0.7, label='Non-relevant', density=True)
    plt.hist(coupling_df[coupling_df['label_included'] == 1]['jaccard_similarity'], 
             bins=30, alpha=0.7, label='Relevant', density=True)
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of Coupling Strengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Shared reference counts
    plt.subplot(2, 2, 3)
    plt.scatter(coupling_df[~relevant_mask]['asreview_ranking'], 
                coupling_df[~relevant_mask]['shared_refs_count'], 
                alpha=0.3, s=30, c='gray', label='Non-relevant')
    plt.scatter(coupling_df[relevant_mask]['asreview_ranking'], 
                coupling_df[relevant_mask]['shared_refs_count'], 
                alpha=0.8, s=100, c='blue', label='Relevant')
    plt.xlabel('ASReview Ranking')
    plt.ylabel('Shared References Count')
    plt.title('Shared References vs Ranking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: ROC-like analysis
    plt.subplot(2, 2, 4)
    thresholds = np.linspace(0, coupling_df['jaccard_similarity'].max(), 100)
    precisions = []
    recalls = []
    
    total_relevant = sum(coupling_df['label_included'])
    for threshold in thresholds:
        selected = coupling_df[coupling_df['jaccard_similarity'] >= threshold]
        if len(selected) > 0:
            precision = sum(selected['label_included']) / len(selected)
            recall = sum(selected['label_included']) / total_relevant
            precisions.append(precision)
            recalls.append(recall)
        else:
            precisions.append(0)
            recalls.append(0)
    
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall: Coupling-based Selection')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/bibliographic_coupling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return coupling_df

def analyze_citation_neighborhoods(G, simulation_df):
    """Analyze the citation neighborhoods of different document types"""
    print("\n=== Citation Neighborhood Analysis ===")
    
    # Get relevant documents and outlier
    relevant_docs = simulation_df[simulation_df['label_included'] == 1]
    outlier_record = simulation_df[simulation_df['record_id'] == 497]
    
    if outlier_record.empty:
        return
    
    outlier_id = outlier_record.iloc[0]['openalex_id']
    
    # Analyze 2-hop neighborhoods
    def get_neighborhood_stats(node_id, max_hops=2):
        """Get statistics about a node's citation neighborhood"""
        if node_id not in G.nodes:
            return None
        
        # 1-hop neighbors
        hop1_neighbors = set(G.neighbors(node_id))
        
        # 2-hop neighbors (excluding the original node)
        hop2_neighbors = set()
        for neighbor in hop1_neighbors:
            hop2_neighbors.update(G.neighbors(neighbor))
        hop2_neighbors -= {node_id}
        hop2_neighbors -= hop1_neighbors
        
        # Count relevant documents in neighborhoods
        hop1_relevant = sum(1 for n in hop1_neighbors if G.nodes[n].get('label_included') == 1)
        hop2_relevant = sum(1 for n in hop2_neighbors if G.nodes[n].get('label_included') == 1)
        
        return {
            'hop1_size': len(hop1_neighbors),
            'hop2_size': len(hop2_neighbors),
            'hop1_relevant': hop1_relevant,
            'hop2_relevant': hop2_relevant,
            'hop1_relevant_ratio': hop1_relevant / max(len(hop1_neighbors), 1),
            'hop2_relevant_ratio': hop2_relevant / max(len(hop2_neighbors), 1)
        }
    
    # Analyze neighborhoods for different document types
    neighborhood_data = []
    
    # Sample some documents for analysis (to keep it manageable)
    relevant_sample = relevant_docs[relevant_docs['record_id'] != 497].sample(min(20, len(relevant_docs)-1))
    non_relevant_sample = simulation_df[simulation_df['label_included'] == 0].sample(min(50, len(simulation_df[simulation_df['label_included'] == 0])))
    
    # Analyze relevant documents
    for _, row in relevant_sample.iterrows():
        openalex_id = row['openalex_id']
        if openalex_id in G.nodes:
            stats = get_neighborhood_stats(openalex_id)
            if stats:
                stats.update({
                    'record_id': row['record_id'],
                    'doc_type': 'relevant',
                    'asreview_ranking': row['asreview_ranking']
                })
                neighborhood_data.append(stats)
    
    # Analyze non-relevant documents
    for _, row in non_relevant_sample.iterrows():
        openalex_id = row['openalex_id']
        if openalex_id in G.nodes:
            stats = get_neighborhood_stats(openalex_id)
            if stats:
                stats.update({
                    'record_id': row['record_id'],
                    'doc_type': 'non_relevant',
                    'asreview_ranking': row['asreview_ranking']
                })
                neighborhood_data.append(stats)
    
    # Analyze outlier
    outlier_stats = get_neighborhood_stats(outlier_id)
    if outlier_stats:
        outlier_stats.update({
            'record_id': 497,
            'doc_type': 'outlier',
            'asreview_ranking': outlier_record.iloc[0]['asreview_ranking']
        })
        neighborhood_data.append(outlier_stats)
    
    neighborhood_df = pd.DataFrame(neighborhood_data)
    
    # Print summary statistics
    print("Citation neighborhood statistics by document type:")
    for doc_type in ['relevant', 'non_relevant', 'outlier']:
        subset = neighborhood_df[neighborhood_df['doc_type'] == doc_type]
        if not subset.empty:
            print(f"\n{doc_type.upper()}:")
            print(f"  1-hop avg size: {subset['hop1_size'].mean():.2f}")
            print(f"  2-hop avg size: {subset['hop2_size'].mean():.2f}")
            print(f"  1-hop avg relevant ratio: {subset['hop1_relevant_ratio'].mean():.4f}")
            print(f"  2-hop avg relevant ratio: {subset['hop2_relevant_ratio'].mean():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: 1-hop relevant ratios
    relevant_data = neighborhood_df[neighborhood_df['doc_type'] == 'relevant']['hop1_relevant_ratio']
    non_relevant_data = neighborhood_df[neighborhood_df['doc_type'] == 'non_relevant']['hop1_relevant_ratio']
    outlier_data = neighborhood_df[neighborhood_df['doc_type'] == 'outlier']['hop1_relevant_ratio']
    
    axes[0,0].boxplot([non_relevant_data, relevant_data, outlier_data], 
                      labels=['Non-relevant', 'Relevant', 'Outlier'])
    axes[0,0].set_title('1-hop Relevant Document Ratio')
    axes[0,0].set_ylabel('Ratio of Relevant Neighbors')
    
    # Plot 2: 2-hop relevant ratios
    relevant_data = neighborhood_df[neighborhood_df['doc_type'] == 'relevant']['hop2_relevant_ratio']
    non_relevant_data = neighborhood_df[neighborhood_df['doc_type'] == 'non_relevant']['hop2_relevant_ratio']
    outlier_data = neighborhood_df[neighborhood_df['doc_type'] == 'outlier']['hop2_relevant_ratio']
    
    axes[0,1].boxplot([non_relevant_data, relevant_data, outlier_data], 
                      labels=['Non-relevant', 'Relevant', 'Outlier'])
    axes[0,1].set_title('2-hop Relevant Document Ratio')
    axes[0,1].set_ylabel('Ratio of Relevant 2-hop Neighbors')
    
    # Plot 3: Neighborhood sizes
    axes[1,0].scatter(neighborhood_df[neighborhood_df['doc_type'] == 'non_relevant']['hop1_size'],
                      neighborhood_df[neighborhood_df['doc_type'] == 'non_relevant']['hop2_size'],
                      alpha=0.5, c='gray', label='Non-relevant', s=30)
    axes[1,0].scatter(neighborhood_df[neighborhood_df['doc_type'] == 'relevant']['hop1_size'],
                      neighborhood_df[neighborhood_df['doc_type'] == 'relevant']['hop2_size'],
                      alpha=0.8, c='blue', label='Relevant', s=60)
    axes[1,0].scatter(neighborhood_df[neighborhood_df['doc_type'] == 'outlier']['hop1_size'],
                      neighborhood_df[neighborhood_df['doc_type'] == 'outlier']['hop2_size'],
                      alpha=1.0, c='red', label='Outlier', s=100, marker='*')
    axes[1,0].set_xlabel('1-hop Neighborhood Size')
    axes[1,0].set_ylabel('2-hop Neighborhood Size')
    axes[1,0].set_title('Citation Neighborhood Sizes')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Ranking vs 2-hop relevant ratio
    relevant_mask = neighborhood_df['doc_type'] == 'relevant'
    non_relevant_mask = neighborhood_df['doc_type'] == 'non_relevant'
    outlier_mask = neighborhood_df['doc_type'] == 'outlier'
    
    axes[1,1].scatter(neighborhood_df[non_relevant_mask]['asreview_ranking'],
                      neighborhood_df[non_relevant_mask]['hop2_relevant_ratio'],
                      alpha=0.5, c='gray', label='Non-relevant', s=30)
    axes[1,1].scatter(neighborhood_df[relevant_mask]['asreview_ranking'],
                      neighborhood_df[relevant_mask]['hop2_relevant_ratio'],
                      alpha=0.8, c='blue', label='Relevant', s=60)
    axes[1,1].scatter(neighborhood_df[outlier_mask]['asreview_ranking'],
                      neighborhood_df[outlier_mask]['hop2_relevant_ratio'],
                      alpha=1.0, c='red', label='Outlier', s=100, marker='*')
    axes[1,1].set_xlabel('ASReview Ranking')
    axes[1,1].set_ylabel('2-hop Relevant Ratio')
    axes[1,1].set_title('Ranking vs 2-hop Relevant Neighborhood')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/citation_neighborhoods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return neighborhood_df

def create_citation_summary_report(connectivity_df, coupling_df, neighborhood_df):
    """Create a summary report of citation analysis findings"""
    print("\n" + "="*60)
    print("CITATION ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    if coupling_df is not None:
        print("\n1. BIBLIOGRAPHIC COUPLING EFFECTIVENESS:")
        
        # How many relevant documents are in top-k by coupling?
        total_relevant = int(sum(coupling_df['label_included']))
        
        for k in [10, 20, 50, 100]:
            top_k = coupling_df.head(k)
            relevant_in_top_k = int(sum(top_k['label_included']))
            percentage = (relevant_in_top_k / total_relevant) * 100 if total_relevant > 0 else 0
            precision = (relevant_in_top_k / k) * 100
            print(f"   Top {k:3d}: {relevant_in_top_k:2d}/{total_relevant:2d} relevant docs ({percentage:5.1f}%), Precision: {precision:5.1f}%")
        
        # Compare with random baseline
        dataset_precision = total_relevant / len(coupling_df) * 100
        print(f"   Random baseline precision: {dataset_precision:.1f}%")
        
        # Outlier's position in coupling ranking
        outlier_position = None
        for i, (_, row) in enumerate(coupling_df.iterrows(), 1):
            if row['record_id'] == 497:
                outlier_position = i
                break
        
        if outlier_position:
            print(f"   Outlier position in coupling ranking: {outlier_position}/{len(coupling_df)}")
            percentile = (1 - outlier_position/len(coupling_df)) * 100
            print(f"   Outlier in top {percentile:.1f}% by bibliographic coupling")
    
    if neighborhood_df is not None:
        print("\n2. CITATION NEIGHBORHOOD PATTERNS:")
        
        # Compare neighborhood enrichment
        relevant_subset = neighborhood_df[neighborhood_df['doc_type'] == 'relevant']
        non_relevant_subset = neighborhood_df[neighborhood_df['doc_type'] == 'non_relevant']
        outlier_subset = neighborhood_df[neighborhood_df['doc_type'] == 'outlier']
        
        if not relevant_subset.empty and not non_relevant_subset.empty:
            rel_1hop = relevant_subset['hop1_relevant_ratio'].mean()
            nonrel_1hop = non_relevant_subset['hop1_relevant_ratio'].mean()
            rel_2hop = relevant_subset['hop2_relevant_ratio'].mean()
            nonrel_2hop = non_relevant_subset['hop2_relevant_ratio'].mean()
            
            print(f"   Avg 1-hop relevant ratio - Relevant: {rel_1hop:.4f}, Non-relevant: {nonrel_1hop:.4f}")
            print(f"   Avg 2-hop relevant ratio - Relevant: {rel_2hop:.4f}, Non-relevant: {nonrel_2hop:.4f}")
            
            if not outlier_subset.empty:
                outlier_1hop = outlier_subset['hop1_relevant_ratio'].iloc[0]
                outlier_2hop = outlier_subset['hop2_relevant_ratio'].iloc[0]
                print(f"   Outlier 1-hop relevant ratio: {outlier_1hop:.4f}")
                print(f"   Outlier 2-hop relevant ratio: {outlier_2hop:.4f}")
    
    print("\n3. OVERALL ASSESSMENT:")
    
    # Determine if citation methods show promise
    citation_promising = False
    reasons = []
    
    if coupling_df is not None and total_relevant > 0:
        # Check if coupling beats random baseline significantly
        top_20_precision = sum(coupling_df.head(20)['label_included']) / 20
        random_precision = sum(coupling_df['label_included']) / len(coupling_df)
        
        if top_20_precision > random_precision * 2:
            citation_promising = True
            reasons.append(f"Bibliographic coupling shows {top_20_precision/random_precision:.1f}x improvement over random")
        
        # Check outlier detectability
        if outlier_position and outlier_position <= len(coupling_df) * 0.2:  # Top 20%
            citation_promising = True
            reasons.append("Outlier is detectable in top 20% by bibliographic coupling")
    
    if citation_promising:
        print("   ✓ Citation-based methods show PROMISE for outlier detection")
        for reason in reasons:
            print(f"     - {reason}")
    else:
        print("   ✗ Citation-based methods show LIMITED promise")
        print("     - Consider focusing on other approaches")
    
    print("\n4. RECOMMENDATIONS:")
    if citation_promising:
        print("   → Implement bibliographic coupling in hybrid outlier detection model")
        print("   → Consider 2-hop citation neighborhood features")
        print("   → Combine with content-based features for best results")
    else:
        print("   → Citation methods may not be sufficient alone")
        print("   → Focus on other metadata or content-based approaches")
        print("   → Consider author networks or venue-based features")

def main():
    """Main execution function"""
    print("Starting comprehensive citation exploration...")
    
    # Load data and build network
    simulation_df, appenzeller_data = load_data()
    G = build_citation_network(appenzeller_data)
    G = map_simulation_to_network(simulation_df, appenzeller_data, G)
    
    # Run comprehensive citation analysis
    connectivity_df = explore_citation_connectivity(G, simulation_df)
    coupling_df = analyze_bibliographic_coupling_patterns(G, simulation_df)
    neighborhood_df = analyze_citation_neighborhoods(G, simulation_df)
    
    # Generate summary report
    create_citation_summary_report(connectivity_df, coupling_df, neighborhood_df)
    
    print(f"\nComprehensive citation analysis complete!")
    print(f"Results and visualizations saved to the 'output/figures' directory.")

if __name__ == "__main__":
    main() 