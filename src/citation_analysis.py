import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    # (these would be external references)
    G.remove_nodes_from(set(G.nodes) - set([n[0] for n in nodes]))
    
    print(f"Created citation network with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def map_simulation_to_network(simulation_df, appenzeller_data, G):
    """Map simulation results to the citation network"""
    # Create a mapping from record_id to openalex_id
    record_to_openalex = dict(zip(simulation_df['record_id'], simulation_df['openalex_id']))
    
    # Add simulation data as node attributes
    for record_id, openalex_id in record_to_openalex.items():
        if openalex_id in G.nodes:
            G.nodes[openalex_id]['record_id'] = record_id
            
            # Get simulation data
            sim_row = simulation_df[simulation_df['record_id'] == record_id].iloc[0]
            G.nodes[openalex_id]['asreview_ranking'] = sim_row['asreview_ranking']
            G.nodes[openalex_id]['label_included'] = sim_row['label_included']
            G.nodes[openalex_id]['asreview_prior'] = sim_row['asreview_prior']
            G.nodes[openalex_id]['is_outlier'] = (record_id == 497)
    
    # Count how many nodes we successfully mapped
    mapped_nodes = sum(1 for n in G.nodes if G.nodes[n].get('record_id') is not None)
    print(f"Mapped {mapped_nodes} simulation records to the citation network")
    
    return G

def analyze_outlier_citations(G, simulation_df):
    """Analyze citation patterns related to the outlier"""
    # Get the OpenAlex ID of our outlier
    outlier_record = simulation_df[simulation_df['record_id'] == 497]
    if outlier_record.empty:
        print("Outlier record (ID 497) not found in simulation data")
        return
    
    outlier_id = outlier_record.iloc[0]['openalex_id']
    if outlier_id not in G.nodes:
        print(f"Outlier OpenAlex ID {outlier_id} not found in citation network")
        return
    
    print(f"\n=== Citation Analysis for Outlier ===")
    print(f"Outlier OpenAlex ID: {outlier_id}")
    
    # Get direct neighbors (citations) of the outlier
    neighbors = list(G.neighbors(outlier_id))
    print(f"Outlier has {len(neighbors)} direct citation connections")
    
    # Find which of these neighbors are relevant documents
    relevant_neighbors = [n for n in neighbors if G.nodes[n].get('label_included') == 1]
    print(f"Number of connections to other relevant documents: {len(relevant_neighbors)}")
    
    # Get all relevant documents
    relevant_nodes = [n for n in G.nodes if G.nodes[n].get('label_included') == 1 and n != outlier_id]
    print(f"Total number of other relevant documents in network: {len(relevant_nodes)}")
    
    # Calculate what percentage of relevant documents are connected to the outlier
    if relevant_nodes:
        connection_percentage = len(relevant_neighbors) / len(relevant_nodes) * 100
        print(f"Percentage of relevant documents connected to outlier: {connection_percentage:.2f}%")
    
    # Analyze common citations between outlier and other relevant documents
    common_citations_count = {}
    for node in relevant_nodes:
        node_neighbors = set(G.neighbors(node))
        outlier_neighbors = set(G.neighbors(outlier_id))
        common = node_neighbors.intersection(outlier_neighbors)
        common_citations_count[node] = len(common)
    
    # Sort by number of common citations
    sorted_common = sorted(common_citations_count.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop relevant documents with most common citations with outlier:")
    for i, (node, count) in enumerate(sorted_common[:5], 1):
        node_record_id = G.nodes[node].get('record_id', 'Unknown')
        node_ranking = G.nodes[node].get('asreview_ranking', 'Unknown')
        print(f"{i}. Record ID: {node_record_id}, Ranking: {node_ranking}, Common citations: {count}")

def calculate_network_metrics(G):
    """Calculate network metrics that might help identify outliers"""
    print("\n=== Network Metrics ===")
    
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Calculate betweenness centrality (handle disconnected graphs)
    print("Computing betweenness centrality...")
    try:
        # First try regular betweenness centrality (works on disconnected graphs)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
    except Exception as e:
        print(f"Error calculating betweenness centrality: {e}")
        # Fallback to a default empty dictionary
        betweenness_centrality = {node: 0.0 for node in G.nodes}
    
    # Calculate PageRank
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Add metrics to nodes
    for node in G.nodes:
        G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
        G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
        G.nodes[node]['pagerank'] = pagerank.get(node, 0)
    
    # Get metrics for relevant documents
    relevant_nodes = [(n, G.nodes[n]) for n in G.nodes 
                     if G.nodes[n].get('label_included') == 1 
                     and G.nodes[n].get('record_id') is not None]
    
    # Create a DataFrame for analysis
    metrics_df = pd.DataFrame([
        {
            'record_id': data.get('record_id'),
            'asreview_ranking': data.get('asreview_ranking'),
            'is_outlier': data.get('is_outlier', False),
            'degree_centrality': data.get('degree_centrality', 0),
            'betweenness_centrality': data.get('betweenness_centrality', 0),
            'pagerank': data.get('pagerank', 0),
            'neighbor_count': len(list(G.neighbors(node)))
        }
        for node, data in relevant_nodes
    ])
    
    # Sort by ASReview ranking
    metrics_df = metrics_df.sort_values('asreview_ranking')
    
    # Print metrics for relevant documents
    print("\nNetwork metrics for relevant documents:")
    print(metrics_df.to_string(index=False))
    
    # Highlight outlier metrics
    outlier_metrics = metrics_df[metrics_df['is_outlier'] == True]
    if not outlier_metrics.empty:
        print("\nOutlier metrics:")
        print(outlier_metrics.to_string(index=False))
    
    return metrics_df

def visualize_network_metrics(metrics_df):
    """Visualize network metrics to identify patterns"""
    # Ranking vs. Network Metrics scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Ranking vs Degree Centrality
    axes[0].scatter(
        metrics_df[metrics_df['is_outlier'] == False]['asreview_ranking'],
        metrics_df[metrics_df['is_outlier'] == False]['degree_centrality'],
        alpha=0.7, s=100, c='blue', label='Relevant Documents'
    )
    if any(metrics_df['is_outlier']):
        axes[0].scatter(
            metrics_df[metrics_df['is_outlier'] == True]['asreview_ranking'],
            metrics_df[metrics_df['is_outlier'] == True]['degree_centrality'],
            alpha=1.0, s=200, c='red', marker='*', label='Outlier'
        )
    axes[0].set_title('Ranking vs. Degree Centrality')
    axes[0].set_xlabel('ASReview Ranking')
    axes[0].set_ylabel('Degree Centrality')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Ranking vs Betweenness Centrality
    axes[1].scatter(
        metrics_df[metrics_df['is_outlier'] == False]['asreview_ranking'],
        metrics_df[metrics_df['is_outlier'] == False]['betweenness_centrality'],
        alpha=0.7, s=100, c='blue', label='Relevant Documents'
    )
    if any(metrics_df['is_outlier']):
        axes[1].scatter(
            metrics_df[metrics_df['is_outlier'] == True]['asreview_ranking'],
            metrics_df[metrics_df['is_outlier'] == True]['betweenness_centrality'],
            alpha=1.0, s=200, c='red', marker='*', label='Outlier'
        )
    axes[1].set_title('Ranking vs. Betweenness Centrality')
    axes[1].set_xlabel('ASReview Ranking')
    axes[1].set_ylabel('Betweenness Centrality')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Ranking vs PageRank
    axes[2].scatter(
        metrics_df[metrics_df['is_outlier'] == False]['asreview_ranking'],
        metrics_df[metrics_df['is_outlier'] == False]['pagerank'],
        alpha=0.7, s=100, c='blue', label='Relevant Documents'
    )
    if any(metrics_df['is_outlier']):
        axes[2].scatter(
            metrics_df[metrics_df['is_outlier'] == True]['asreview_ranking'],
            metrics_df[metrics_df['is_outlier'] == True]['pagerank'],
            alpha=1.0, s=200, c='red', marker='*', label='Outlier'
        )
    axes[2].set_title('Ranking vs. PageRank')
    axes[2].set_xlabel('ASReview Ranking')
    axes[2].set_ylabel('PageRank')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('output/figures/network_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_citation_network(G):
    """Create a visualization of the citation network focusing on the outlier"""
    # Create a subgraph with the outlier and relevant documents
    relevant_nodes = [n for n in G.nodes if G.nodes[n].get('label_included') == 1 and G.nodes[n].get('record_id') is not None]
    
    # Add direct neighbors of the outlier
    outlier_node = None
    for n in G.nodes:
        if G.nodes[n].get('is_outlier', False):
            outlier_node = n
            break
    
    if outlier_node:
        outlier_neighbors = list(G.neighbors(outlier_node))
        subgraph_nodes = set(relevant_nodes + outlier_neighbors)
        subgraph = G.subgraph(subgraph_nodes)
    else:
        subgraph = G.subgraph(relevant_nodes)
    
    # Create a visualization
    plt.figure(figsize=(14, 12))
    
    # Define node colors and sizes
    node_colors = []
    node_sizes = []
    
    for node in subgraph.nodes:
        if G.nodes[node].get('is_outlier', False):
            node_colors.append('red')
            node_sizes.append(800)
        elif G.nodes[node].get('asreview_prior', 0) == 1:
            node_colors.append('green')
            node_sizes.append(600)
        elif G.nodes[node].get('label_included', 0) == 1:
            node_colors.append('blue')
            node_sizes.append(400)
        else:
            node_colors.append('gray')
            node_sizes.append(200)
    
    # Use a layout that spaces nodes well
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Draw the graph
    nx.draw_networkx(
        subgraph, pos, 
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.7,
        with_labels=False,
        width=0.5,
        edge_color='lightgray'
    )
    
    # Add labels for important nodes
    labels = {}
    for node in subgraph.nodes:
        if (G.nodes[node].get('is_outlier', False) or 
            G.nodes[node].get('asreview_prior', 0) == 1 or
            G.nodes[node].get('label_included', 0) == 1):
            labels[node] = G.nodes[node].get('record_id', '')
    
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=10)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Outlier'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Prior Document'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Relevant Document'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label='Connected Document')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Citation Network: Outlier and Relevant Documents')
    plt.axis('off')
    plt.savefig('output/figures/citation_network.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_bibliographic_coupling(G):
    """Analyze bibliographic coupling (common references) between documents"""
    print("\n=== Bibliographic Coupling Analysis ===")
    
    # Get all nodes with simulation data
    sim_nodes = [n for n in G.nodes if G.nodes[n].get('record_id') is not None]
    
    # Get our outlier
    outlier_node = None
    for n in G.nodes:
        if G.nodes[n].get('is_outlier', False):
            outlier_node = n
            break
    
    if not outlier_node:
        print("Outlier node not found in network")
        return
    
    # Calculate bibliographic coupling strength (number of shared references)
    coupling_scores = {}
    
    outlier_references = set(G.neighbors(outlier_node))
    
    for node in sim_nodes:
        if node == outlier_node:
            continue
        
        node_references = set(G.neighbors(node))
        shared_refs = outlier_references.intersection(node_references)
        
        # Calculate Jaccard similarity for bibliographic coupling
        union_refs = outlier_references.union(node_references)
        jaccard = len(shared_refs) / len(union_refs) if union_refs else 0
        
        coupling_scores[node] = {
            'shared_count': len(shared_refs),
            'jaccard': jaccard,
            'record_id': G.nodes[node].get('record_id'),
            'asreview_ranking': G.nodes[node].get('asreview_ranking'),
            'label_included': G.nodes[node].get('label_included')
        }
    
    # Convert to DataFrame for analysis
    coupling_df = pd.DataFrame.from_dict(coupling_scores, orient='index')
    
    # Sort by Jaccard similarity descending
    coupling_df = coupling_df.sort_values('jaccard', ascending=False)
    
    # Print top results
    print("\nTop 10 documents by bibliographic coupling with outlier:")
    print(coupling_df.head(10).to_string())
    
    # Analyze relationship between coupling and relevance
    rel_coupling = coupling_df[coupling_df['label_included'] == 1]['jaccard'].mean()
    non_rel_coupling = coupling_df[coupling_df['label_included'] == 0]['jaccard'].mean()
    
    print(f"\nAverage bibliographic coupling (Jaccard similarity):")
    print(f"With relevant documents: {rel_coupling:.4f}")
    print(f"With non-relevant documents: {non_rel_coupling:.4f}")
    
    # Visualize bibliographic coupling vs. ranking
    plt.figure(figsize=(12, 6))
    
    # Plot for relevant documents
    plt.scatter(
        coupling_df[coupling_df['label_included'] == 1]['asreview_ranking'],
        coupling_df[coupling_df['label_included'] == 1]['jaccard'],
        alpha=0.7, s=100, c='blue', label='Relevant Documents'
    )
    
    # Plot for non-relevant documents (sample to avoid overplotting)
    non_rel_sample = coupling_df[coupling_df['label_included'] == 0].sample(
        min(100, len(coupling_df[coupling_df['label_included'] == 0]))
    )
    plt.scatter(
        non_rel_sample['asreview_ranking'],
        non_rel_sample['jaccard'],
        alpha=0.3, s=50, c='gray', label='Non-relevant Documents (Sample)'
    )
    
    plt.title('Bibliographic Coupling with Outlier vs. Document Ranking')
    plt.xlabel('ASReview Ranking')
    plt.ylabel('Bibliographic Coupling (Jaccard Similarity)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('output/figures/bibliographic_coupling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return coupling_df

def main():
    """Main execution function"""
    print("Starting citation analysis...")
    
    # Load data
    simulation_df, appenzeller_data = load_data()
    
    # Build citation network
    G = build_citation_network(appenzeller_data)
    
    # Map simulation data to network
    G = map_simulation_to_network(simulation_df, appenzeller_data, G)
    
    # Analyze outlier citations
    analyze_outlier_citations(G, simulation_df)
    
    # Calculate network metrics
    metrics_df = calculate_network_metrics(G)
    
    # Visualize network metrics
    visualize_network_metrics(metrics_df)
    
    # Visualize citation network
    visualize_citation_network(G)
    
    # Analyze bibliographic coupling
    analyze_bibliographic_coupling(G)
    
    # Clean graph for GEXF export (remove None values)
    print("\nPreparing network for export...")
    export_G = G.copy()
    
    # Replace None values in node attributes
    for node, attrs in export_G.nodes(data=True):
        for key, value in list(attrs.items()):
            if value is None:
                export_G.nodes[node][key] = "None"  # Convert None to string
    
    # Export network file for external visualization if needed
    nx.write_gexf(export_G, "output/citation_network.gexf")
    
    print("\nAnalysis complete! Results and visualizations saved to the 'output' directory.")

if __name__ == "__main__":
    main() 