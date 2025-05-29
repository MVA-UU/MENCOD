import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import Counter
import re
from wordcloud import WordCloud

# Create directories if they don't exist
os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)

def load_data():
    """Load the simulation dataset"""
    df = pd.read_csv('../data/simulation.csv')
    print(f"Loaded dataset with {len(df)} records")
    return df

def preprocess_text(text):
    """Basic text preprocessing"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def analyze_text_similarity(df):
    """Analyze text similarity between the outlier and other documents"""
    print("\n=== Text Similarity Analysis ===")
    
    # Get outlier and other relevant documents
    outlier = df[df['record_id'] == 497]
    relevant_docs = df[df['label_included'] == 1]
    relevant_nonoutlier = relevant_docs[relevant_docs['record_id'] != 497]
    
    # Extract abstracts and titles
    abstracts = df['abstract'].fillna('').apply(preprocess_text)
    titles = df['title'].fillna('').apply(preprocess_text)
    
    # Create TF-IDF matrices
    print("Computing TF-IDF for abstracts...")
    abstract_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    abstract_tfidf = abstract_vectorizer.fit_transform(abstracts)
    
    print("Computing TF-IDF for titles...")
    title_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    title_tfidf = title_vectorizer.fit_transform(titles)
    
    # Get indices
    outlier_idx = outlier.index[0]
    relevant_nonoutlier_idx = relevant_nonoutlier.index.tolist()
    
    # Calculate similarities
    abstract_similarities = cosine_similarity(abstract_tfidf[outlier_idx:outlier_idx+1], abstract_tfidf[relevant_nonoutlier_idx])[0]
    title_similarities = cosine_similarity(title_tfidf[outlier_idx:outlier_idx+1], title_tfidf[relevant_nonoutlier_idx])[0]
    
    # Create result DataFrame
    similarity_df = pd.DataFrame({
        'record_id': relevant_nonoutlier['record_id'].values,
        'asreview_ranking': relevant_nonoutlier['asreview_ranking'].values,
        'abstract_similarity': abstract_similarities,
        'title_similarity': title_similarities,
        'average_similarity': (abstract_similarities + title_similarities) / 2
    })
    
    # Sort by average similarity
    similarity_df = similarity_df.sort_values('average_similarity', ascending=False)
    
    print("\nTop 10 relevant documents by text similarity to outlier:")
    print(similarity_df.head(10).to_string(index=False))
    
    # Calculate average similarities
    avg_abstract_sim = similarity_df['abstract_similarity'].mean()
    avg_title_sim = similarity_df['title_similarity'].mean()
    
    print(f"\nAverage abstract similarity with outlier: {avg_abstract_sim:.4f}")
    print(f"Average title similarity with outlier: {avg_title_sim:.4f}")
    
    # Visualize similarity distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(similarity_df['average_similarity'], bins=20, kde=True)
    plt.axvline(x=similarity_df['average_similarity'].mean(), color='red', linestyle='--', 
                label=f'Mean: {similarity_df["average_similarity"].mean():.4f}')
    plt.title('Distribution of Text Similarity Between Outlier and Other Relevant Documents')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('output/figures/similarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize ranking vs. similarity
    plt.figure(figsize=(12, 6))
    plt.scatter(similarity_df['asreview_ranking'], similarity_df['average_similarity'], 
                alpha=0.7, s=100, c=similarity_df['asreview_ranking'], cmap='viridis')
    plt.colorbar(label='ASReview Ranking')
    
    # Add trend line
    z = np.polyfit(similarity_df['asreview_ranking'], similarity_df['average_similarity'], 1)
    p = np.poly1d(z)
    plt.plot(similarity_df['asreview_ranking'], p(similarity_df['asreview_ranking']), "r--", alpha=0.7)
    
    plt.title('Relationship Between ASReview Ranking and Text Similarity to Outlier')
    plt.xlabel('ASReview Ranking')
    plt.ylabel('Average Text Similarity')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/figures/ranking_vs_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return similarity_df

def analyze_key_terms(df):
    """Analyze key terms that distinguish the outlier from other documents"""
    print("\n=== Key Terms Analysis ===")
    
    # Get outlier and other relevant documents
    outlier = df[df['record_id'] == 497]
    relevant_docs = df[df['label_included'] == 1]
    relevant_nonoutlier = relevant_docs[relevant_docs['record_id'] != 497]
    
    # Combine title and abstract for analysis
    outlier_text = outlier['title'].fillna('') + ' ' + outlier['abstract'].fillna('')
    relevant_text = relevant_nonoutlier['title'].fillna('') + ' ' + relevant_nonoutlier['abstract'].fillna('')
    
    # Preprocess texts
    outlier_text = outlier_text.apply(preprocess_text)
    relevant_text = relevant_text.apply(preprocess_text)
    
    # Create document-term matrices
    vectorizer = CountVectorizer(max_features=10000, stop_words='english', min_df=2)
    
    # Fit on all texts
    all_text = pd.concat([outlier_text, relevant_text])
    dtm = vectorizer.fit_transform(all_text)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get outlier terms
    outlier_dtm = dtm[0:1]
    outlier_terms = outlier_dtm.toarray()[0]
    
    # Get average term frequencies for other relevant documents
    relevant_dtm = dtm[1:]
    relevant_terms = relevant_dtm.mean(axis=0).A1
    
    # Calculate term importance ratio (outlier vs. relevant)
    # Add small constant to avoid division by zero
    term_ratio = np.log((outlier_terms + 0.1) / (relevant_terms + 0.1))
    
    # Get distinctive terms (terms that appear more in outlier)
    distinctive_indices = np.argsort(-term_ratio)
    distinctive_terms = [(feature_names[i], term_ratio[i]) for i in distinctive_indices 
                         if outlier_terms[i] > 0][:30]
    
    print("\nTop distinctive terms in outlier (compared to other relevant documents):")
    for term, score in distinctive_terms[:15]:
        print(f"{term}: {score:.4f}")
    
    # Create word cloud of distinctive terms
    if distinctive_terms:
        term_dict = {term: max(score, 0.1) for term, score in distinctive_terms}
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             max_words=50, colormap='viridis')
        wordcloud.generate_from_frequencies(term_dict)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Distinctive Terms in Outlier Document')
        plt.axis('off')
        plt.savefig('output/figures/outlier_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return distinctive_terms

def create_topic_model(df):
    """Create a topic model to see if the outlier covers different topics"""
    print("\n=== Topic Modeling Analysis ===")
    
    # Prepare text data
    abstracts = df['abstract'].fillna('').apply(preprocess_text)
    
    # Create document-term matrix
    vectorizer = CountVectorizer(max_features=5000, stop_words='english', min_df=5)
    dtm = vectorizer.fit_transform(abstracts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Create topic model
    n_topics = 8
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    
    print(f"Fitting LDA topic model with {n_topics} topics...")
    topic_distributions = lda.fit_transform(dtm)
    
    # Get top terms for each topic
    n_top_words = 10
    topic_terms = []
    for topic_idx, topic in enumerate(lda.components_):
        top_terms_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_terms = [feature_names[i] for i in top_terms_idx]
        topic_terms.append(top_terms)
        print(f"Topic #{topic_idx+1}: {', '.join(top_terms)}")
    
    # Add topic distributions to the dataframe
    topic_df = pd.DataFrame(topic_distributions, columns=[f'Topic {i+1}' for i in range(n_topics)])
    topic_df['record_id'] = df['record_id'].values
    topic_df['label_included'] = df['label_included'].values
    topic_df['is_outlier'] = df['record_id'] == 497
    
    # Get outlier's topic distribution
    outlier_topics = topic_df[topic_df['is_outlier']].iloc[0, :n_topics]
    dominant_topic = outlier_topics.argmax()
    
    print(f"\nOutlier's dominant topic: Topic #{dominant_topic+1}")
    print(f"Topic distribution for outlier: {outlier_topics.values}")
    
    # Compare to average topic distribution of other relevant documents
    relevant_nonoutlier = topic_df[(topic_df['label_included'] == 1) & (~topic_df['is_outlier'])]
    avg_topics = relevant_nonoutlier.iloc[:, :n_topics].mean()
    
    print(f"Average topic distribution for other relevant documents: {avg_topics.values}")
    
    # Visualize topic distributions
    plt.figure(figsize=(12, 6))
    
    # Plot outlier's topic distribution
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_topics+1), outlier_topics, color='blue', alpha=0.7)
    plt.title('Outlier Document Topic Distribution')
    plt.xlabel('Topic Number')
    plt.ylabel('Topic Weight')
    plt.xticks(range(1, n_topics+1))
    plt.grid(True, alpha=0.3)
    
    # Plot average distribution for other relevant documents
    plt.subplot(1, 2, 2)
    plt.bar(range(1, n_topics+1), avg_topics, color='green', alpha=0.7)
    plt.title('Average Topic Distribution (Other Relevant Docs)')
    plt.xlabel('Topic Number')
    plt.ylabel('Topic Weight')
    plt.xticks(range(1, n_topics+1))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/topic_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize documents in topic space using t-SNE
    print("Computing t-SNE projection of topic space...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(topic_df)-1))
    tsne_result = tsne.fit_transform(topic_distributions)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot non-relevant documents
    plt.scatter(
        tsne_result[df['label_included'] == 0, 0],
        tsne_result[df['label_included'] == 0, 1],
        alpha=0.3, s=30, c='gray', label='Non-relevant'
    )
    
    # Plot relevant documents
    relevant_mask = (df['label_included'] == 1) & (df['record_id'] != 497)
    plt.scatter(
        tsne_result[relevant_mask, 0],
        tsne_result[relevant_mask, 1],
        alpha=0.7, s=80, c='blue', label='Relevant'
    )
    
    # Highlight the outlier
    outlier_idx = df[df['record_id'] == 497].index[0]
    plt.scatter(
        tsne_result[outlier_idx, 0],
        tsne_result[outlier_idx, 1],
        alpha=1.0, s=200, c='red', marker='*', label='Outlier'
    )
    
    plt.title('Documents in Topic Space (t-SNE)')
    plt.legend()
    plt.savefig('output/figures/topic_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return topic_df

def main():
    """Main execution function"""
    print("Starting text analysis...")
    
    # Load data
    df = load_data()
    
    # Analyze text similarity
    similarity_df = analyze_text_similarity(df)
    
    # Analyze key terms
    distinctive_terms = analyze_key_terms(df)
    
    # Create topic model
    topic_df = create_topic_model(df)
    
    print("\nText analysis complete! Results and visualizations saved to the 'output' directory.")

if __name__ == "__main__":
    main() 