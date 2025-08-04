import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime

class TopicOrganizer:
    def __init__(self, transcripts_folder="transcripts", output_folder="organized_topics"):
        self.transcripts_folder = transcripts_folder
        self.output_folder = output_folder
        self.model = None
        self.embeddings = None
        self.texts = []
        self.files = []
        self.metadata = []
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
    def load_transcripts(self):
        """Load all transcript files and extract text"""
        print("Loading transcript files...")
        
        transcript_files = [f for f in os.listdir(self.transcripts_folder) if f.endswith('.json')]
        print(f"Found {len(transcript_files)} transcript files")
        
        for file in transcript_files:
            try:
                with open(os.path.join(self.transcripts_folder, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                text = data.get('text', '').strip()
                if text:  # Only include non-empty texts
                    self.texts.append(text)
                    self.files.append(file)
                    
                    # Extract metadata
                    metadata = {
                        'file': file,
                        'detected_language': data.get('detected_language', 'unknown'),
                        'language_probability': data.get('language_probability', 0),
                        'segments_count': len(data.get('segments', [])),
                        'text_length': len(text)
                    }
                    self.metadata.append(metadata)
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.texts)} transcriptions")
        return len(self.texts)
    
    def generate_embeddings(self):
        """Generate sentence embeddings for topic clustering"""
        print("Loading multilingual sentence transformer model...")
        
        # Use multilingual model that works well with Arabic
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        print("Generating embeddings...")
        self.embeddings = self.model.encode(
            self.texts, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def cluster_by_topics(self, method='kmeans', n_clusters=10, eps=0.3, min_samples=5):
        """Cluster texts by topics using different algorithms"""
        print(f"Clustering using {method}...")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            raise ValueError("Method must be 'kmeans' or 'dbscan'")
        
        cluster_labels = clusterer.fit_predict(self.embeddings)
        
        # For DBSCAN, handle noise points (label -1)
        if method == 'dbscan':
            # Relabel noise points to a separate cluster
            max_label = cluster_labels.max()
            cluster_labels[cluster_labels == -1] = max_label + 1
        
        print(f"Created {len(set(cluster_labels))} clusters")
        return cluster_labels
    
    def extract_topic_keywords(self, cluster_labels):
        """Extract keywords for each cluster to create topic names"""
        print("Extracting topic keywords...")
        
        df = pd.DataFrame({
            'file': self.files,
            'text': self.texts,
            'cluster': cluster_labels,
            'metadata': self.metadata
        })
        
        topic_keywords = {}
        topic_summaries = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_texts = df[df['cluster'] == cluster_id]['text'].tolist()
            
            if len(cluster_texts) == 0:
                continue
                
            # Combine all texts in cluster
            combined_text = ' '.join(cluster_texts)
            
            # Extract keywords using TF-IDF
            try:
                # Adjust parameters based on cluster size
                cluster_size = len(cluster_texts)
                min_df = max(1, cluster_size // 2)  # At least half the documents
                max_features = min(5, cluster_size)  # Don't exceed cluster size
                
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words=None,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=1.0  # Allow terms that appear in all documents
                )
                
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                if len(feature_names) > 0:
                    # Get top keywords
                    tfidf_scores = tfidf_matrix.toarray().mean(axis=0)
                    top_indices = tfidf_scores.argsort()[-min(3, len(feature_names)):][::-1]
                    keywords = [feature_names[i] for i in top_indices]
                else:
                    # Fallback: use first few words from the text
                    words = combined_text.split()[:3]
                    keywords = [word for word in words if len(word) > 2]
                
                topic_keywords[cluster_id] = keywords
                
                # Create a simple summary (first 100 chars of first text)
                summary = cluster_texts[0][:100] + "..." if len(cluster_texts[0]) > 100 else cluster_texts[0]
                topic_summaries[cluster_id] = summary
                
            except Exception as e:
                print(f"Error extracting keywords for cluster {cluster_id}: {e}")
                # Fallback: use cluster ID and first few words
                words = combined_text.split()[:3]
                keywords = [f"topic_{cluster_id}"] + [word for word in words if len(word) > 2]
                topic_keywords[cluster_id] = keywords
                topic_summaries[cluster_id] = "No summary available"
        
        return df, topic_keywords, topic_summaries
    
    def save_organized_topics(self, df, topic_keywords, topic_summaries):
        """Save organized topics to files"""
        print("Saving organized topics...")
        
        # Create summary file
        summary_data = {
            'total_files': len(df),
            'total_clusters': int(len(df['cluster'].unique())),
            'clustering_date': datetime.now().isoformat(),
            'clusters': {}
        }
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == int(cluster_id)]
            
            # Create cluster folder name
            keywords_str = '_'.join(topic_keywords.get(cluster_id, [f"topic_{cluster_id}"]))
            folder_name = f"cluster_{int(cluster_id):02d}_{keywords_str}"
            folder_name = re.sub(r'[^\w\s-]', '', folder_name)[:50]  # Clean folder name
            
            cluster_folder = os.path.join(self.output_folder, folder_name)
            os.makedirs(cluster_folder, exist_ok=True)
            
            # Save cluster data
            cluster_data = {
                'cluster_id': int(cluster_id),
                'keywords': topic_keywords.get(cluster_id, []),
                'summary': topic_summaries.get(cluster_id, ''),
                'file_count': len(cluster_df),
                'files': cluster_df.to_dict('records')
            }
            
            with open(os.path.join(cluster_folder, 'cluster_info.json'), 'w', encoding='utf-8') as f:
                json.dump(cluster_data, f, ensure_ascii=False, indent=2)
            
            # Save individual files
            for _, row in cluster_df.iterrows():
                file_data = {
                    'original_file': row['file'],
                    'text': row['text'],
                    'cluster_id': int(row['cluster']),
                    'metadata': row['metadata']
                }
                
                output_filename = f"transcript_{row['file'].replace('.json', '')}.json"
                with open(os.path.join(cluster_folder, output_filename), 'w', encoding='utf-8') as f:
                    json.dump(file_data, f, ensure_ascii=False, indent=2)
            
            # Add to summary
            summary_data['clusters'][int(cluster_id)] = {
                'folder_name': folder_name,
                'keywords': topic_keywords.get(cluster_id, []),
                'summary': topic_summaries.get(cluster_id, ''),
                'file_count': len(cluster_df)
            }
        
        # Save overall summary
        with open(os.path.join(self.output_folder, 'organization_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved organized topics to {self.output_folder}")
    
    def visualize_clusters(self, cluster_labels, topic_keywords):
        """Create visualization of clusters"""
        print("Creating cluster visualization...")
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # Plot clusters
        unique_labels = set(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1], 
                c=[colors[i]], 
                label=f'Cluster {label}: {", ".join(topic_keywords.get(label, [f"topic_{label}"]))[:30]}...',
                alpha=0.7
            )
        
        plt.title('Topic Clusters Visualization (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_folder, 'cluster_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved cluster visualization")
    
    def run_organization(self, method='kmeans', n_clusters=10):
        """Run the complete topic organization pipeline"""
        print("=== Starting Topic Organization ===")
        
        # Step 1: Load transcripts
        num_transcripts = self.load_transcripts()
        if num_transcripts == 0:
            print("No transcripts found!")
            return
        
        # Step 2: Generate embeddings
        self.generate_embeddings()
        
        # Step 3: Cluster by topics
        cluster_labels = self.cluster_by_topics(method=method, n_clusters=n_clusters)
        
        # Step 4: Extract keywords and summaries
        df, topic_keywords, topic_summaries = self.extract_topic_keywords(cluster_labels)
        
        # Step 5: Save organized topics
        self.save_organized_topics(df, topic_keywords, topic_summaries)
        
        # Step 6: Create visualization
        try:
            self.visualize_clusters(cluster_labels, topic_keywords)
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        # Print summary
        print("\n=== Organization Complete ===")
        print(f"Total files processed: {len(df)}")
        print(f"Number of clusters: {len(df['cluster'].unique())}")
        print(f"Output folder: {self.output_folder}")
        
        # Show cluster summary
        print("\nCluster Summary:")
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_size = len(df[df['cluster'] == cluster_id])
            keywords = topic_keywords.get(cluster_id, [f"topic_{cluster_id}"])
            print(f"  Cluster {cluster_id}: {cluster_size} files - Keywords: {', '.join(keywords[:3])}")

def main():
    """Main function to run topic organization"""
    organizer = TopicOrganizer()
    
    # You can adjust these parameters
    method = 'kmeans'  # 'kmeans' or 'dbscan'
    n_clusters = 15    # Number of clusters (for kmeans)
    
    organizer.run_organization(method=method, n_clusters=n_clusters)

if __name__ == "__main__":
    main() 