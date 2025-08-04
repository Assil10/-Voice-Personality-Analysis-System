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
import google.generativeai as genai
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EnhancedTopicOrganizer:
    def __init__(self, transcripts_folder="transcripts", output_folder="organized_topics_enhanced"):
        self.transcripts_folder = transcripts_folder
        self.output_folder = output_folder
        self.model = None
        self.embeddings = None
        self.texts = []
        self.files = []
        self.metadata = []
        self.dates = []
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize Gemini client
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_ai_summaries = True
                print("✅ Gemini API configured successfully!")
            else:
                print("⚠️  No GEMINI_API_KEY found. Set GEMINI_API_KEY environment variable for AI summaries.")
                print("   The system will use basic summaries instead.")
                self.use_ai_summaries = False
        except Exception as e:
            print(f"⚠️  Error configuring Gemini API: {e}")
            print("   The system will use basic summaries instead.")
            self.use_ai_summaries = False
        
    def extract_date_from_filename(self, filename):
        """Extract date from filename patterns like audioclip17166390760007570_..."""
        try:
            # Look for timestamp patterns in filename - the format is audioclip + 17 digits
            timestamp_pattern = r'audioclip(\d{17})'
            match = re.search(timestamp_pattern, filename)
            
            if match:
                timestamp = int(match.group(1))
                
                # Based on debug output, these appear to be in microseconds
                # Convert to seconds by dividing by 1,000,000
                timestamp_seconds = timestamp / 1000000
                
                # Check if the result is reasonable (between 1970 and 2030)
                if 0 < timestamp_seconds < 2000000000:  # Reasonable Unix timestamp range
                    date_obj = datetime.fromtimestamp(timestamp_seconds)
                    return date_obj
                else:
                    # Try alternative: maybe it's in a different epoch or format
                    # Try dividing by different factors
                    for divisor in [10000, 100000, 1000000, 10000000]:
                        test_timestamp = timestamp / divisor
                        if 0 < test_timestamp < 2000000000:
                            try:
                                date_obj = datetime.fromtimestamp(test_timestamp)
                                # Check if the date is reasonable (not in the future or too far in the past)
                                if 2020 <= date_obj.year <= 2030:
                                    return date_obj
                            except:
                                continue
                
                return None
            else:
                # Try alternative patterns
                timestamp_pattern_alt = r'audioclip(\d{13,15})'
                match = re.search(timestamp_pattern_alt, filename)
                
                if match:
                    timestamp = int(match.group(1))
                    # Convert milliseconds to seconds if needed
                    if timestamp > 1e12:  # If it's in milliseconds
                        timestamp = timestamp / 1000
                    
                    date_obj = datetime.fromtimestamp(timestamp)
                    return date_obj
                
                # Try to extract any date-like patterns
                date_patterns = [
                    r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})',  # YYYY-MM-DD or YYYY_MM_DD
                    r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})',  # MM-DD-YYYY or MM_DD_YYYY
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, filename)
                    if match:
                        if len(match.group(1)) == 4:  # YYYY-MM-DD
                            year, month, day = match.groups()
                        else:  # MM-DD-YYYY
                            month, day, year = match.groups()
                        
                        return datetime(int(year), int(month), int(day))
                
                return None
        except Exception as e:
            print(f"Error extracting date from {filename}: {e}")
            return None
    
    def load_transcripts(self):
        """Load all transcript files and extract text with dates"""
        print("Loading transcript files...")
        
        transcript_files = [f for f in os.listdir(self.transcripts_folder) if f.endswith('.json')]
        print(f"Found {len(transcript_files)} transcript files")
        
        for file in transcript_files:
            try:
                with open(os.path.join(self.transcripts_folder, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Try enhanced_text first, then original_text, then text
                text = data.get('enhanced_text', data.get('original_text', data.get('text', ''))).strip()
                if text:  # Only include non-empty texts
                    self.texts.append(text)
                    self.files.append(file)
                    
                    # Extract date from filename
                    date_obj = self.extract_date_from_filename(file)
                    self.dates.append(date_obj)
                    
                    # Extract metadata
                    metadata = {
                        'file': file,
                        'detected_language': data.get('detected_language', 'unknown'),
                        'language_probability': data.get('language_probability', 0),
                        'segments_count': len(data.get('segments', [])),
                        'text_length': len(text),
                        'date': date_obj.isoformat() if date_obj else None,
                        'date_obj': date_obj
                    }
                    self.metadata.append(metadata)
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.texts)} transcriptions")
        print(f"Found dates for {sum(1 for d in self.dates if d is not None)} files")
        return len(self.texts)
    
    def generate_embeddings(self):
        """Generate sentence embeddings for topic clustering"""
        print("Loading BAAI/bge-large-zh-v1.5 model (excellent for Arabic)...")
        
        # Use BAAI/bge-large-zh-v1.5 - specifically excellent for Arabic and multilingual tasks
        # This model has 1024 dimensions vs 384, providing much richer representations
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        
        print("Generating embeddings...")
        self.embeddings = self.model.encode(
            self.texts, 
            show_progress_bar=True,
            batch_size=16,  # Reduced batch size for larger model
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
    
    def generate_ai_summary(self, texts, keywords, cluster_id):
        """Generate AI-powered summary for a cluster using Gemini"""
        if not self.use_ai_summaries:
            return f"Cluster {cluster_id}: {texts[0][:100]}..."
        
        try:
            # Combine all texts in the cluster
            combined_text = " ".join(texts)
            
            # Create prompt for summary
            prompt = f"""
            You are analyzing a cluster of Arabic voice message transcriptions. 
            
            Keywords identified: {', '.join(keywords)}
            Number of messages: {len(texts)}
            
            Here are the transcriptions:
            {combined_text}
            
            Please provide a concise summary in English that:
            1. Identifies the main topic/theme of these messages
            2. Describes what the conversation is about
            3. Mentions any key points or recurring themes
            4. Is 2-3 sentences long
            
            Summary:
            """
            
            response = self.gemini_model.generate_content(prompt)
            summary = response.text.strip()
            return summary
            
        except Exception as e:
            print(f"Error generating AI summary for cluster {cluster_id}: {e}")
            return f"Cluster {cluster_id}: {texts[0][:100]}..."
    
    def extract_topic_keywords(self, cluster_labels):
        """Extract keywords for each cluster to create topic names"""
        print("Extracting topic keywords...")
        
        df = pd.DataFrame({
            'file': self.files,
            'text': self.texts,
            'cluster': cluster_labels,
            'metadata': self.metadata,
            'date': self.dates
        })
        
        topic_keywords = {}
        topic_summaries = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_texts = cluster_df['text'].tolist()
            
            if len(cluster_texts) == 0:
                continue
                
            # Combine all texts in cluster
            combined_text = ' '.join(cluster_texts)
            
            # Extract keywords using TF-IDF
            try:
                # Adjust parameters based on cluster size
                cluster_size = len(cluster_texts)
                min_df = max(1, cluster_size // 2)  # At least half the documents
                max_features = min(10, cluster_size * 2)  # More features for better keywords
                
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words=None,
                    ngram_range=(1, 3),  # Include trigrams for better phrases
                    min_df=min_df,
                    max_df=1.0  # Allow terms that appear in all documents
                )
                
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                if len(feature_names) > 0:
                    # Get top keywords
                    tfidf_scores = tfidf_matrix.toarray().mean(axis=0)
                    top_indices = tfidf_scores.argsort()[-min(5, len(feature_names)):][::-1]
                    keywords = [feature_names[i] for i in top_indices]
                else:
                    # Fallback: use first few words from the text
                    words = combined_text.split()[:5]
                    keywords = [word for word in words if len(word) > 2]
                
                topic_keywords[cluster_id] = keywords
                
                # Generate AI summary
                summary = self.generate_ai_summary(cluster_texts, keywords, cluster_id)
                topic_summaries[cluster_id] = summary
                
            except Exception as e:
                print(f"Error extracting keywords for cluster {cluster_id}: {e}")
                # Fallback: use cluster ID and first few words
                words = combined_text.split()[:3]
                keywords = [f"topic_{cluster_id}"] + [word for word in words if len(word) > 2]
                topic_keywords[cluster_id] = keywords
                topic_summaries[cluster_id] = f"Cluster {cluster_id}: {cluster_texts[0][:100]}..."
        
        return df, topic_keywords, topic_summaries
    
    def save_organized_topics(self, df, topic_keywords, topic_summaries):
        """Save organized topics to files with chronological ordering"""
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
            
            # Sort by date if available
            cluster_df = cluster_df.sort_values('date', na_position='last')
            
            # Create cluster folder name
            keywords_str = '_'.join(topic_keywords.get(cluster_id, [f"topic_{cluster_id}"])[:3])
            folder_name = f"cluster_{int(cluster_id):02d}_{keywords_str}"
            folder_name = re.sub(r'[^\w\s-]', '', folder_name)[:50]  # Clean folder name
            
            cluster_folder = os.path.join(self.output_folder, folder_name)
            os.makedirs(cluster_folder, exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            cluster_records = []
            for _, row in cluster_df.iterrows():
                record = row.to_dict()
                # Convert datetime objects to ISO format strings
                if record.get('date') and hasattr(record['date'], 'isoformat'):
                    record['date'] = record['date'].isoformat()
                if record.get('metadata', {}).get('date_obj') and hasattr(record['metadata']['date_obj'], 'isoformat'):
                    record['metadata']['date_obj'] = record['metadata']['date_obj'].isoformat()
                cluster_records.append(record)
            
            # Save cluster data
            cluster_data = {
                'cluster_id': int(cluster_id),
                'keywords': topic_keywords.get(cluster_id, []),
                'ai_summary': topic_summaries.get(cluster_id, ''),
                'file_count': len(cluster_df),
                'files': cluster_records,
                'date_range': {
                    'earliest': cluster_df['date'].min().isoformat() if cluster_df['date'].notna().any() and hasattr(cluster_df['date'].min(), 'isoformat') else None,
                    'latest': cluster_df['date'].max().isoformat() if cluster_df['date'].notna().any() and hasattr(cluster_df['date'].max(), 'isoformat') else None
                }
            }
            
            with open(os.path.join(cluster_folder, 'cluster_info.json'), 'w', encoding='utf-8') as f:
                json.dump(cluster_data, f, ensure_ascii=False, indent=2)
            
            # Save individual files in chronological order
            for idx, (_, row) in enumerate(cluster_df.iterrows()):
                file_data = {
                    'original_file': row['file'],
                    'text': row['text'],
                    'cluster_id': int(row['cluster']),
                    'metadata': row['metadata'],
                    'chronological_order': idx + 1,
                    'date': row['date'].isoformat() if hasattr(row['date'], 'isoformat') else row['date']
                }
                
                # Convert datetime objects in metadata
                if file_data['metadata'].get('date_obj') and hasattr(file_data['metadata']['date_obj'], 'isoformat'):
                    file_data['metadata']['date_obj'] = file_data['metadata']['date_obj'].isoformat()
                
                output_filename = f"{idx+1:02d}_transcript_{row['file'].replace('.json', '')}.json"
                with open(os.path.join(cluster_folder, output_filename), 'w', encoding='utf-8') as f:
                    json.dump(file_data, f, ensure_ascii=False, indent=2)
            
            # Add to summary
            summary_data['clusters'][int(cluster_id)] = {
                'folder_name': folder_name,
                'keywords': topic_keywords.get(cluster_id, []),
                'ai_summary': topic_summaries.get(cluster_id, ''),
                'file_count': len(cluster_df),
                'date_range': {
                    'earliest': cluster_df['date'].min().isoformat() if cluster_df['date'].notna().any() and hasattr(cluster_df['date'].min(), 'isoformat') else None,
                    'latest': cluster_df['date'].max().isoformat() if cluster_df['date'].notna().any() and hasattr(cluster_df['date'].max(), 'isoformat') else None
                }
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
        
        plt.title('Enhanced Topic Clusters Visualization (t-SNE)', fontsize=16)
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
        """Run the complete enhanced topic organization pipeline"""
        print("=== Starting Enhanced Topic Organization ===")
        
        # Step 1: Load transcripts
        num_transcripts = self.load_transcripts()
        if num_transcripts == 0:
            print("No transcripts found!")
            return
        
        # Step 2: Generate embeddings
        self.generate_embeddings()
        
        # Step 3: Cluster by topics
        cluster_labels = self.cluster_by_topics(method=method, n_clusters=n_clusters)
        
        # Step 4: Extract keywords and generate AI summaries
        df, topic_keywords, topic_summaries = self.extract_topic_keywords(cluster_labels)
        
        # Step 5: Save organized topics
        self.save_organized_topics(df, topic_keywords, topic_summaries)
        
        # Step 6: Create visualization
        try:
            self.visualize_clusters(cluster_labels, topic_keywords)
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        # Print summary
        print("\n=== Enhanced Organization Complete ===")
        print(f"Total files processed: {len(df)}")
        print(f"Number of clusters: {len(df['cluster'].unique())}")
        print(f"Output folder: {self.output_folder}")
        
        # Show cluster summary with AI summaries
        print("\nCluster Summary:")
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_size = len(df[df['cluster'] == cluster_id])
            keywords = topic_keywords.get(cluster_id, [f"topic_{cluster_id}"])
            summary = topic_summaries.get(cluster_id, "No summary available")
            print(f"\n  Cluster {cluster_id}: {cluster_size} files")
            print(f"    Keywords: {', '.join(keywords[:3])}")
            print(f"    Summary: {summary}")

def main():
    """Main function to run enhanced topic organization"""
    organizer = EnhancedTopicOrganizer()
    
    # You can adjust these parameters
    method = 'kmeans'  # 'kmeans' or 'dbscan'
    n_clusters = 4     # Number of clusters (adjusted for your data)
    
    organizer.run_organization(method=method, n_clusters=n_clusters)

if __name__ == "__main__":
    main() 