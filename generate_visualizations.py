#!/usr/bin/env python3
"""
Generate Voice Personality Visualizations
========================================
Saves visualizations to files without displaying them
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

class GenerateVisualizations:
    def __init__(self, analysis_folder="analysis_results"):
        self.analysis_folder = analysis_folder
        self.data = self.load_latest_analysis()
        self.visualization_data = {}
        
        # Create visualizations folder
        self.visualizations_folder = "visualizations"
        os.makedirs(self.visualizations_folder, exist_ok=True)
        
    def load_latest_analysis(self):
        """Load the most recent analysis data"""
        if not os.path.exists(self.analysis_folder):
            return None
            
        # Find the most recent analysis folder
        folders = [f for f in os.listdir(self.analysis_folder) if f.startswith(('multimodal_analysis_', 'standalone_analysis_', 'personality_analysis_'))]
        if not folders:
            return None
            
        latest_folder = max(folders, key=lambda x: os.path.getctime(os.path.join(self.analysis_folder, x)))
        data_file = os.path.join(self.analysis_folder, latest_folder, "multimodal_analysis.json")
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def create_topic_clusters_visualization(self):
        """Create topic clusters visualization"""
        if not self.data or 'topic_clusters' not in self.data:
            print("❌ No cluster data available")
            return
            
        topic_clusters = self.data['topic_clusters']
        
        # Prepare data
        cluster_ids = [f"Cluster {cluster['cluster_id']}" for cluster in topic_clusters]
        sizes = [cluster['size'] for cluster in topic_clusters]
        keywords = [', '.join(cluster.get('top_words', [])[:3]) for cluster in topic_clusters]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart of cluster sizes
        bars = ax1.bar(cluster_ids, sizes, color=sns.color_palette("husl", len(topic_clusters)))
        ax1.set_title('Topic Cluster Sizes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Messages')
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{size}', ha='center', va='bottom')
        
        # Pie chart of cluster distribution
        ax2.pie(sizes, labels=cluster_ids, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Cluster Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_folder, 'topic_clusters_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store data for GUI
        self.visualization_data['topic_clusters'] = {
            'cluster_ids': cluster_ids,
            'sizes': sizes,
            'keywords': keywords
        }
        
        print("✅ Topic clusters visualization saved")

    def create_sentiment_timeline(self):
        """Create sentiment timeline visualization"""
        if not self.data or 'sentiment_data' not in self.data:
            print("❌ No sentiment data available")
            return
            
        sentiment_data = self.data['sentiment_data']
        
        # Prepare timeline data
        positive_count = sum(1 for s in sentiment_data if s['sentiment']['state'] == 'positive')
        negative_count = sum(1 for s in sentiment_data if s['sentiment']['state'] == 'negative')
        neutral_count = sum(1 for s in sentiment_data if s['sentiment']['state'] == 'neutral')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive_count, negative_count, neutral_count]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_title('Sentiment Counts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Messages')
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_folder, 'sentiment_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store data for GUI
        self.visualization_data['sentiment'] = {
            'labels': labels,
            'sizes': sizes,
            'colors': colors,
            'total': len(sentiment_data),
            'positive_percent': (positive_count / len(sentiment_data)) * 100,
            'negative_percent': (negative_count / len(sentiment_data)) * 100,
            'neutral_percent': (neutral_count / len(sentiment_data)) * 100
        }
        
        print("✅ Sentiment timeline visualization saved")

    def create_personality_radar(self):
        """Create personality radar chart"""
        if not self.data or 'voice_data' not in self.data:
            print("❌ No voice data available")
            return
            
        voice_data = self.data['voice_data']
        
        # Extract personality traits from voice data
        personality_traits = {}
        for voice_item in voice_data:
            if 'personality' in voice_item:
                for trait, value in voice_item['personality'].items():
                    if trait not in personality_traits:
                        personality_traits[trait] = []
                    personality_traits[trait].append(value)
        
        if not personality_traits:
            print("❌ No personality traits found")
            return
        
        # Convert string values to numeric scores
        trait_scores = {}
        for trait, values in personality_traits.items():
            numeric_values = []
            for value in values:
                if value == 'high':
                    numeric_values.append(3)
                elif value == 'medium':
                    numeric_values.append(2)
                elif value == 'low':
                    numeric_values.append(1)
                else:
                    numeric_values.append(2)  # Default to medium
            
            trait_scores[trait] = np.mean(numeric_values)
        
        # Create radar chart
        categories = list(trait_scores.keys())
        values = list(trait_scores.values())
        
        # Create radar chart
        categories = list(trait_scores.keys())
        values = list(trait_scores.values())
        
        # Number of variables
        N = len(categories)
        
        # Create angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add the first value at the end to close the plot
        values += values[:1]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label='Personality Profile')
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max(values) * 1.1)
        ax.set_title('Personality Radar Chart', size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_folder, 'personality_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store data for GUI
        self.visualization_data['personality'] = {
            'categories': categories,
            'values': values[:-1],  # Remove the duplicate first value
            'trait_scores': trait_scores
        }
        
        print("✅ Personality radar chart saved")

    def create_voice_characteristics_heatmap(self):
        """Create voice characteristics heatmap"""
        if not self.data or 'voice_data' not in self.data:
            print("❌ No voice data available")
            return
            
        voice_data = self.data['voice_data']
        
        # Extract voice characteristics
        characteristics = ['duration', 'energy', 'pitch', 'tempo', 'spectral_centroid', 'spectral_bandwidth']
        data_matrix = []
        
        for voice_item in voice_data:
            row = []
            for char in characteristics:
                if char in voice_item:
                    row.append(voice_item[char])
                else:
                    row.append(0)
            data_matrix.append(row)
        
        if not data_matrix:
            print("❌ No voice characteristics data available")
            return
        
        # Create heatmap
        df = pd.DataFrame(data_matrix, columns=characteristics)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Voice Characteristics Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_folder, 'voice_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store data for GUI
        self.visualization_data['voice_characteristics'] = {
            'characteristics': characteristics,
            'correlation_matrix': df.corr().to_dict(),
            'summary_stats': df.describe().to_dict()
        }
        
        print("✅ Voice characteristics heatmap saved")

    def create_communication_style_analysis(self):
        """Create communication style analysis"""
        if not self.data or 'sentiment_data' not in self.data:
            print("❌ No sentiment data available")
            return
            
        sentiment_data = self.data['sentiment_data']
        
        # Analyze communication patterns
        communication_styles = {
            'Direct': 0,
            'Indirect': 0,
            'Formal': 0,
            'Informal': 0,
            'Emotional': 0,
            'Analytical': 0
        }
        
        # Simple analysis based on sentiment and text length
        for item in sentiment_data:
            text_length = len(item.get('text', ''))
            sentiment_score = item['sentiment']['score']
            
            # Direct vs Indirect (based on sentiment strength)
            if abs(sentiment_score) > 0.5:
                communication_styles['Direct'] += 1
            else:
                communication_styles['Indirect'] += 1
            
            # Formal vs Informal (based on text length)
            if text_length > 50:
                communication_styles['Formal'] += 1
            else:
                communication_styles['Informal'] += 1
            
            # Emotional vs Analytical (based on sentiment)
            if sentiment_score > 0.3 or sentiment_score < -0.3:
                communication_styles['Emotional'] += 1
            else:
                communication_styles['Analytical'] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        styles = list(communication_styles.keys())
        counts = list(communication_styles.values())
        colors = sns.color_palette("husl", len(styles))
        
        bars = ax1.bar(styles, counts, color=colors)
        ax1.set_title('Communication Styles', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=styles, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Communication Style Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_folder, 'communication_styles.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store data for GUI
        self.visualization_data['communication_styles'] = {
            'styles': styles,
            'counts': counts,
            'colors': colors,
            'total_messages': len(sentiment_data)
        }
        
        print("✅ Communication styles analysis saved")

    def save_visualization_data(self):
        """Save all visualization data to a JSON file"""
        output_file = os.path.join(self.visualizations_folder, 'visualization_data.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.visualization_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Visualization data saved to {output_file}")

    def generate_all_visualizations(self):
        """Generate all visualizations and save data"""
        print("🎨 Generating Voice Personality Visualizations...")
        print("=" * 60)
        
        self.create_topic_clusters_visualization()
        self.create_sentiment_timeline()
        self.create_personality_radar()
        self.create_voice_characteristics_heatmap()
        self.create_communication_style_analysis()
        
        # Save all data to file
        self.save_visualization_data()
        
        print("\n✅ All visualizations completed!")
        print(f"📁 Saved as PNG files in '{self.visualizations_folder}' folder:")
        print("  • topic_clusters_visualization.png")
        print("  • sentiment_timeline.png")
        print("  • personality_radar.png")
        print("  • voice_heatmap.png")
        print("  • communication_styles.png")
        print("  • visualization_data.json")

def main():
    """Main function to generate visualizations"""
    generator = GenerateVisualizations()
    generator.generate_all_visualizations()

if __name__ == "__main__":
    main() 