#!/usr/bin/env python3
"""
Simple script to run topic organization with configurable parameters
"""

from organize_by_topic import TopicOrganizer
import os

def main():
    print("🎯 Voice Message Topic Organizer")
    print("=" * 50)
    
    # Count available transcript files
    transcripts_folder = 'transcripts'
    transcript_files = [f for f in os.listdir(transcripts_folder) if f.endswith('.json')]
    num_files = len(transcript_files)
    
    # Calculate appropriate number of clusters (max 1/2 of files, min 2)
    n_clusters = max(2, min(num_files // 2, 8))
    
    # Configuration - you can modify these values
    config = {
        'method': 'kmeans',      # 'kmeans' or 'dbscan'
        'n_clusters': n_clusters,  # Number of clusters (for kmeans)
        'eps': 0.3,             # Epsilon for DBSCAN (if using dbscan)
        'min_samples': 3,       # Min samples for DBSCAN (if using dbscan)
        'transcripts_folder': 'transcripts',
        'output_folder': 'organized_topics'
    }
    
    print(f"📁 Input folder: {config['transcripts_folder']}")
    print(f"📁 Output folder: {config['output_folder']}")
    print(f"📄 Found {num_files} transcript files")
    print(f"🔧 Method: {config['method']}")
    print(f"📊 Number of clusters: {config['n_clusters']}")
    print()
    
    # Create organizer
    organizer = TopicOrganizer(
        transcripts_folder=config['transcripts_folder'],
        output_folder=config['output_folder']
    )
    
    # Run organization
    organizer.run_organization(
        method=config['method'],
        n_clusters=config['n_clusters']
    )
    
    print("\n✅ Topic organization completed!")
    print(f"📂 Check the '{config['output_folder']}' folder for results")
    print("\n📋 What you'll find:")
    print("  • Organized topic folders")
    print("  • cluster_visualization.png (visual overview)")
    print("  • organization_summary.json (detailed summary)")

if __name__ == "__main__":
    main() 