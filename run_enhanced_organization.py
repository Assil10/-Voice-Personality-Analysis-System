#!/usr/bin/env python3
"""
Enhanced Voice Message Topic Organizer with AI Summaries and Date Extraction
"""

import os
from dotenv import load_dotenv
from enhanced_topic_organizer import EnhancedTopicOrganizer

# Load environment variables from .env file
load_dotenv()

def main():
    print("🎯 Enhanced Voice Message Topic Organizer")
    print("=" * 60)
    print("✨ Features:")
    print("  • AI-powered topic summaries")
    print("  • Date extraction from filenames")
    print("  • Chronological ordering within topics")
    print("  • Improved clustering accuracy")
    print("=" * 60)
    
    # Count available transcript files
    transcripts_folder = 'transcripts'
    transcript_files = [f for f in os.listdir(transcripts_folder) if f.endswith('.json')]
    num_files = len(transcript_files)
    
    # Calculate appropriate number of clusters
    n_clusters = max(2, min(num_files // 2, 6))
    
    # Configuration
    config = {
        'method': 'kmeans',
        'n_clusters': n_clusters,
        'transcripts_folder': 'transcripts',
        'output_folder': 'organized_topics_enhanced'
    }
    
    print(f"📁 Input folder: {config['transcripts_folder']}")
    print(f"📁 Output folder: {config['output_folder']}")
    print(f"📄 Found {num_files} transcript files")
    print(f"🔧 Method: {config['method']}")
    print(f"📊 Number of clusters: {config['n_clusters']}")
    print()
    
    # Check for Gemini API key
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  No GEMINI_API_KEY found. Set GEMINI_API_KEY environment variable for AI summaries.")
        print("   The system will use basic summaries instead.")
        print()
    else:
        print("✅ Gemini API key found! AI summaries will be generated.")
        print()
    
    # Create organizer
    organizer = EnhancedTopicOrganizer(
        transcripts_folder=config['transcripts_folder'],
        output_folder=config['output_folder']
    )
    
    # Run organization
    organizer.run_organization(
        method=config['method'],
        n_clusters=config['n_clusters']
    )
    
    print("\n✅ Enhanced topic organization completed!")
    print(f"📂 Check the '{config['output_folder']}' folder for results")
    print("\n📋 What you'll find:")
    print("  • AI-generated topic summaries")
    print("  • Chronologically ordered files")
    print("  • Date ranges for each cluster")
    print("  • Enhanced cluster visualization")
    print("  • Detailed organization summary")

if __name__ == "__main__":
    main() 