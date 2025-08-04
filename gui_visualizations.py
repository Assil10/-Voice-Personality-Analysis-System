#!/usr/bin/env python3
"""
GUI Voice Personality Visualizations
===================================
Using tkinter for GUI with separate buttons for each visualization
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class GUIVoiceVisualizations:
    def __init__(self, analysis_folder="analysis_results"):
        self.analysis_folder = analysis_folder
        self.data = self.load_latest_analysis()
        self.visualization_data = self.load_visualization_data()
        self.current_figure = None
        self.canvas = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("🎭 Voice Personality Analysis Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Create GUI
        self.create_gui()
        
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

    def load_visualization_data(self):
        """Load visualization data from JSON file"""
        visualization_file = os.path.join('visualizations', 'visualization_data.json')
        if os.path.exists(visualization_file):
            try:
                with open(visualization_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading visualization data: {e}")
                return {}
        else:
            print("⚠️  No visualization_data.json found in 'visualizations' folder. Run generate_visualizations.py first.")
            return {}

    def create_gui(self):
        """Create the GUI layout"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="🎭 Voice Personality Analysis Dashboard",
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for buttons
        left_panel = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Button style
        button_style = {
            'font': ('Arial', 12, 'bold'),
            'width': 20,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 3,
            'cursor': 'hand2'
        }
        
        # Visualization buttons
        tk.Label(left_panel, text="📊 Visualizations", font=("Arial", 14, "bold"), 
                bg='#ecf0f1', fg='#2c3e50').pack(pady=10)
        
        # Topic Clusters button
        topic_btn = tk.Button(
            left_panel,
            text="📊 Topic Clusters",
            command=self.show_topic_clusters,
            bg='#3498db',
            fg='white',
            **button_style
        )
        topic_btn.pack(pady=5)
        
        # Sentiment Timeline button
        sentiment_btn = tk.Button(
            left_panel,
            text="😊 Sentiment Timeline",
            command=self.show_sentiment_timeline,
            bg='#e74c3c',
            fg='white',
            **button_style
        )
        sentiment_btn.pack(pady=5)
        
        # Personality Radar button
        personality_btn = tk.Button(
            left_panel,
            text="👤 Personality Radar",
            command=self.show_personality_radar,
            bg='#9b59b6',
            fg='white',
            **button_style
        )
        personality_btn.pack(pady=5)
        
        # Voice Heatmap button
        voice_btn = tk.Button(
            left_panel,
            text="🎵 Voice Heatmap",
            command=self.show_voice_heatmap,
            bg='#f39c12',
            fg='white',
            **button_style
        )
        voice_btn.pack(pady=5)
        
        # Communication Styles button
        comm_btn = tk.Button(
            left_panel,
            text="💬 Communication Styles",
            command=self.show_communication_styles,
            bg='#1abc9c',
            fg='white',
            **button_style
        )
        comm_btn.pack(pady=5)
        
        # Show All button
        all_btn = tk.Button(
            left_panel,
            text="🎨 Show All",
            command=self.show_all_visualizations,
            bg='#34495e',
            fg='white',
            **button_style
        )
        all_btn.pack(pady=10)
        
        # Separator
        separator = ttk.Separator(left_panel, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)
        
        # Exit button (separate from navigation)
        exit_btn = tk.Button(
            left_panel,
            text="❌ Exit Application",
            command=self.exit_application,
            bg='#e74c3c',
            fg='white',
            **button_style
        )
        exit_btn.pack(pady=10)
        
        # Right panel for visualization display
        self.right_panel = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to display visualizations",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.status_label.pack(side=tk.BOTTOM, pady=5)

    def clear_display(self):
        """Clear the current visualization display"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None

    def display_figure(self, fig, title):
        """Display a matplotlib figure in the GUI"""
        self.clear_display()
        
        # Create canvas
        self.current_figure = fig
        self.canvas = FigureCanvasTkAgg(fig, self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Update status
        self.status_label.config(text=f"Displaying: {title}")

    def show_topic_clusters(self):
        """Show topic clusters visualization"""
        if 'topic_clusters' not in self.visualization_data:
            messagebox.showerror("Error", "No topic clusters data available. Run generate_visualizations.py first.")
            return
            
        data = self.visualization_data['topic_clusters']
        cluster_ids = data['cluster_ids']
        sizes = data['sizes']
        keywords = data['keywords']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of cluster sizes
        bars = ax1.bar(cluster_ids, sizes, color=sns.color_palette("husl", len(cluster_ids)))
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
        self.display_figure(fig, "Topic Clusters")
        
        # Show additional info in status
        info_text = f"📋 Clusters: {len(cluster_ids)} | Total Messages: {sum(sizes)}"
        self.status_label.config(text=info_text)

    def show_sentiment_timeline(self):
        """Show sentiment timeline visualization"""
        if 'sentiment' not in self.visualization_data:
            messagebox.showerror("Error", "No sentiment data available. Run generate_visualizations.py first.")
            return
            
        data = self.visualization_data['sentiment']
        labels = data['labels']
        sizes = data['sizes']
        colors = data['colors']
        total = data['total']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
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
        self.display_figure(fig, "Sentiment Analysis")
        
        # Show additional info in status
        info_text = f"😊 Total: {total} | Positive: {data['positive_percent']:.1f}% | Negative: {data['negative_percent']:.1f}% | Neutral: {data['neutral_percent']:.1f}%"
        self.status_label.config(text=info_text)

    def show_personality_radar(self):
        """Show personality radar chart"""
        if 'personality' not in self.visualization_data:
            messagebox.showerror("Error", "No personality data available. Run generate_visualizations.py first.")
            return
            
        data = self.visualization_data['personality']
        categories = data['categories']
        values = data['values']
        
        # Number of variables
        N = len(categories)
        
        # Create angles for each trait
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add the first value to the end to complete the circle
        values += values[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set title
        plt.title('Personality Traits Radar Chart', fontsize=16, fontweight='bold', pad=20)
        
        # Set y-axis limits
        ax.set_ylim(0, max(values) * 1.2)
        
        plt.tight_layout()
        self.display_figure(fig, "Personality Radar")

    def show_voice_heatmap(self):
        """Show voice characteristics heatmap"""
        if 'voice_characteristics' not in self.visualization_data:
            messagebox.showerror("Error", "No voice characteristics data available. Run generate_visualizations.py first.")
            return
            
        data = self.visualization_data['voice_characteristics']
        characteristics = data['characteristics']
        correlation_matrix = pd.DataFrame(data['correlation_matrix'])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Voice Characteristics Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.display_figure(fig, "Voice Heatmap")

    def show_communication_styles(self):
        """Show communication style analysis"""
        if 'communication_styles' not in self.visualization_data:
            messagebox.showerror("Error", "No communication styles data available. Run generate_visualizations.py first.")
            return
            
        data = self.visualization_data['communication_styles']
        styles = data['styles']
        counts = data['counts']
        colors = data['colors']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        bars = ax1.bar(styles, counts, color=colors)
        ax1.set_title('Communication Style Frequency', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Communication Style')
        ax1.set_ylabel('Frequency')
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
        self.display_figure(fig, "Communication Styles")

    def show_all_visualizations(self):
        """Show all visualizations in sequence"""
        messagebox.showinfo("Info", "Showing all visualizations in sequence...")
        
        # Show each visualization with a delay
        self.show_topic_clusters()
        self.root.after(2000, self.show_sentiment_timeline)
        self.root.after(4000, self.show_personality_radar)
        self.root.after(6000, self.show_voice_heatmap)
        self.root.after(8000, self.show_communication_styles)

    def exit_application(self):
        """Exit the application"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.quit()
            self.root.destroy()

    def run(self):
        """Run the GUI application"""
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Start the GUI
        self.root.mainloop()

def main():
    """Run the GUI visualizations"""
    app = GUIVoiceVisualizations()
    app.run()

if __name__ == "__main__":
    main() 