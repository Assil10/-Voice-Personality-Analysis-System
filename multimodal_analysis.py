#!/usr/bin/env python3
"""
Multi-Modal Voice Personality Analysis
=====================================
Combines Voice Characteristics + Text Sentiment + Topic Clustering
with both Gemini AI and Standalone Statistical Analysis
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import librosa
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import random
import re
from collections import Counter

class MultiModalVoiceAnalysis:
    def __init__(self, analysis_folder="analysis_results"):
        self.analysis_folder = analysis_folder
        self.transcript_folder = "transcripts"
        self.audio_folder = "audio_files"
        
        # Load environment variables
        load_dotenv()
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Initialize Gemini if API key is available
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_available = True
        else:
            self.gemini_available = False
            print("⚠️  Gemini API key not found. Will use standalone analysis only.")
        
        # Load Whisper model
        print("🎤 Loading Whisper model...")
        self.whisper_model = whisper.load_model("medium")
        
        # Analysis results
        self.voice_data = []
        self.sentiment_data = []
        self.topic_clusters = []
        self.combined_analysis = {}
        
    def load_transcripts(self):
        """Load all transcript files"""
        print("📄 Loading transcripts...")
        transcripts = []
        
        if not os.path.exists(self.transcript_folder):
            print(f"❌ Transcript folder '{self.transcript_folder}' not found")
            return []
            
        for filename in os.listdir(self.transcript_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(self.transcript_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Try enhanced_text first, then original_text, then text
                        text = data.get('enhanced_text', data.get('original_text', data.get('text', ''))).strip()
                        if text:
                                                    transcripts.append({
                            'filename': filename,
                            'text': text,
                            'audio_file': data.get('audio_file', data.get('file', '')),
                            'duration': data.get('duration', 0),
                            'language': data.get('language', 'ar')
                        })
                except Exception as e:
                    print(f"⚠️  Error loading {filename}: {e}")
                    
        print(f"✅ Loaded {len(transcripts)} transcripts")
        return transcripts

    def extract_voice_characteristics(self, audio_file_path):
        """Extract comprehensive voice characteristics"""
        try:
            # Load audio
            y, sr = librosa.load(audio_file_path, sr=None)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Apply gain for quiet audio
            if np.max(np.abs(y)) < 0.1:
                y = y * 10
            
            # Basic features
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Energy features
            rms_energy = librosa.feature.rms(y=y)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            pitch_mean = np.mean(pitch_values) if pitch_values else 0
            pitch_std = np.std(pitch_values) if pitch_values else 0
            
            # Speech rate estimation (words per second)
            # This is a simplified estimation
            speech_rate = len(y) / (sr * duration) * 100  # Simplified metric
            
            # Energy variation
            energy_variation = np.std(rms_energy)
            pitch_variation = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            features = {
                'duration': float(duration),
                'sample_rate': int(sr),
                'spectral_centroid': float(np.mean(spectral_centroids)),
                'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff': float(np.mean(spectral_rolloff)),
                'spectral_contrast': float(np.mean(spectral_contrast)),
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist(),
                'tempo': float(tempo),
                'rms_energy': float(np.mean(rms_energy)),
                'zero_crossing_rate': float(np.mean(zero_crossing_rate)),
                'pitch_mean': float(pitch_mean),
                'pitch_std': float(pitch_std),
                'speech_rate': float(speech_rate),
                'energy_variation': float(energy_variation),
                'pitch_variation': float(pitch_variation)
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting voice features from {audio_file_path}: {e}")
            return None

    def analyze_sentiment_statistical(self, text):
        """Statistical sentiment analysis without AI"""
        # Arabic sentiment dictionaries
        positive_words = [
            'ممتاز', 'رائع', 'جميل', 'سعيد', 'مفرح', 'مبهج', 'مشجع', 'مثير',
            'مفيد', 'مهم', 'ضروري', 'مطلوب', 'مقبول', 'مناسب', 'مريح', 'سهل',
            'ميسر', 'متاح', 'موجود', 'مضمون', 'مؤكد', 'صحيح', 'دقيق', 'مفصل',
            'واضح', 'مفهوم', 'حب', 'أحب', 'سعادة', 'فرح', 'مرتاح', 'مريح',
            'جيد', 'حسن', 'طيب', 'لطيف', 'عظيم', 'ممتاز', 'مدهش', 'مثير',
            'مشوق', 'ممتع', 'مسلي', 'مريح', 'هادئ', 'ساكن', 'مطمئن'
        ]
        
        negative_words = [
            'سيء', 'مشكلة', 'صعب', 'حزين', 'مؤلم', 'مزعج', 'مقلق', 'مخيف',
            'مخيب', 'مخجل', 'مخرب', 'مخيب', 'مخجل', 'مخرب', 'مخيب', 'مخجل',
            'مخرب', 'مخيب', 'مخجل', 'مخرب', 'مخيب', 'مخجل', 'مخرب', 'مخيب',
            'مخجل', 'مخرب', 'مخيب', 'مخجل', 'مخرب', 'مخيب', 'مخجل', 'مخرب',
            'كره', 'أكره', 'حزن', 'أحزن', 'ألم', 'مؤلم', 'صعب', 'مشكلة',
            'مقلق', 'مخيف', 'مخيب', 'مخجل', 'مخرب', 'مخيب', 'مخجل', 'مخرب'
        ]
        
        # Count words
        words = text.split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        # Debug: Print found words
        found_positive = [word for word in words if word in positive_words]
        found_negative = [word for word in words if word in negative_words]
        if found_positive or found_negative:
            print(f"🔍 Sentiment words found - Positive: {found_positive}, Negative: {found_negative}")
        
        if total_words == 0:
            return {'score': 0, 'state': 'neutral', 'confidence': 0}
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_words
        
        # Determine emotional state
        if sentiment_score > 0.1:
            emotional_state = 'positive'
        elif sentiment_score < -0.1:
            emotional_state = 'negative'
        else:
            emotional_state = 'neutral'
        
        # Calculate confidence
        confidence = min(abs(sentiment_score) * 10, 1.0)
        
        return {
            'score': float(sentiment_score),
            'state': emotional_state,
            'confidence': float(confidence),
            'positive_words': positive_count,
            'negative_words': negative_count,
            'total_words': total_words
        }

    def analyze_personality_traits(self, text):
        """Analyze personality traits from text patterns"""
        traits = {}
        
        # Extroversion indicators
        extroversion_patterns = [
            r'أنا', r'نحن', r'نفعل', r'نذهب', r'نتحدث', r'نلتقي', r'نشارك',
            r'اجتماع', r'أصدقاء', r'عائلة', r'ناس', r'مجموعة', r'فريق'
        ]
        extroversion_count = sum(len(re.findall(pattern, text)) for pattern in extroversion_patterns)
        traits['extroversion'] = {
            'score': min(extroversion_count, 20),
            'indicators': extroversion_count,
            'description': 'Social interaction and group activities'
        }
        
        # Introversion indicators
        introversion_patterns = [
            r'أفكر', r'أقرأ', r'أدرس', r'أعمل', r'أفهم', r'أحلل', r'أخطط',
            r'هدوء', r'سلام', r'راحة', r'خصوصية', r'وقت', r'نفسي', r'ذاتي'
        ]
        introversion_count = sum(len(re.findall(pattern, text)) for pattern in introversion_patterns)
        traits['introversion'] = {
            'score': min(introversion_count, 20),
            'indicators': introversion_count,
            'description': 'Reflection and solitary activities'
        }
        
        # Emotional indicators
        emotional_patterns = [
            r'أشعر', r'أحب', r'أكره', r'أحزن', r'أفرح', r'أغضب', r'أخاف',
            r'مشاعر', r'عواطف', r'قلب', r'روح', r'ألم', r'فرح', r'حزن'
        ]
        emotional_count = sum(len(re.findall(pattern, text)) for pattern in emotional_patterns)
        traits['emotional'] = {
            'score': min(emotional_count, 20),
            'indicators': emotional_count,
            'description': 'Emotional expression and sensitivity'
        }
        
        # Analytical indicators
        analytical_patterns = [
            r'لأن', r'لذلك', r'بالتالي', r'نتيجة', r'سبب', r'سبب', r'تحليل',
            r'منطق', r'عقل', r'تفكير', r'استنتاج', r'استدلال', r'برهان'
        ]
        analytical_count = sum(len(re.findall(pattern, text)) for pattern in analytical_patterns)
        traits['analytical'] = {
            'score': min(analytical_count, 20),
            'indicators': analytical_count,
            'description': 'Logical thinking and analysis'
        }
        
        return traits

    def analyze_communication_style(self, text):
        """Analyze communication style patterns"""
        styles = {}
        
        # Formal vs Informal
        formal_patterns = [r'أرجو', r'يرجى', r'من فضلك', r'شكراً', r'عذراً', r'آسف']
        informal_patterns = [r'يا', r'هلا', r'مرحبا', r'أهلا', r'أهلاً', r'مرحباً']
        
        formal_count = sum(len(re.findall(pattern, text)) for pattern in formal_patterns)
        informal_count = sum(len(re.findall(pattern, text)) for pattern in informal_patterns)
        
        styles['formal'] = {'count': formal_count, 'description': 'Polite and respectful language'}
        styles['informal'] = {'count': informal_count, 'description': 'Casual and friendly language'}
        
        # Direct vs Indirect
        direct_patterns = [r'أريد', r'أحتاج', r'أطلب', r'أوافق', r'أرفض', r'أقول']
        indirect_patterns = [r'ربما', r'قد', r'يمكن', r'أعتقد', r'أظن', r'أحسب']
        
        direct_count = sum(len(re.findall(pattern, text)) for pattern in direct_patterns)
        indirect_count = sum(len(re.findall(pattern, text)) for pattern in indirect_patterns)
        
        styles['direct'] = {'count': direct_count, 'description': 'Clear and straightforward communication'}
        styles['indirect'] = {'count': indirect_count, 'description': 'Hesitant and cautious communication'}
        
        # Empathetic vs Assertive
        empathetic_patterns = [r'أفهم', r'أشعر', r'أعرف', r'أدرك', r'أقدر', r'أحترم']
        assertive_patterns = [r'يجب', r'لازم', r'ضروري', r'مطلوب', r'مهم', r'أساسي']
        
        empathetic_count = sum(len(re.findall(pattern, text)) for pattern in empathetic_patterns)
        assertive_count = sum(len(re.findall(pattern, text)) for pattern in assertive_patterns)
        
        styles['empathetic'] = {'count': empathetic_count, 'description': 'Understanding and supportive'}
        styles['assertive'] = {'count': assertive_count, 'description': 'Confident and decisive'}
        
        return styles

    def cluster_topics(self, transcripts, n_clusters=5):
        """Cluster transcripts by topic using TF-IDF and K-means"""
        print("🗂️  Clustering topics...")
        
        # Extract texts
        texts = [t['text'] for t in transcripts if t['text'].strip()]
        
        if len(texts) < n_clusters:
            n_clusters = len(texts)
        
        if not texts:
            return []
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Analyze clusters
            clusters = []
            for i in range(n_clusters):
                cluster_texts = [texts[j] for j in range(len(texts)) if cluster_labels[j] == i]
                cluster_transcripts = [transcripts[j] for j in range(len(transcripts)) if cluster_labels[j] == i]
                
                # Get top words for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_words = [feature_names[idx] for idx in top_indices]
                
                clusters.append({
                    'cluster_id': i,
                    'size': len(cluster_texts),
                    'top_words': top_words,
                    'texts': cluster_texts,
                    'transcripts': cluster_transcripts
                })
            
            return clusters
            
        except Exception as e:
            print(f"❌ Error in topic clustering: {e}")
            return []

    def analyze_voice_personality(self, voice_features):
        """Analyze personality from voice characteristics"""
        personality = {}
        
        # Confidence analysis
        if voice_features['rms_energy'] > 0.05:
            personality['confidence'] = 'high'
        elif voice_features['rms_energy'] > 0.03:
            personality['confidence'] = 'medium'
        else:
            personality['confidence'] = 'low'
        
        # Emotional expressiveness
        if voice_features['pitch_variation'] > 0.2:
            personality['expressiveness'] = 'high'
        elif voice_features['pitch_variation'] > 0.1:
            personality['expressiveness'] = 'medium'
        else:
            personality['expressiveness'] = 'low'
        
        # Communication pace
        if voice_features['speech_rate'] > 150:
            personality['pace'] = 'fast'
        elif voice_features['speech_rate'] > 100:
            personality['pace'] = 'medium'
        else:
            personality['pace'] = 'slow'
        
        # Voice stability
        if voice_features['energy_variation'] < 0.05:
            personality['stability'] = 'high'
        elif voice_features['energy_variation'] < 0.1:
            personality['stability'] = 'medium'
        else:
            personality['stability'] = 'low'
        
        return personality

    def get_custom_prompt(self):
        """Get custom prompt from user input"""
        print("\n🤖 CUSTOM PROMPT FOR GEMINI AI ANALYSIS")
        print("=" * 50)
        print("You can now customize the prompt for better analysis results.")
        print("Leave empty to use the default prompt, or type your custom prompt below.")
        print("\nCurrent default prompt focuses on:")
        print("• Voice characteristics analysis")
        print("• Sentiment patterns")
        print("• Topic clustering analysis")
        print("• Integrated personality profile")
        print("• Psychological insights")
        print("• Practical recommendations")
        print("\n" + "-" * 50)
        
        custom_prompt = input("Enter your custom prompt (or press Enter for default): ").strip()
        
        if not custom_prompt:
            print("✅ Using default prompt")
            return None
        else:
            print("✅ Using custom prompt")
            return custom_prompt

    def create_gemini_analysis(self, voice_data, sentiment_data, topic_clusters, custom_prompt=None):
        """Create comprehensive analysis using Gemini AI"""
        if not self.gemini_available:
            return None
        
        print("🤖 Creating Gemini AI analysis...")
        
        # Prepare data for Gemini
        analysis_data = {
            'voice_characteristics': voice_data,
            'sentiment_patterns': sentiment_data,
            'topic_clusters': topic_clusters
        }
        
        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            prompt = f"""
{custom_prompt}

Data: {json.dumps(analysis_data, ensure_ascii=False, indent=2)}

Provide a comprehensive, professional analysis based on your custom prompt.
            """
        else:
            prompt = f"""
        You are a professional psychologist and communication expert. Analyze this multi-modal voice personality data and provide:

        **VOICE CHARACTERISTICS ANALYSIS:**
        - Confidence level and communication style
        - Emotional expressiveness and voice patterns
        - Speaking pace and energy patterns
        - Voice stability and consistency

        **SENTIMENT ANALYSIS:**
        - Overall emotional state and patterns
        - Positive/negative/neutral distribution
        - Emotional stability and consistency
        - Communication mood and tone

        **TOPIC CLUSTERING ANALYSIS:**
        - Main conversation themes and interests
        - Topic diversity and focus areas
        - Professional vs personal balance
        - Social interaction patterns

        **INTEGRATED PERSONALITY PROFILE:**
        - Overall personality type and characteristics
        - Communication strengths and areas for improvement
        - Social interaction style and preferences
        - Professional and personal development recommendations

        **PSYCHOLOGICAL INSIGHTS:**
        - Emotional intelligence assessment
        - Social skills and relationship patterns
        - Stress management and coping mechanisms
        - Leadership and teamwork potential

        **PRACTICAL RECOMMENDATIONS:**
        - Communication improvement strategies
        - Career path suggestions
        - Personal development opportunities
        - Relationship and social interaction advice

        Data: {json.dumps(analysis_data, ensure_ascii=False, indent=2)}

        Provide a comprehensive, professional analysis in a structured format with clear sections and actionable insights.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return None

    def create_standalone_psychological_analysis(self, voice_data, sentiment_data, topic_clusters):
        """Create comprehensive psychological analysis without AI"""
        print("🧠 Creating standalone psychological analysis...")
        
        # Analyze voice characteristics
        voice_analysis = self.analyze_voice_psychology(voice_data)
        
        # Analyze sentiment patterns
        sentiment_analysis = self.analyze_sentiment_psychology(sentiment_data)
        
        # Analyze topic psychology
        topic_analysis = self.analyze_topic_psychology(topic_clusters)
        
        # Create integrated personality profile
        personality_profile = self.create_integrated_personality_profile(
            voice_analysis, sentiment_analysis, topic_analysis
        )
        
        # Generate psychological insights
        psychological_insights = self.generate_psychological_insights(
            voice_analysis, sentiment_analysis, topic_analysis
        )
        
        # Generate practical recommendations
        recommendations = self.generate_practical_recommendations(
            voice_analysis, sentiment_analysis, topic_analysis
        )
        
        analysis = f"""
STANDALONE PSYCHOLOGICAL ANALYSIS - MULTI-MODAL VOICE PERSONALITY
================================================================

VOICE CHARACTERISTICS PSYCHOLOGICAL ANALYSIS:
--------------------------------------------
{voice_analysis}

SENTIMENT PATTERNS PSYCHOLOGICAL ANALYSIS:
-----------------------------------------
{sentiment_analysis}

TOPIC CLUSTERING PSYCHOLOGICAL ANALYSIS:
---------------------------------------
{topic_analysis}

INTEGRATED PERSONALITY PROFILE:
------------------------------
{personality_profile}

PSYCHOLOGICAL INSIGHTS:
----------------------
{psychological_insights}

PRACTICAL RECOMMENDATIONS:
-------------------------
{recommendations}

ANALYSIS METHODOLOGY:
--------------------
This analysis was performed using advanced statistical methods and psychological frameworks:
- Voice Analysis: Acoustic feature extraction and pattern recognition
- Sentiment Analysis: Arabic language sentiment dictionaries and statistical scoring
- Topic Analysis: TF-IDF vectorization and K-means clustering
- Personality Assessment: Multi-modal integration of voice, text, and content patterns
- Psychological Framework: Based on established personality psychology principles

Note: This analysis provides comprehensive insights without requiring external AI services.
        """
        
        return analysis

    def analyze_voice_psychology(self, voice_data):
        """Analyze psychological aspects of voice characteristics"""
        if not voice_data:
            return "No voice data available for analysis."
        
        # Calculate averages
        avg_energy = np.mean([v['rms_energy'] for v in voice_data])
        avg_pitch = np.mean([v['pitch_mean'] for v in voice_data])
        avg_tempo = np.mean([v['tempo'] for v in voice_data])
        avg_pitch_var = np.mean([v['pitch_variation'] for v in voice_data])
        avg_energy_var = np.mean([v['energy_variation'] for v in voice_data])
        
        analysis = f"""
CONFIDENCE LEVEL ASSESSMENT:
- Average Energy: {avg_energy:.3f} ({'High' if avg_energy > 0.05 else 'Medium' if avg_energy > 0.03 else 'Low'} confidence)
- Energy Variation: {avg_energy_var:.3f} ({'Stable' if avg_energy_var < 0.05 else 'Variable' if avg_energy_var < 0.1 else 'Unstable'} voice)
- Psychological Interpretation: {'High confidence and assertiveness' if avg_energy > 0.05 and avg_energy_var < 0.05 else 'Moderate confidence with some variability' if avg_energy > 0.03 else 'Lower confidence, may need support'}

EMOTIONAL EXPRESSIVENESS:
- Average Pitch: {avg_pitch:.0f} Hz ({'High' if avg_pitch > 1200 else 'Medium' if avg_pitch > 800 else 'Low'} pitch)
- Pitch Variation: {avg_pitch_var:.3f} ({'Highly expressive' if avg_pitch_var > 0.2 else 'Moderately expressive' if avg_pitch_var > 0.1 else 'Less expressive'})
- Psychological Interpretation: {'High emotional expressiveness and engagement' if avg_pitch_var > 0.2 else 'Balanced emotional expression' if avg_pitch_var > 0.1 else 'Reserved emotional expression'}

COMMUNICATION PACE:
- Average Tempo: {avg_tempo:.0f} BPM ({'Fast' if avg_tempo > 150 else 'Medium' if avg_tempo > 100 else 'Slow'} speaker)
- Psychological Interpretation: {'Energetic and enthusiastic communicator' if avg_tempo > 150 else 'Balanced and measured speaker' if avg_tempo > 100 else 'Thoughtful and deliberate communicator'}

VOICE STABILITY ANALYSIS:
- Energy Stability: {'High' if avg_energy_var < 0.05 else 'Medium' if avg_energy_var < 0.1 else 'Low'}
- Pitch Stability: {'High' if avg_pitch_var < 0.1 else 'Medium' if avg_pitch_var < 0.2 else 'Low'}
- Psychological Interpretation: {'Consistent and reliable communication style' if avg_energy_var < 0.05 and avg_pitch_var < 0.1 else 'Variable but engaging communication' if avg_energy_var < 0.1 else 'Dynamic and expressive communication'}
        """
        
        return analysis

    def analyze_sentiment_psychology(self, sentiment_data):
        """Analyze psychological aspects of sentiment patterns"""
        if not sentiment_data:
            return "No sentiment data available for analysis."
        
        # Calculate statistics
        positive_count = sum(1 for s in sentiment_data if s['sentiment']['state'] == 'positive')
        negative_count = sum(1 for s in sentiment_data if s['sentiment']['state'] == 'negative')
        neutral_count = sum(1 for s in sentiment_data if s['sentiment']['state'] == 'neutral')
        total_messages = len(sentiment_data)
        
        avg_sentiment = np.mean([s['sentiment']['score'] for s in sentiment_data])
        sentiment_std = np.std([s['sentiment']['score'] for s in sentiment_data])
        
        # Personality traits analysis
        all_traits = {}
        for data in sentiment_data:
            for trait, info in data['personality_traits'].items():
                if trait not in all_traits:
                    all_traits[trait] = []
                all_traits[trait].append(info['score'])
        
        trait_averages = {trait: np.mean(scores) for trait, scores in all_traits.items()}
        
        analysis = f"""
EMOTIONAL STATE ANALYSIS:
- Positive Messages: {positive_count} ({positive_count/total_messages*100:.1f}%)
- Negative Messages: {negative_count} ({negative_count/total_messages*100:.1f}%)
- Neutral Messages: {neutral_count} ({neutral_count/total_messages*100:.1f}%)
- Average Sentiment Score: {avg_sentiment:.3f} ({'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'} overall)
- Sentiment Consistency: {sentiment_std:.3f} ({'Stable' if sentiment_std < 0.1 else 'Variable' if sentiment_std < 0.2 else 'Highly variable'} emotional state)

PSYCHOLOGICAL INTERPRETATION:
- Overall Mood: {'Optimistic and positive outlook' if avg_sentiment > 0.1 else 'Realistic and balanced perspective' if avg_sentiment > -0.1 else 'Challenging emotional state'}
- Emotional Stability: {'High emotional stability' if sentiment_std < 0.1 else 'Moderate emotional variability' if sentiment_std < 0.2 else 'High emotional variability'}
- Communication Tone: {'Enthusiastic and engaging' if positive_count > negative_count else 'Balanced and measured' if abs(positive_count - negative_count) < 5 else 'More reserved or challenging'}

PERSONALITY TRAITS ANALYSIS:
{chr(10).join([f"- {trait.capitalize()}: {score:.1f}/20 ({'High' if score > 15 else 'Medium' if score > 10 else 'Low'} indicator)" for trait, score in trait_averages.items()])}

PERSONALITY INTERPRETATION:
- {'High social engagement and extroversion' if trait_averages.get('extroversion', 0) > 15 else 'Balanced social interaction' if trait_averages.get('extroversion', 0) > 10 else 'More introverted tendencies'}
- {'Strong analytical thinking' if trait_averages.get('analytical', 0) > 15 else 'Balanced analytical approach' if trait_averages.get('analytical', 0) > 10 else 'More intuitive approach'}
- {'High emotional sensitivity' if trait_averages.get('emotional', 0) > 15 else 'Balanced emotional expression' if trait_averages.get('emotional', 0) > 10 else 'More reserved emotional expression'}
        """
        
        return analysis

    def analyze_topic_psychology(self, topic_clusters):
        """Analyze psychological aspects of topic clustering"""
        if not topic_clusters:
            return "No topic data available for analysis."
        
        # Analyze topic distribution
        total_messages = sum(cluster['size'] for cluster in topic_clusters)
        largest_cluster = max(cluster['size'] for cluster in topic_clusters)
        topic_diversity = len(set([word for cluster in topic_clusters for word in cluster['top_words']]))
        
        # Analyze topic themes
        all_words = []
        for cluster in topic_clusters:
            all_words.extend(cluster['top_words'])
        
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(10)
        
        analysis = f"""
TOPIC DIVERSITY ANALYSIS:
- Total Topic Clusters: {len(topic_clusters)}
- Largest Cluster: {largest_cluster} messages ({largest_cluster/total_messages*100:.1f}% of total)
- Topic Diversity: {topic_diversity} unique words
- Cluster Distribution: {'Focused' if largest_cluster/total_messages > 0.4 else 'Balanced' if largest_cluster/total_messages > 0.25 else 'Diverse'} topic distribution

PSYCHOLOGICAL INTERPRETATION:
- Topic Focus: {'High focus on specific interests' if largest_cluster/total_messages > 0.4 else 'Balanced interests across topics' if largest_cluster/total_messages > 0.25 else 'Diverse interests and curiosity'}
- Communication Breadth: {'Specialized communicator' if topic_diversity < 50 else 'Versatile communicator' if topic_diversity < 100 else 'Highly versatile communicator'}
- Social Interaction Style: {'Deep engagement in specific areas' if largest_cluster/total_messages > 0.4 else 'Balanced social interaction' if largest_cluster/total_messages > 0.25 else 'Broad social engagement'}

MOST COMMON THEMES:
{chr(10).join([f"- {word}: {count} occurrences" for word, count in most_common[:5]])}

TOPIC PSYCHOLOGY:
- Primary Interests: {'Professional/Work-focused' if any('work' in word or 'job' in word for word, _ in most_common[:3]) else 'Personal/Relationship-focused' if any('family' in word or 'friend' in word for word, _ in most_common[:3]) else 'General/Conversational'}
- Communication Depth: {'Deep and focused' if largest_cluster/total_messages > 0.4 else 'Balanced depth and breadth' if largest_cluster/total_messages > 0.25 else 'Broad and exploratory'}
- Social Engagement: {'Specialized social circles' if topic_diversity < 50 else 'Diverse social engagement' if topic_diversity < 100 else 'Very broad social engagement'}
        """
        
        return analysis

    def create_integrated_personality_profile(self, voice_analysis, sentiment_analysis, topic_analysis):
        """Create integrated personality profile from all analyses"""
        profile = f"""
INTEGRATED PERSONALITY PROFILE:
==============================

OVERALL PERSONALITY TYPE:
Based on the multi-modal analysis, this individual demonstrates:

COMMUNICATION STYLE:
- Voice Characteristics: {'Confident and expressive' if 'High confidence' in voice_analysis else 'Balanced and measured' if 'Medium confidence' in voice_analysis else 'Reserved and thoughtful'}
- Sentiment Patterns: {'Positive and optimistic' if 'Positive overall' in sentiment_analysis else 'Balanced and realistic' if 'Neutral overall' in sentiment_analysis else 'Challenging or reserved'}
- Topic Engagement: {'Focused and specialized' if 'Focused' in topic_analysis else 'Balanced and versatile' if 'Balanced' in topic_analysis else 'Diverse and exploratory'}

SOCIAL INTERACTION PATTERNS:
- Social Engagement: {'High social engagement' if 'High social engagement' in sentiment_analysis else 'Balanced social interaction' if 'Balanced' in sentiment_analysis else 'More reserved social style'}
- Communication Depth: {'Deep and focused' if 'Deep engagement' in topic_analysis else 'Balanced depth and breadth' if 'Balanced' in topic_analysis else 'Broad and exploratory'}
- Emotional Expression: {'Highly expressive' if 'Highly expressive' in voice_analysis else 'Moderately expressive' if 'Moderately expressive' in voice_analysis else 'Reserved emotional expression'}

PROFESSIONAL CHARACTERISTICS:
- Leadership Potential: {'High leadership potential' if 'High confidence' in voice_analysis and 'High social engagement' in sentiment_analysis else 'Moderate leadership potential' if 'Medium confidence' in voice_analysis else 'Supportive team member'}
- Communication Effectiveness: {'Highly effective communicator' if 'High confidence' in voice_analysis and 'Positive' in sentiment_analysis else 'Effective communicator' if 'Medium confidence' in voice_analysis else 'Developing communication skills'}
- Adaptability: {'Highly adaptable' if 'Diverse' in topic_analysis else 'Moderately adaptable' if 'Balanced' in topic_analysis else 'Specialized focus'}

PERSONAL DEVELOPMENT AREAS:
- Strengths: {'Confidence and expressiveness' if 'High confidence' in voice_analysis else 'Balanced communication' if 'Medium confidence' in voice_analysis else 'Thoughtful approach'}
- Growth Opportunities: {'Specialization' if 'Diverse' in topic_analysis else 'Breadth of engagement' if 'Focused' in topic_analysis else 'Confidence building'}
- Communication Enhancement: {'Maintain current strengths' if 'High confidence' in voice_analysis else 'Build confidence and expressiveness' if 'Low confidence' in voice_analysis else 'Balance expression and thoughtfulness'}
        """
        
        return profile

    def generate_psychological_insights(self, voice_analysis, sentiment_analysis, topic_analysis):
        """Generate psychological insights from all analyses"""
        insights = f"""
PSYCHOLOGICAL INSIGHTS:
======================

EMOTIONAL INTELLIGENCE ASSESSMENT:
- Self-Awareness: {'High' if 'Stable' in voice_analysis and 'Stable' in sentiment_analysis else 'Moderate' if 'Variable' in voice_analysis or 'Variable' in sentiment_analysis else 'Developing'}
- Emotional Expression: {'Highly expressive' if 'Highly expressive' in voice_analysis else 'Moderately expressive' if 'Moderately expressive' in voice_analysis else 'Reserved'}
- Emotional Regulation: {'Strong' if 'Stable' in sentiment_analysis else 'Moderate' if 'Variable' in sentiment_analysis else 'Developing'}

SOCIAL SKILLS AND RELATIONSHIP PATTERNS:
- Social Engagement: {'High' if 'High social engagement' in sentiment_analysis else 'Moderate' if 'Balanced' in sentiment_analysis else 'Lower'}
- Communication Adaptability: {'High' if 'Diverse' in topic_analysis else 'Moderate' if 'Balanced' in topic_analysis else 'Specialized'}
- Relationship Depth: {'Deep and focused' if 'Focused' in topic_analysis else 'Balanced' if 'Balanced' in topic_analysis else 'Broad and varied'}

STRESS MANAGEMENT AND COPING MECHANISMS:
- Stress Response: {'Resilient' if 'Stable' in voice_analysis and 'Stable' in sentiment_analysis else 'Variable' if 'Variable' in voice_analysis or 'Variable' in sentiment_analysis else 'May need support'}
- Coping Style: {'Active and expressive' if 'Highly expressive' in voice_analysis else 'Balanced' if 'Moderately expressive' in voice_analysis else 'Reserved and internal'}
- Support Seeking: {'Likely to seek support' if 'High social engagement' in sentiment_analysis else 'Balanced support seeking' if 'Balanced' in sentiment_analysis else 'May prefer independent coping'}

LEADERSHIP AND TEAMWORK POTENTIAL:
- Leadership Style: {'Confident and engaging' if 'High confidence' in voice_analysis else 'Balanced and supportive' if 'Medium confidence' in voice_analysis else 'Supportive and thoughtful'}
- Team Collaboration: {'Highly collaborative' if 'High social engagement' in sentiment_analysis else 'Balanced collaboration' if 'Balanced' in sentiment_analysis else 'Specialized collaboration'}
- Communication Leadership: {'Inspirational communicator' if 'Highly expressive' in voice_analysis else 'Effective communicator' if 'Moderately expressive' in voice_analysis else 'Thoughtful communicator'}
        """
        
        return insights

    def generate_practical_recommendations(self, voice_analysis, sentiment_analysis, topic_analysis):
        """Generate practical recommendations from all analyses"""
        recommendations = f"""
PRACTICAL RECOMMENDATIONS:
=========================

COMMUNICATION IMPROVEMENT STRATEGIES:
- Voice Enhancement: {'Maintain confident speaking style' if 'High confidence' in voice_analysis else 'Practice voice projection and energy' if 'Low confidence' in voice_analysis else 'Balance confidence and thoughtfulness'}
- Emotional Expression: {'Continue expressive communication' if 'Highly expressive' in voice_analysis else 'Practice emotional expression' if 'Reserved' in voice_analysis else 'Develop balanced expression'}
- Pace and Timing: {'Maintain engaging pace' if 'Fast' in voice_analysis else 'Consider varying pace for emphasis' if 'Slow' in voice_analysis else 'Balance pace with clarity'}

CAREER PATH SUGGESTIONS:
- Leadership Roles: {'Excellent fit for leadership' if 'High confidence' in voice_analysis and 'High social engagement' in sentiment_analysis else 'Good potential for leadership' if 'Medium confidence' in voice_analysis else 'Supportive leadership style'}
- Communication Roles: {'Perfect for public speaking' if 'Highly expressive' in voice_analysis else 'Good for team communication' if 'Moderately expressive' in voice_analysis else 'Excellent for written communication'}
- Specialized vs Generalist: {'Specialized roles' if 'Focused' in topic_analysis else 'Generalist roles' if 'Diverse' in topic_analysis else 'Balanced roles'}

PERSONAL DEVELOPMENT OPPORTUNITIES:
- Confidence Building: {'Maintain current confidence' if 'High confidence' in voice_analysis else 'Build confidence through practice' if 'Low confidence' in voice_analysis else 'Develop balanced confidence'}
- Social Skills: {'Expand social networks' if 'High social engagement' in sentiment_analysis else 'Balance social engagement' if 'Balanced' in sentiment_analysis else 'Develop social comfort'}
- Topic Expertise: {'Deepen specialized knowledge' if 'Focused' in topic_analysis else 'Explore diverse interests' if 'Diverse' in topic_analysis else 'Balance depth and breadth'}

RELATIONSHIP AND SOCIAL INTERACTION ADVICE:
- Communication Style: {'Maintain expressive style' if 'Highly expressive' in voice_analysis else 'Develop more expression' if 'Reserved' in voice_analysis else 'Balance expression and listening'}
- Social Engagement: {'Continue active social engagement' if 'High social engagement' in sentiment_analysis else 'Balance social and personal time' if 'Balanced' in sentiment_analysis else 'Gradually increase social comfort'}
- Relationship Building: {'Focus on deep relationships' if 'Focused' in topic_analysis else 'Build diverse relationships' if 'Diverse' in topic_analysis else 'Balance relationship types'}

PROFESSIONAL DEVELOPMENT PLAN:
- Short-term Goals: {'Maintain current strengths' if 'High confidence' in voice_analysis else 'Build confidence and expressiveness' if 'Low confidence' in voice_analysis else 'Develop balanced communication'}
- Medium-term Goals: {'Expand leadership opportunities' if 'High confidence' in voice_analysis else 'Develop leadership skills' if 'Medium confidence' in voice_analysis else 'Build supportive leadership style'}
- Long-term Goals: {'Executive leadership potential' if 'High confidence' in voice_analysis and 'High social engagement' in sentiment_analysis else 'Management potential' if 'Medium confidence' in voice_analysis else 'Specialized expertise development'}
        """
        
        return recommendations

    def create_statistical_summary(self, voice_data, sentiment_data, topic_clusters):
        """Create comprehensive statistical summary"""
        print("📊 Creating statistical summary...")
        
        # Voice characteristics summary
        voice_summary = {
            'average_duration': np.mean([v['duration'] for v in voice_data]),
            'average_energy': np.mean([v['rms_energy'] for v in voice_data]),
            'average_pitch': np.mean([v['pitch_mean'] for v in voice_data]),
            'average_tempo': np.mean([v['tempo'] for v in voice_data]),
            'energy_variation': np.mean([v['energy_variation'] for v in voice_data]),
            'pitch_variation': np.mean([v['pitch_variation'] for v in voice_data])
        }
        
        # Sentiment summary
        sentiment_summary = {
            'positive_count': sum(1 for s in sentiment_data if s['sentiment']['state'] == 'positive'),
            'negative_count': sum(1 for s in sentiment_data if s['sentiment']['state'] == 'negative'),
            'neutral_count': sum(1 for s in sentiment_data if s['sentiment']['state'] == 'neutral'),
            'average_sentiment_score': np.mean([s['sentiment']['score'] for s in sentiment_data]),
            'sentiment_consistency': np.std([s['sentiment']['score'] for s in sentiment_data])
        }
        
        # Topic summary
        topic_summary = {
            'total_clusters': len(topic_clusters),
            'largest_cluster': max([c['size'] for c in topic_clusters]) if topic_clusters else 0,
            'topic_diversity': len(set([word for c in topic_clusters for word in c['top_words']])),
            'most_common_words': self.get_most_common_words(topic_clusters)
        }
        
        return {
            'voice_summary': voice_summary,
            'sentiment_summary': sentiment_summary,
            'topic_summary': topic_summary
        }

    def get_most_common_words(self, topic_clusters):
        """Get most common words across all clusters"""
        all_words = []
        for cluster in topic_clusters:
            all_words.extend(cluster['top_words'])
        
        word_counts = Counter(all_words)
        return word_counts.most_common(10)

    def run_complete_analysis(self):
        """Run the complete multi-modal analysis"""
        print("🎭 Starting Multi-Modal Voice Personality Analysis")
        print("=" * 60)
        
        # Load transcripts
        transcripts = self.load_transcripts()
        if not transcripts:
            print("❌ No transcripts found. Please run transcription first.")
            return
        
        # Extract voice characteristics
        print("🎵 Extracting voice characteristics...")
        for transcript in transcripts:
            audio_file = transcript.get('audio_file', '')
            if audio_file:
                # Construct full path to audio file
                audio_file_path = os.path.join(self.audio_folder, audio_file)
                print(f"🔍 Checking audio file: {audio_file_path}")
                if os.path.exists(audio_file_path):
                    print(f"✅ Audio file exists: {audio_file_path}")
                    voice_features = self.extract_voice_characteristics(audio_file_path)
                    if voice_features:
                        voice_features['filename'] = transcript['filename']
                        voice_features['personality'] = self.analyze_voice_personality(voice_features)
                        self.voice_data.append(voice_features)
                        print(f"✅ Voice features extracted for {transcript['filename']}")
                    else:
                        print(f"❌ Failed to extract voice features for {transcript['filename']}")
                else:
                    print(f"❌ Audio file not found: {audio_file_path}")
            else:
                print(f"❌ No audio file specified for {transcript['filename']}")
        
        # Analyze sentiment and personality
        print("😊 Analyzing sentiment and personality...")
        for transcript in transcripts:
            sentiment = self.analyze_sentiment_statistical(transcript['text'])
            personality_traits = self.analyze_personality_traits(transcript['text'])
            communication_styles = self.analyze_communication_style(transcript['text'])
            
            self.sentiment_data.append({
                'filename': transcript['filename'],
                'sentiment': sentiment,
                'personality_traits': personality_traits,
                'communication_styles': communication_styles,
                'text': transcript['text'][:200] + "..." if len(transcript['text']) > 200 else transcript['text']
            })
        
        # Cluster topics
        self.topic_clusters = self.cluster_topics(transcripts)
        
        # Create statistical summary
        statistical_summary = self.create_statistical_summary(
            self.voice_data, self.sentiment_data, self.topic_clusters
        )
        
        # Create analysis (Gemini if available, standalone if not)
        gemini_analysis = None
        standalone_analysis = None
        
        if self.gemini_available:
            custom_prompt = self.get_custom_prompt()
            gemini_analysis = self.create_gemini_analysis(
                self.voice_data, self.sentiment_data, self.topic_clusters, custom_prompt
            )
        else:
            standalone_analysis = self.create_standalone_psychological_analysis(
                self.voice_data, self.sentiment_data, self.topic_clusters
            )
        
        # Combine all results
        self.combined_analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_transcripts': len(transcripts),
            'voice_data': self.voice_data,
            'sentiment_data': self.sentiment_data,
            'topic_clusters': self.topic_clusters,
            'statistical_summary': statistical_summary,
            'gemini_analysis': gemini_analysis,
            'standalone_analysis': standalone_analysis,
            'custom_prompt_used': custom_prompt if self.gemini_available else None,
            'analysis_metadata': {
                'gemini_available': self.gemini_available,
                'voice_features_extracted': len(self.voice_data),
                'sentiment_analyzed': len(self.sentiment_data),
                'topics_clustered': len(self.topic_clusters)
            }
        }
        
        # Save results
        self.save_analysis_results()
        
        # Print summary
        self.print_analysis_summary()
        
        return self.combined_analysis

    def save_analysis_results(self):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(self.analysis_folder, f"multimodal_analysis_{timestamp}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Save main analysis data
        analysis_file = os.path.join(output_folder, "multimodal_analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self.combined_analysis, f, ensure_ascii=False, indent=2)
        
        # Save analysis separately if available
        if self.combined_analysis.get('gemini_analysis'):
            gemini_file = os.path.join(output_folder, "gemini_analysis.txt")
            with open(gemini_file, 'w', encoding='utf-8') as f:
                f.write(self.combined_analysis['gemini_analysis'])
        
        if self.combined_analysis.get('standalone_analysis'):
            standalone_file = os.path.join(output_folder, "standalone_analysis.txt")
            with open(standalone_file, 'w', encoding='utf-8') as f:
                f.write(self.combined_analysis['standalone_analysis'])
        
        # Save custom prompt if used
        if self.combined_analysis.get('custom_prompt_used'):
            prompt_file = os.path.join(output_folder, "custom_prompt.txt")
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(f"Custom Prompt Used:\n{self.combined_analysis['custom_prompt_used']}")
        
        # Save statistical summary
        summary_file = os.path.join(output_folder, "statistical_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self.format_statistical_summary())
        
        print(f"✅ Analysis results saved to: {output_folder}")

    def format_statistical_summary(self):
        """Format statistical summary for text output"""
        summary = self.combined_analysis['statistical_summary']
        
        report = f"""
MULTI-MODAL VOICE PERSONALITY ANALYSIS - STATISTICAL SUMMARY
===========================================================

VOICE CHARACTERISTICS SUMMARY:
-----------------------------
Average Duration: {summary['voice_summary']['average_duration']:.1f} seconds
Average Energy: {summary['voice_summary']['average_energy']:.3f}
Average Pitch: {summary['voice_summary']['average_pitch']:.0f} Hz
Average Tempo: {summary['voice_summary']['average_tempo']:.0f} BPM
Energy Variation: {summary['voice_summary']['energy_variation']:.3f}
Pitch Variation: {summary['voice_summary']['pitch_variation']:.3f}

SENTIMENT ANALYSIS SUMMARY:
--------------------------
Positive Messages: {summary['sentiment_summary']['positive_count']}
Negative Messages: {summary['sentiment_summary']['negative_count']}
Neutral Messages: {summary['sentiment_summary']['neutral_count']}
Average Sentiment Score: {summary['sentiment_summary']['average_sentiment_score']:.3f}
Sentiment Consistency: {summary['sentiment_summary']['sentiment_consistency']:.3f}

TOPIC CLUSTERING SUMMARY:
------------------------
Total Topic Clusters: {summary['topic_summary']['total_clusters']}
Largest Cluster Size: {summary['topic_summary']['largest_cluster']}
Topic Diversity: {summary['topic_summary']['topic_diversity']} unique words

Most Common Words:
{chr(10).join([f"  {word}: {count}" for word, count in summary['topic_summary']['most_common_words']])}

ANALYSIS METADATA:
-----------------
Total Transcripts Analyzed: {self.combined_analysis['analysis_metadata']['voice_features_extracted']}
Voice Features Extracted: {self.combined_analysis['analysis_metadata']['voice_features_extracted']}
Sentiment Analysis Performed: {self.combined_analysis['analysis_metadata']['sentiment_analyzed']}
Topics Clustered: {self.combined_analysis['analysis_metadata']['topics_clustered']}
Gemini AI Available: {self.combined_analysis['analysis_metadata']['gemini_available']}
        """
        
        return report

    def print_analysis_summary(self):
        """Print analysis summary to console"""
        print("\n" + "=" * 60)
        print("🎭 MULTI-MODAL ANALYSIS COMPLETE")
        print("=" * 60)
        
        summary = self.combined_analysis['statistical_summary']
        
        print(f"📊 Analysis Summary:")
        print(f"  • Transcripts analyzed: {self.combined_analysis['total_transcripts']}")
        print(f"  • Voice features extracted: {len(self.voice_data)}")
        print(f"  • Sentiment analyzed: {len(self.sentiment_data)}")
        print(f"  • Topics clustered: {len(self.topic_clusters)}")
        print(f"  • Gemini AI: {'✅ Available' if self.gemini_available else '❌ Not available'}")
        
        print(f"\n🎵 Voice Characteristics:")
        print(f"  • Average duration: {summary['voice_summary']['average_duration']:.1f}s")
        print(f"  • Average energy: {summary['voice_summary']['average_energy']:.3f}")
        print(f"  • Average pitch: {summary['voice_summary']['average_pitch']:.0f} Hz")
        print(f"  • Average tempo: {summary['voice_summary']['average_tempo']:.0f} BPM")
        
        print(f"\n😊 Sentiment Analysis:")
        print(f"  • Positive: {summary['sentiment_summary']['positive_count']}")
        print(f"  • Negative: {summary['sentiment_summary']['negative_count']}")
        print(f"  • Neutral: {summary['sentiment_summary']['neutral_count']}")
        print(f"  • Average score: {summary['sentiment_summary']['average_sentiment_score']:.3f}")
        
        print(f"\n🗂️  Topic Clustering:")
        print(f"  • Total clusters: {summary['topic_summary']['total_clusters']}")
        print(f"  • Largest cluster: {summary['topic_summary']['largest_cluster']} messages")
        print(f"  • Topic diversity: {summary['topic_summary']['topic_diversity']} unique words")
        
        if self.combined_analysis.get('gemini_analysis'):
            print(f"\n🤖 Gemini AI Analysis: ✅ Completed")
        elif self.combined_analysis.get('standalone_analysis'):
            print(f"\n🧠 Standalone Analysis: ✅ Completed")
        else:
            print(f"\n❌ No analysis completed")

def main():
    """Run the multi-modal analysis"""
    analyzer = MultiModalVoiceAnalysis()
    results = analyzer.run_complete_analysis()
    
    if results:
        print(f"\n✅ Multi-modal analysis completed successfully!")
        print(f"📁 Results saved in: analysis_results/")
    else:
        print(f"\n❌ Analysis failed. Please check your data and try again.")

if __name__ == "__main__":
    main() 