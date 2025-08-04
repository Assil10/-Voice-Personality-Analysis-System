import whisper
import os
import json
import sys
import librosa
import numpy as np
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_gemini():
    """Setup Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  No GEMINI_API_KEY found. Set GEMINI_API_KEY environment variable for enhanced processing.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini API configured successfully!")
        return model
    except Exception as e:
        print(f"❌ Error setting up Gemini: {e}")
        return None

def load_model_enhanced():
    """Load the medium model for better Arabic transcription"""
    try:
        print("🚀 Loading Whisper medium model for enhanced processing...")
        model = whisper.load_model("medium")
        print("✅ Medium model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading medium model: {e}")
        print("Falling back to small model...")
        try:
            model = whisper.load_model("small")
            print("✅ Small model loaded successfully!")
            return model
        except Exception as e2:
            print(f"Error loading small model: {e2}")
            sys.exit(1)

def load_audio_enhanced(file_path):
    """Enhanced audio loading with better preprocessing"""
    try:
        # Load audio with librosa
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # Enhanced normalization
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)
            
            # Apply noise reduction for better quality
            if np.max(np.abs(audio)) < 0.1:
                print("  ⚠️  Audio seems quiet, applying gain...")
                audio = audio * 2.0
        
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def enhance_transcription_with_gemini(text, gemini_model):
    """Use Gemini to enhance and clean up transcription with Tunisian dialect awareness"""
    if not gemini_model:
        return text
    
    try:
        prompt = f"""
        You are an expert in Tunisian Arabic dialect and transcription enhancement. 
        
        This text is in TUNISIAN ARABIC ACCENT - a unique dialect with:
        - Special pronunciation patterns (like "sh" → "s", "th" → "t", "dh" → "d")
        - French loanwords mixed in naturally
        - Unique expressions and idioms
        - Informal, conversational style
        - Code-switching between Arabic and French
        
        Please clean up and enhance this transcribed text. The text may contain:
        - Tunisian Arabic words/phrases with local pronunciation
        - French words/phrases naturally mixed in
        - Tunisian expressions and idioms
        - Transcription errors due to accent
        - Missing punctuation
        - Unclear words due to dialect
        
        Original text: "{text}"
        
        Please provide:
        1. Cleaned and corrected text in proper Tunisian Arabic
        2. Fix transcription errors while preserving Tunisian dialect
        3. Add proper punctuation where needed
        4. Keep the original meaning and Tunisian flavor intact
        5. Preserve any French words that are naturally used
        6. Understand the context and hidden meanings in Tunisian expressions
        
        Return only the cleaned text, nothing else.
        """
        
        response = gemini_model.generate_content(prompt)
        enhanced_text = response.text.strip()
        
        # Remove quotes if Gemini added them
        if enhanced_text.startswith('"') and enhanced_text.endswith('"'):
            enhanced_text = enhanced_text[1:-1]
        
        return enhanced_text
        
    except Exception as e:
        print(f"⚠️  Gemini enhancement failed: {e}")
        return text

def main():
    print("🎯 Enhanced Arabic/English Voice Transcription System")
    print("=" * 60)
    print("⚡ Medium model + Gemini API enhancement")
    print("🌍 Optimized for mixed Arabic/English content")
    print("🧹 Automatic cleanup and correction")
    print("=" * 60)
    
    # Setup
    input_folder = "audio_files"
    output_folder = "transcripts"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get audio files
    wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    print(f"📁 Found {len(wav_files)} WAV files to transcribe")
    
    if not wav_files:
        print("❌ No WAV files found in audio_files folder")
        return
    
    # Load models
    whisper_model = load_model_enhanced()
    gemini_model = setup_gemini()
    
    # Process files sequentially
    print(f"\n🚀 Starting enhanced transcription...")
    start_time = time.time()
    
    successful = 0
    failed = 0
    
    for i, file in enumerate(wav_files, 1):
        try:
            print(f"\n📝 Processing file {i}/{len(wav_files)}: {file}")
            file_path = os.path.join(input_folder, file)
            
            # Load audio
            audio = load_audio_enhanced(file_path)
            if audio is None:
                print(f"❌ Failed to load audio: {file}")
                failed += 1
                continue
            
            # Transcribe with enhanced settings
            print("  🎯 Transcribing with medium model...")
            result = whisper_model.transcribe(
                audio,
                language=None,  # Auto-detect for mixed content
                task="transcribe",
                verbose=False,
                word_timestamps=False,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            original_text = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            language_probability = result.get('language_probability', 0)
            
            # Enhance with Gemini
            if gemini_model:
                print("  🧠 Enhancing with Gemini API...")
                enhanced_text = enhance_transcription_with_gemini(original_text, gemini_model)
            else:
                enhanced_text = original_text
            
            # Calculate confidence
            segments = result.get('segments', [])
            avg_confidence = np.mean([seg.get('avg_logprob', 0) for seg in segments]) if segments else 0
            
            # Save result
            output_path = os.path.join(output_folder, file.replace('.wav', '.json'))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "file": file,
                    "original_text": original_text,
                    "enhanced_text": enhanced_text,
                    "detected_language": detected_language,
                    "language_probability": language_probability,
                    "segments": segments,
                    "avg_confidence": float(avg_confidence),
                    "transcription_quality": "high" if avg_confidence > -1.0 else "medium" if avg_confidence > -2.0 else "low",
                    "model_used": "medium",
                    "gemini_enhanced": gemini_model is not None,
                    "processing_time": time.time()
                }, f, ensure_ascii=False, indent=2)
            
            successful += 1
            print(f"✅ Completed: {file}")
            print(f"  🌍 Language: {detected_language} (confidence: {language_probability:.2f})")
            print(f"  📊 Confidence: {avg_confidence:.2f}")
            print(f"  📝 Original: {original_text[:80]}...")
            print(f"  ✨ Enhanced: {enhanced_text[:80]}...")
            
            # Progress update
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(wav_files) - i) / rate if rate > 0 else 0
            print(f"  📊 Progress: {i}/{len(wav_files)} ({rate:.1f} files/min, ETA: {eta/60:.1f} min)")
            
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
            failed += 1
            continue
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 Enhanced transcription completed!")
    print(f"📊 Results:")
    print(f"   • Total files: {len(wav_files)}")
    print(f"   • Successful: {successful}")
    print(f"   • Failed: {failed}")
    print(f"   • Total time: {total_time/60:.1f} minutes")
    print(f"   • Average time per file: {total_time/len(wav_files):.1f} seconds")
    print(f"   • Processing rate: {len(wav_files)/(total_time/60):.1f} files/minute")
    print(f"   • Gemini enhancement: {'✅ Enabled' if gemini_model else '❌ Disabled'}")
    
    # Estimate for 2000 files
    estimated_time_2000 = (2000 / len(wav_files)) * total_time / 3600
    print(f"\n💡 For 2000 files, estimated time: {estimated_time_2000:.1f} hours")
    print(f"📁 Results saved in '{output_folder}' folder")

if __name__ == "__main__":
    main() 