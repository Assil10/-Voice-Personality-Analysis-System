import whisper
import os
import json
import sys
import librosa
import numpy as np
import time

def load_model_fast():
    """Load the medium model for fast processing"""
    try:
        print("🚀 Loading Whisper medium model for fast processing...")
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

def load_audio_fast(file_path):
    """Fast audio loading with minimal preprocessing"""
    try:
        # Load audio with librosa - minimal processing for speed
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # Quick normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def main():
    print("⚡ Fast Voice Transcription System")
    print("=" * 50)
    print("🚀 Medium model - No Gemini API")
    print("🌍 Auto-detect language")
    print("⚡ Optimized for speed")
    print("=" * 50)
    
    # Setup
    input_folder = "audio_files"
    output_folder = "transcripts_fast"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get audio files
    wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    print(f"📁 Found {len(wav_files)} WAV files to transcribe")
    
    if not wav_files:
        print("❌ No WAV files found in audio_files folder")
        return
    
    # Load model once
    model = load_model_fast()
    
    # Process files sequentially
    print(f"\n🚀 Starting fast transcription...")
    start_time = time.time()
    
    successful = 0
    failed = 0
    
    for i, file in enumerate(wav_files, 1):
        try:
            print(f"\n📝 Processing file {i}/{len(wav_files)}: {file}")
            file_path = os.path.join(input_folder, file)
            
            # Load audio
            audio = load_audio_fast(file_path)
            if audio is None:
                print(f"❌ Failed to load audio: {file}")
                failed += 1
                continue
            
            # Fast transcription with auto-detect
            print("  🎯 Transcribing...")
            result = model.transcribe(
                audio,
                language=None,  # Auto-detect language
                task="transcribe",
                verbose=False,
                word_timestamps=False,  # Disable for speed
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            text = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            language_probability = result.get('language_probability', 0)
            
            # Calculate confidence
            segments = result.get('segments', [])
            avg_confidence = np.mean([seg.get('avg_logprob', 0) for seg in segments]) if segments else 0
            
            # Save result
            output_path = os.path.join(output_folder, file.replace('.wav', '.json'))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "file": file,
                    "text": text,
                    "detected_language": detected_language,
                    "language_probability": language_probability,
                    "segments": segments,
                    "avg_confidence": float(avg_confidence),
                    "transcription_quality": "high" if avg_confidence > -1.0 else "medium" if avg_confidence > -2.0 else "low",
                    "model_used": "medium",
                    "gemini_enhanced": False,
                    "processing_time": time.time()
                }, f, ensure_ascii=False, indent=2)
            
            successful += 1
            print(f"✅ Completed: {file}")
            print(f"  🌍 Language: {detected_language} (confidence: {language_probability:.2f})")
            print(f"  📊 Confidence: {avg_confidence:.2f}")
            print(f"  📝 Text: {text[:80]}...")
            
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
    print(f"\n🎉 Fast transcription completed!")
    print(f"📊 Results:")
    print(f"   • Total files: {len(wav_files)}")
    print(f"   • Successful: {successful}")
    print(f"   • Failed: {failed}")
    print(f"   • Total time: {total_time/60:.1f} minutes")
    print(f"   • Average time per file: {total_time/len(wav_files):.1f} seconds")
    print(f"   • Processing rate: {len(wav_files)/(total_time/60):.1f} files/minute")
    print(f"   • Gemini enhancement: ❌ Disabled (for speed)")
    
    # Estimate for 2000 files
    estimated_time_2000 = (2000 / len(wav_files)) * total_time / 3600
    print(f"\n💡 For 2000 files, estimated time: {estimated_time_2000:.1f} hours")
    print(f"📁 Results saved in '{output_folder}' folder")

if __name__ == "__main__":
    main() 