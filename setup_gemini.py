#!/usr/bin/env python3
"""
Setup script for Gemini API key configuration
"""

import os
import sys

def setup_gemini_api():
    print("🔧 Gemini API Setup")
    print("=" * 40)
    print()
    print("To use AI-powered summaries, you need to:")
    print("1. Get a free Gemini API key from: https://makersuite.google.com/app/apikey")
    print("2. Set it as an environment variable")
    print()
    
    # Check if key is already set
    if os.getenv('GEMINI_API_KEY'):
        print("✅ GEMINI_API_KEY is already set!")
        print(f"   Key: {os.getenv('GEMINI_API_KEY')[:10]}...")
        return True
    
    print("❌ GEMINI_API_KEY not found.")
    print()
    
    # Ask user to input the key
    print("Enter your Gemini API key (or press Enter to skip):")
    api_key = input("API Key: ").strip()
    
    if api_key:
        # Set environment variable for current session
        os.environ['GEMINI_API_KEY'] = api_key
        print()
        print("✅ API key set for current session!")
        print("   Note: This will only work for the current terminal session.")
        print("   For permanent setup, set the environment variable in your system.")
        return True
    else:
        print()
        print("⚠️  No API key provided. AI summaries will not be available.")
        return False

def test_gemini_connection():
    """Test the Gemini API connection"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ No API key available for testing.")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test
        response = model.generate_content("Say 'Hello, Gemini is working!'")
        print("✅ Gemini API test successful!")
        print(f"   Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Voice Message Topic Organizer - Gemini Setup")
    print("=" * 50)
    print()
    
    # Setup API key
    if setup_gemini_api():
        print()
        print("🧪 Testing Gemini API connection...")
        test_gemini_connection()
    
    print()
    print("📋 Next steps:")
    print("1. Run: python run_enhanced_organization.py")
    print("2. Your voice messages will be organized with AI summaries!")
    print()
    print("💡 For permanent setup, add this to your environment variables:")
    print("   GEMINI_API_KEY=your_api_key_here") 