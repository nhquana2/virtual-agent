"""
Test script to verify Google Cloud TTS is working correctly.
Run with: uv run python scripts/test_google_tts.py
"""

from google.cloud import texttospeech


def test_google_tts():
    """Test Google Cloud TTS API directly."""
    print("Testing Google Cloud Text-to-Speech API...")
    
    try:
        # Initialize the client
        client = texttospeech.TextToSpeechClient()
        print("✓ Client initialized successfully")
        
        # List available voices for Vietnamese
        print("\nListing Vietnamese voices...")
        response = client.list_voices(language_code="vi-VN")
        print(f"Found {len(response.voices)} Vietnamese voices:")
        for voice in response.voices:
            print(f"  - {voice.name} ({voice.ssml_gender.name})")
        
        # Test synthesis with Vietnamese text
        print("\nTesting Vietnamese speech synthesis...")
        synthesis_input = texttospeech.SynthesisInput(text="Xin chào, tôi là trợ lý ảo.")
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="vi-VN",
            name="vi-VN-Standard-A",
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # Save to file
        output_path = "scripts/test_output.mp3"
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            print(f"✓ Audio saved to {output_path}")
            print(f"  Audio size: {len(response.audio_content)} bytes")
        
        print("\n✅ Google Cloud TTS is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Text-to-Speech API is enabled:")
        print("   gcloud services enable texttospeech.googleapis.com")
        print("2. Make sure you have Application Default Credentials:")
        print("   gcloud auth application-default login")
        print("3. Make sure you have a billing account linked to your project")


if __name__ == "__main__":
    test_google_tts()
