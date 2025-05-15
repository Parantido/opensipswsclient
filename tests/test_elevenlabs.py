# Save as test_elevenlabs.py
try:
    import elevenlabs
    print("ElevenLabs imported successfully")
    print(f"Version: {elevenlabs.__version__}")
    
    # Try to set API key
    elevenlabs.api_key = "TEST_KEY"
    print("API key set successfully")
    
    # Try to access the generate function
    if hasattr(elevenlabs, "generate"):
        print("Generate function found")
    else:
        print("Generate function not found")
        
except Exception as e:
    print(f"Error importing ElevenLabs: {e}")
