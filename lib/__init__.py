"""
OpenSIPS WebSocket Client with ElevenLabs TTS Integration

A Python library for interacting with OpenSIPS via WebSockets with 
ElevenLabs text-to-speech streaming capabilities and Deepgram speech recognition.
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .opensips_ws_lib import OpenSIPSClient, SIPMessage
from .elevenlabs_streamer import ElevenLabsStreamer, stream_tts_to_call
from .deepgram_transcriber import DeepgramTranscriber, transcribe_call
from .config_manager import ConfigManager, load_config, get_api_key

# For convenience, create a single entrypoint
def create_client(server_uri=None, username=None, password=None, domain=None,
                 config_path=None, debug_level=1, via_host=None, via_port=None):
    """
    Create an OpenSIPSClient instance with the given parameters or from config.

    Args:
        server_uri: OpenSIPS WebSocket server URI
        username: SIP username
        password: SIP password
        domain: SIP domain
        config_path: Path to config file
        debug_level: Debug level (1-9)
        via_host: Host to use in Via headers (to avoid using domains with underscores)
        via_port: Port to use in Via headers

    Returns:
        OpenSIPSClient instance
    """
    # Initialize configuration
    config = {}

    # If config path is provided, load it
    if config_path:
        loaded_config = load_config(config_path)

        # Use config values if not provided as arguments
        server_uri = server_uri or loaded_config.get("opensips", {}).get("server") or loaded_config.get("VOICEBOT_OPENSIPS_SERVER")
        username = username or loaded_config.get("opensips", {}).get("username") or loaded_config.get("VOICEBOT_OPENSIPS_USERNAME")
        password = password or loaded_config.get("opensips", {}).get("password") or loaded_config.get("VOICEBOT_OPENSIPS_PASSWORD")
        domain = domain or loaded_config.get("opensips", {}).get("domain") or loaded_config.get("VOICEBOT_OPENSIPS_DOMAIN")
        debug_level = debug_level or loaded_config.get("opensips", {}).get("debug_level") or loaded_config.get("VOICEBOT_OPENSIPS_DEBUG_LEVEL", 1)

        # Get via_host and via_port from config if not provided
        via_host = via_host or loaded_config.get("opensips", {}).get("via_host") or loaded_config.get("VOICEBOT_OPENSIPS_VIA_HOST", "127.0.0.1")
        via_port = via_port or loaded_config.get("opensips", {}).get("via_port") or loaded_config.get("VOICEBOT_OPENSIPS_VIA_PORT", "5060")

    # Store via_host and via_port in config
    config["via_host"] = via_host or "127.0.0.1"
    config["via_port"] = via_port or "5060"

    # Create and return client with config
    return OpenSIPSClient(
        server_uri=server_uri,
        username=username,
        password=password,
        domain=domain,
        debug_level=debug_level,
        config=config
    )
