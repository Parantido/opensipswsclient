"""
Advanced Voice Bot Example with LLM Integration
Demonstrates using OpenAI or Groq for dynamic conversation capabilities
"""

import asyncio
import argparse
import logging
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

## setting parent directory path
#sys.path.append('../')
#
#from opensips_ws_lib import OpenSIPSClient
#from elevenlabs_streamer import ElevenLabsStreamer
#from deepgram_transcriber import DeepgramTranscriber
#from config_manager import load_config, get_api_key

# Add the parent directory (where lib/ is located) to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import directly from the lib package (using __init__.py)
from lib import OpenSIPSClient, ElevenLabsStreamer, DeepgramTranscriber
from lib import load_config, get_api_key, create_client

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System prompt for the LLM
SYSTEM_PROMPT = """
You are an AI voice assistant for a medical clinic, speaking to a patient over the phone.
Your main responsibilities are:
1. Confirming appointments
2. Answering basic questions about the clinic
3. Taking notes about patient concerns

Keep your responses concise and natural for spoken conversation (1-3 sentences per turn).
The patient can hear you in real-time, so make your responses suitable for speech.

IMPORTANT CONVERSATION RULES:
- Introduce yourself at the beginning of the call
- Ask one question at a time and wait for a response
- Speak naturally as if on a phone call
- Keep responses brief (max 30 words per response)
- Don't use bullet points, formatting, or anything visual
- Don't remind the patient they're talking to an AI
- End the conversation politely when the patient's needs are met

Available information:
- Patient name: {patient_name}
- Appointment: {appointment_date} at {appointment_time}
- Doctor: {doctor_name}
- Clinic address: 123 Medical Center Drive
- Parking: Free parking available in Lot B
"""

class LLMVoiceBot:
    """
    Advanced Voice Bot that uses LLM for dynamic conversation
    Integrates ElevenLabs TTS and Deepgram STT with OpenAI or Groq
    """
    
    def __init__(self, 
                opensips_client, 
                call_id: str,
                elevenlabs_api_key: str,
                elevenlabs_voice_id: str,
                deepgram_api_key: str,
                llm_provider: str = "openai",
                openai_api_key: Optional[str] = None,
                groq_api_key: Optional[str] = None,
                llm_model: str = "gpt-3.5-turbo",
                context_data: Dict[str, str] = None,
                debug: bool = False):
        """
        Initialize the LLM-powered voice bot
        
        Args:
            opensips_client: OpenSIPSClient instance
            call_id: Active call ID
            elevenlabs_api_key: ElevenLabs API key
            elevenlabs_voice_id: ElevenLabs voice ID to use
            deepgram_api_key: Deepgram API key
            llm_provider: LLM provider to use ("openai" or "groq")
            openai_api_key: OpenAI API key (required if provider is "openai")
            groq_api_key: Groq API key (required if provider is "groq")
            llm_model: LLM model to use
            context_data: Data to fill placeholders in messages
            debug: Enable debug mode
        """
        self.client = opensips_client
        self.call_id = call_id
        self.elevenlabs_api_key = elevenlabs_api_key
        self.voice_id = elevenlabs_voice_id
        self.deepgram_api_key = deepgram_api_key
        self.llm_provider = llm_provider
        self.openai_api_key = openai_api_key
        self.groq_api_key = groq_api_key
        self.llm_model = llm_model
        self.context_data = context_data or {}
        self.debug = debug
        
        # Initialize API clients
        self._initialize_apis()
        
        # Initialize components
        self.tts = None  # ElevenLabs streamer
        self.stt = None  # Deepgram transcriber
        
        # Conversation state
        self.conversation_history = []
        self.is_listening = False
        self.is_speaking = False
        self.conversation_active = False
        self.conversation_ended = False
        
        # For tracking responses
        self.user_transcript = ""
        self.last_user_response = ""
        self.response_event = asyncio.Event()
        
        logger.info(f"LLM Voice Bot initialized with {llm_provider} provider")
    
    def _initialize_apis(self):
        """Initialize API clients for LLM and TTS"""
        # Set up ElevenLabs
        try:
            import elevenlabs
            import os

            # Set API key directly - the new method
            elevenlabs.api_key = self.elevenlabs_api_key
    
            # Also set it in environment for other components
            os.environ["ELEVENLABS_API_KEY"] = self.elevenlabs_api_key
        except ImportError:
            logger.warning("ElevenLabs library not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize ElevenLabs: {e}")
        
        # Set up LLM client based on provider
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            try:
                import openai
                self.llm_client = openai.OpenAI(api_key=self.openai_api_key)
            except ImportError:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
        elif self.llm_provider == "groq":
            if not self.groq_api_key:
                raise ValueError("Groq API key is required for Groq provider")
            try:
                import groq
                self.llm_client = groq.Groq(api_key=self.groq_api_key)
            except ImportError:
                raise ImportError("Groq library not installed. Install with: pip install groq")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    async def start_conversation(self):
        """Start the interactive conversation with LLM"""
        logger.info("Starting LLM-powered conversation")
        
        # Initialize TTS with ElevenLabs
        self.tts = ElevenLabsStreamer(self.client, self.call_id, use_local_audio=False)
        
        # Initialize STT with Deepgram
        self.stt = DeepgramTranscriber(
            client=self.client,
            call_id=self.call_id,
            api_key=self.deepgram_api_key
        )
        
        # Set up conversation history with system prompt
        system_prompt = self._fill_placeholders(SYSTEM_PROMPT)
        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Start the conversation
        self.conversation_active = True
        
        # Start transcription
        await self.stt.start_transcription(self.handle_transcription)
        
        # Generate initial bot message
        first_response = await self._get_llm_response()
        
        # Begin conversation loop
        while self.conversation_active and not self.conversation_ended:
            # Speak the current bot response
            await self._speak_bot_response(first_response)
            
            # Wait for and process user response
            user_responded = await self._wait_for_user_response()
            
            if not user_responded or not self.conversation_active:
                break
                
            # Get next bot response from LLM
            first_response = await self._get_llm_response()
            
            # Check for conversation end indicators in LLM response
            if self._should_end_conversation(first_response):
                logger.info("LLM indicated conversation should end")
                # Speak the final message before ending
                await self._speak_bot_response(first_response)
                break
        
        # Clean up
        await self.end_conversation()
        logger.info("LLM conversation ended")
    
    async def _speak_bot_response(self, response: str):
        """Speak a bot response using ElevenLabs TTS"""
        if not response:
            return
            
        # Log the bot's message
        logger.info(f"Bot: {response}")
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Speak the message
        self.is_speaking = True
        await self.tts.stream_from_elevenlabs(
            text=response, 
            voice_id=self.voice_id
        )
        self.is_speaking = False
        
        # Reset for next user response
        self.last_user_response = ""
        self.user_transcript = ""
        self.is_listening = True
        self.response_event.clear()
    
    async def _wait_for_user_response(self, timeout: float = 10.0) -> bool:
        """Wait for user to respond, with timeout"""
        try:
            # Wait for response event or timeout
            await asyncio.wait_for(self.response_event.wait(), timeout)
            
            return self.response_event.is_set()
                
        except asyncio.TimeoutError:
            logger.info("Response timeout - user didn't respond")
            return False
    
    async def _get_llm_response(self) -> str:
        """Get a response from the LLM based on conversation history"""
        try:
            # Format conversation history for the LLM API
            messages = []
            
            # Add system message
            system_message = next((msg for msg in self.conversation_history if msg["role"] == "system"), None)
            if system_message:
                messages.append({"role": "system", "content": system_message["content"]})
            
            # Add conversation messages (excluding system message)
            for msg in self.conversation_history:
                if msg["role"] != "system":
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Call the appropriate LLM API based on provider
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                return response.choices[0].message.content
                
            elif self.llm_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return "I'm sorry, I'm having trouble processing your request. Could you please try again?"
    
    def _should_end_conversation(self, response: str) -> bool:
        """Check if the conversation should end based on the LLM response"""
        # Look for ending phrases in the response
        ending_phrases = [
            "goodbye", 
            "bye", 
            "have a nice day", 
            "have a good day",
            "thank you for calling",
            "thanks for calling"
        ]
        
        # Check if any ending phrase is in the response
        response_lower = response.lower()
        for phrase in ending_phrases:
            if phrase in response_lower:
                return True
                
        # Check if we've had a long enough conversation (more than 10 turns)
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        if len(user_messages) >= 10:
            return True
                
        return False
    
    async def handle_transcription(self, transcript: str, is_final: bool):
        """Handle transcription results from Deepgram"""
        # Update the current transcript
        self.user_transcript = transcript
        
        # For debugging
        if self.debug:
            logger.debug(f"Transcript: {transcript} (final: {is_final})")
        
        # Only process final transcriptions
        if is_final and self.is_listening:
            # Store the response
            self.last_user_response = transcript
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": transcript,
                "timestamp": datetime.now().isoformat()
            })
            
            # Log user response
            logger.info(f"User: {transcript}")
            
            # Signal that we have a response
            self.is_listening = False
            self.response_event.set()
    
    def _fill_placeholders(self, text: str) -> str:
        """Fill placeholders in the text with context data"""
        for key, value in self.context_data.items():
            placeholder = f"{{{key}}}"
            text = text.replace(placeholder, value)
        return text
    
    async def end_conversation(self):
        """End the conversation and clean up resources"""
        logger.info("Ending LLM conversation")
        
        # Stop TTS
        if self.tts:
            self.tts.stop_streaming()
        
        # Stop STT
        if self.stt:
            self.stt.stop_transcription()
        
        # Save conversation history
        self.save_conversation_history()
        
        # Mark conversation as ended
        self.conversation_active = False
        self.conversation_ended = True
    
    def save_conversation_history(self):
        """Save the conversation history to a file"""
        if not self.conversation_history:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs("conversation_logs", exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_logs/llm_conversation_{timestamp}.json"
            
            # Save conversation history
            with open(filename, 'w') as f:
                json.dump({
                    "call_id": self.call_id,
                    "timestamp": datetime.now().isoformat(),
                    "llm_provider": self.llm_provider,
                    "llm_model": self.llm_model,
                    "conversation": self.conversation_history
                }, f, indent=2)
                
            logger.info(f"Conversation history saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")

async def run_llm_voice_bot(args, config):
    """Run the LLM voice bot demo using config"""
    # Get API keys from config or args
    deepgram_api_key = args.deepgram_api_key or config.get_api_key("deepgram")
    elevenlabs_api_key = args.elevenlabs_api_key or config.get_api_key("elevenlabs")
    openai_api_key = args.openai_api_key or config.get_api_key("openai")
    groq_api_key = args.groq_api_key or config.get_api_key("groq")
    
    # Check which LLM provider to use
    llm_provider = args.llm_provider or config.get("llm_provider", "openai")
    
    # Verify we have the necessary API key for the selected provider
    if llm_provider == "openai" and not openai_api_key:
        logger.error("OpenAI API key is required when using the OpenAI provider")
        return False
    elif llm_provider == "groq" and not groq_api_key:
        logger.error("Groq API key is required when using the Groq provider")
        return False
        
    if not deepgram_api_key:
        logger.error("Deepgram API key is required. Set it in the config file or provide it with --deepgram-api-key")
        return False
        
    if not elevenlabs_api_key:
        logger.error("ElevenLabs API key is required. Set it in the config file or provide it with --elevenlabs-api-key")
        return False
    
    # Get SIP connection details from config or args
    server_uri = args.server or config.get("opensips", {}).get("server") or config.get("VOICEBOT_OPENSIPS_SERVER")
    username = args.username or config.get("opensips", {}).get("username") or config.get("VOICEBOT_OPENSIPS_USERNAME")
    password = args.password or config.get("opensips", {}).get("password") or config.get("VOICEBOT_OPENSIPS_PASSWORD")
    domain = args.domain or config.get("opensips", {}).get("domain") or config.get("VOICEBOT_OPENSIPS_DOMAIN")
    debug_level = args.debug_level or config.get("opensips", {}).get("debug_level") or config.get("VOICEBOT_OPENSIPS_DEBUG_LEVEL", 1)
    
    # Check for required SIP connection details
    if not all([server_uri, username, password, domain]):
        logger.error("Missing SIP connection details. Check your config file or provide them as arguments.")
        return False
    
    # Get voice settings from config or args
    voice_id = args.voice_id or config.get("voice", {}).get("elevenlabs_voice_id") or config.get("VOICEBOT_ELEVENLABS_VOICE_ID", "gUbIduqGzBP438teh4ZA")
    
    # Get LLM model from config or args
    llm_model = args.llm_model or config.get("llm_model")
    if not llm_model:
        if llm_provider == "openai":
            llm_model = "gpt-3.5-turbo"
        elif llm_provider == "groq":
            llm_model = "llama3-8b-8192"
    
    # Create client instance
    client = create_client(
        config_path=args.config,  # Use config file if provided
        server_uri=server_uri,    # Override with any explicit parameters
        username=username,
        password=password,
        domain=domain,
        debug_level=int(debug_level)
    )
    
    try:
        # Connect to server
        logger.info(f"Connecting to OpenSIPS WebSocket server: {server_uri}")
        connected = await client.connect()
        
        if not connected:
            logger.error("Failed to connect to OpenSIPS WebSocket server")
            return False
        
        # Register with server
        logger.info("Registering with SIP server...")
        registered = await client.register()
        
        if not registered:
            logger.error("Failed to register with SIP server")
            return False
        
        logger.info("Successfully registered!")
        
        # Get call destination from args or config
        destination = args.destination or config.get("VOICEBOT_DESTINATION")
        if not destination:
            logger.error("Destination number/extension is required")
            return False
        
        # Place a call to the destination
        logger.info(f"Placing call to extension {destination}...")
        result = await client.place_call(destination, timeout=30)
        
        if not result["success"]:
            logger.error(f"Call failed: {result['state']}")
            return False
            
        # Call was answered
        call_id = result["call_id"]
        logger.info(f"Call answered! Call-ID: {call_id}")
        
        # Wait a moment before starting the conversation
        await asyncio.sleep(1)
        
        # Get conversation context from config or args
        context_data = {
            "patient_name": args.patient_name or config.get("VOICEBOT_PATIENT_NAME", "John Smith"),
            "doctor_name": args.doctor_name or config.get("VOICEBOT_DOCTOR_NAME", "Dr. Johnson"),
            "appointment_date": args.appointment_date or config.get("VOICEBOT_APPOINTMENT_DATE", "May 20th"),
            "appointment_time": args.appointment_time or config.get("VOICEBOT_APPOINTMENT_TIME", "2:30 PM")
        }
        
        # Create and start the LLM voice bot
        bot = LLMVoiceBot(
            opensips_client=client,
            call_id=call_id,
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_voice_id=voice_id,
            deepgram_api_key=deepgram_api_key,
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            llm_model=llm_model,
            context_data=context_data,
            debug=int(debug_level) > 3
        )
        
        # Run the conversation
        await bot.start_conversation()
        
        # Wait a moment before ending call
        await asyncio.sleep(1)
        
        # End the call
        logger.info("Ending call...")
        await client.end_call(call_id)
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # Disconnect from the server
        if hasattr(client, 'disconnect') and client.connection:
            await client.disconnect()
        logger.info("Exiting...")

async def main():
    """Main entry point for the LLM voice bot demo"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='LLM-Powered Voice Bot Demo')
    parser.add_argument('--server', help='OpenSIPS WebSocket server URI (e.g., wss://example.com:8080/ws)')
    parser.add_argument('--username', help='SIP username')
    parser.add_argument('--password', help='SIP password')
    parser.add_argument('--domain', help='SIP domain')
    parser.add_argument('--destination', help='Destination extension to call')
    
    # Config file
    parser.add_argument('--config', help='Path to config file')
    
    # API keys
    parser.add_argument('--deepgram-api-key', help='Deepgram API key')
    parser.add_argument('--elevenlabs-api-key', help='ElevenLabs API key')
    parser.add_argument('--openai-api-key', help='OpenAI API key')
    parser.add_argument('--groq-api-key', help='Groq API key')
    
    # LLM settings
    parser.add_argument('--llm-provider', choices=['openai', 'groq'], 
                        help='LLM provider to use (openai or groq)')
    parser.add_argument('--llm-model', help='LLM model to use')
    
    # ElevenLabs settings
    parser.add_argument('--voice-id', help='ElevenLabs voice ID')
    
    # Conversation context
    parser.add_argument('--patient-name', help='Patient name for the conversation')
    parser.add_argument('--doctor-name', help='Doctor name for the conversation')
    parser.add_argument('--appointment-date', help='Appointment date for the conversation')
    parser.add_argument('--appointment-time', help='Appointment time for the conversation')
    
    # Debug settings
    parser.add_argument('--debug-level', type=int, choices=range(1, 10), 
                        help='Debug level 1-9, with 9 being most verbose')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run the LLM voice bot demo
    result = await run_llm_voice_bot(args, config)
    
    # Exit with appropriate status code
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
