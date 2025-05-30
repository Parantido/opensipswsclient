"""
Deepgram Speech Recognition Integration for OpenSIPS WebSocket SIP Client
Enables real-time speech-to-text transcription from active SIP calls
"""

import asyncio
import base64
import json
import wave
import pyaudio
import time
import threading
import queue
import socket
import struct
import logging
from typing import Dict, Optional, Callable, List, Union
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

NEW_SDK_VERSION = False

try:
    from deepgram import Deepgram, DeepgramClientOptions
    from deepgram.clients.live.v1 import LiveOptions, LiveTranscriptionEvents
    from deepgram.transcription import PrerecordedOptions

    # New SDK format
    NEW_SDK_VERSION = True

except ImportError:
    try:
        # Try the older import format (2.x)
        from deepgram import (
            DeepgramClient,
            LiveTranscriptionOptions,
            LiveOptions,
            PrerecordedOptions
        )
        NEW_SDK_VERSION = False
    except ImportError:
        # Keep NEW_SDK_VERSION defined as False
        pass

class DeepgramTranscriber:
    """
    Class for transcribing audio from SIP calls using Deepgram API
    Handles real-time capture and processing of RTP audio
    """
    
    def __init__(self, client, call_id: str, api_key: str, 
                 sample_rate: int = 8000, channels: int = 1,
                 language: str = "en-US", model: str = "nova-2"):
        self.client = client  # OpenSIPSClient instance
        self.call_id = call_id
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.channels = channels
        self.language = language
        self.model = model
        
        # Transcription settings
        self.is_transcribing = False
        self.transcription_queue = queue.Queue()
        self.transcription_thread = None
        self.callback = None
        
        # RTP socket for receiving audio
        self.rtp_socket = None
        self.local_port = None
        
        # Audio buffer and processing
        self.audio_buffer = bytearray()
        self.buffer_lock = threading.Lock()
        self.deepgram_client = None
        
        # Transcription results
        self.transcript = ""
        self.is_final = False
    
    def initialize_rtp_receiver(self, local_port: Optional[int] = None) -> bool:
        """Initialize RTP receiver for collecting audio"""
        if self.client.calls.get(self.call_id) is None:
            raise ValueError(f"Call with ID {self.call_id} not found")
            
        call = self.client.calls[self.call_id]
        if not call.get("answered", False):
            raise ValueError(f"Call {self.call_id} not answered yet")
        
        # Use specified local port or get from call
        self.local_port = local_port or call.get("local_port", 10000) + 2
        
        # Create UDP socket for RTP
        self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to local port
        try:
            self.rtp_socket.bind(('0.0.0.0', self.local_port))
            logger.info(f"RTP receiver listening on port {self.local_port}")
        except OSError as e:
            logger.error(f"Failed to bind to port {self.local_port}: {e}")
            # Try alternate port if binding fails
            alt_port = self.local_port + 2
            try:
                self.rtp_socket.bind(('0.0.0.0', alt_port))
                self.local_port = alt_port
                logger.info(f"RTP receiver listening on alternate port {alt_port}")
            except OSError as e2:
                logger.error(f"Failed to bind to alternate port {alt_port}: {e2}")
                return False
        
        # Set socket timeout to make it non-blocking
        self.rtp_socket.settimeout(0.1)
        return True
    
    async def start_transcription(self, callback: Callable[[str, bool], None]) -> bool:
        """
        Start transcribing audio from the call
    
        Args:
            callback: Function to call with transcription results
                     First parameter is the transcript, second is is_final (bool)
    
        Returns:
            bool: Success status
        """
        try:
            global NEW_SDK_VERSION
    
            # Initialize Deepgram client based on SDK version
            if NEW_SDK_VERSION:
                # For Deepgram SDK 4.x
                self.deepgram_client = Deepgram(self.api_key)
            else:
                # For Deepgram SDK 2.x
                self.deepgram_client = DeepgramClient(self.api_key)
    
            # Store callback
            self.callback = callback
    
            # Initialize RTP receiver if not already done
            if not self.rtp_socket:
                if not self.initialize_rtp_receiver():
                    raise Exception("Failed to initialize RTP receiver")
    
            # Start transcription in a separate thread
            self.is_transcribing = True
            self.transcription_thread = threading.Thread(
                target=self._transcription_worker
            )
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
    
            return True
        except ImportError:
            raise ImportError("Deepgram library not installed. Install with: pip install deepgram-sdk")
        except Exception as e:
            logger.error(f"Error starting transcription: {e}")
            return False
    
    def _transcription_worker(self):
        """Worker thread to handle audio capture and transcription"""
        try:
            # Import Deepgram here to handle version differences
            try:
                # For Deepgram SDK 4.x
                from deepgram import Deepgram
                from deepgram.clients.live.v1 import LiveOptions, LiveTranscriptionEvents
                from deepgram.options import UrlParams
    
                # Create connection to Deepgram
                dg_client = Deepgram(self.api_key)
    
                # Configure live transcription
                options = {
                    "model": self.model,
                    "language": self.language,
                    "encoding": "linear16",
                    "channels": self.channels,
                    "sample_rate": self.sample_rate,
                    "interim_results": True,
                    "punctuate": True,
                    "smart_format": True
                }
    
                # Create a live transcription connection
                connection = dg_client.listen.live.v("1")
    
                # Define callback functions
                def on_message(result, **kwargs):
                    try:
                        # Process transcription results
                        if "type" in result and result["type"] == "Results":
                            results = result.get("results", {}).get("channels", [{}])[0]
                            alternatives = results.get("alternatives", [{}])
                            if alternatives:
                                transcript = alternatives[0].get("transcript", "")
                                is_final = not results.get("is_interim", True)
    
                                # Call the callback with the results
                                if self.callback and transcript:
                                    asyncio.run_coroutine_threadsafe(
                                        self.callback(transcript, is_final),
                                        asyncio.get_event_loop()
                                    )
                    except Exception as e:
                        logger.error(f"Error processing transcription result: {e}")
    
                def on_error(error, **kwargs):
                    logger.error(f"Deepgram error: {error}")
    
                def on_close():
                    logger.info("Deepgram connection closed")
    
                def on_open():
                    logger.info("Deepgram connection opened")
    
                # Register callbacks
                connection.on(LiveTranscriptionEvents.Transcript, on_message)
                connection.on(LiveTranscriptionEvents.Error, on_error)
                connection.on(LiveTranscriptionEvents.Close, on_close)
                connection.on(LiveTranscriptionEvents.Open, on_open)
    
                # Start the connection
                connection.start(options)
    
            except ImportError:
                # Try older Deepgram SDK versions (pre-4.0)
                try:
                    from deepgram import (
                        DeepgramClient,
                        LiveTranscriptionOptions,
                        LiveOptions,
                        PrerecordedOptions
                    )
    
                    # Initialize live transcription
                    options = LiveTranscriptionOptions(
                        model=self.model,
                        language=self.language,
                        encoding="linear16",
                        channels=self.channels,
                        sample_rate=self.sample_rate,
                        interim_results=True,
                        punctuate=True,
                        smart_format=True
                    )
    
                    # Create a live transcription connection
                    live_options = LiveOptions(
                        options=options
                    )
    
                    # Create connection object
                    deepgram_client = DeepgramClient(self.api_key)
                    connection = deepgram_client.listen.live.v("1")
    
                    # Define callback functions
                    def on_message(result, **kwargs):
                        try:
                            # Process transcription results
                            if result.get("type") == "Results":
                                results = result.get("results", {}).get("channels", [{}])[0]
                                alternatives = results.get("alternatives", [{}])
                                if alternatives:
                                    transcript = alternatives[0].get("transcript", "")
                                    is_final = not results.get("is_interim", True)
    
                                    # Call the callback with the results
                                    if self.callback and transcript:
                                        asyncio.run_coroutine_threadsafe(
                                            self.callback(transcript, is_final),
                                            asyncio.get_event_loop()
                                        )
                        except Exception as e:
                            logger.error(f"Error processing transcription result: {e}")
    
                    def on_error(error, **kwargs):
                        logger.error(f"Deepgram error: {error}")
    
                    def on_close():
                        logger.info("Deepgram connection closed")
    
                    def on_open():
                        logger.info("Deepgram connection opened")
    
                    # Register callbacks
                    connection.on("message", on_message)
                    connection.on("error", on_error)
                    connection.on("close", on_close)
                    connection.on("open", on_open)
    
                    # Start the connection
                    connection.start(live_options)
                except ImportError:
                    logger.error("Failed to import Deepgram. Please install with: pip install deepgram-sdk")
                    return
                except Exception as e:
                    logger.error(f"Error initializing older Deepgram SDK: {e}")
                    return
    
            # Process audio packets
            buffer = bytearray()
            while self.is_transcribing:
                try:
                    # Receive RTP packet
                    packet, addr = self.rtp_socket.recvfrom(2048)
    
                    # Process RTP header (12 bytes)
                    if len(packet) < 12:
                        continue
    
                    # Extract payload type
                    pt = packet[1] & 0x7F  # Payload type is in the second byte
    
                    # Extract audio data (skip RTP header)
                    audio_data = packet[12:]
    
                    # For PCMU (G.711 µ-law, payload type 0) or PCMA (G.711 A-law, payload type 8)
                    # We need to convert to linear PCM for Deepgram
                    if pt == 0:  # PCMU
                        # Convert µ-law to linear PCM
                        audio_data = ulaw_to_linear(audio_data)
                    elif pt == 8:  # PCMA
                        # Convert A-law to linear PCM
                        audio_data = alaw_to_linear(audio_data)
    
                    # Send audio data to Deepgram
                    if audio_data:
                        connection.send(audio_data)
    
                except socket.timeout:
                    # Socket timeout is expected, just continue
                    pass
                except Exception as e:
                    logger.error(f"Error processing audio packet: {e}")
                    time.sleep(0.1)  # Prevent tight loop on error
    
            # Close the connection when done
            try:
                connection.finish()
            except:
                pass
    
        except Exception as e:
            logger.error(f"Error in transcription worker: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def stop_transcription(self):
        """Stop the audio transcription"""
        self.is_transcribing = False
        
        if self.rtp_socket:
            self.rtp_socket.close()
            self.rtp_socket = None
        
        # Wait for transcription thread to finish
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)
        
        # Clear the queue
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except queue.Empty:
                break
                
        return True
    
    def _ulaw_to_linear(self, u_law_data):
        """Convert G.711 µ-law to linear PCM"""
        # This is a simplified conversion
        # For production, consider using specialized audio libraries
        result = bytearray(len(u_law_data) * 2)
        
        for i, byte in enumerate(u_law_data):
            # Flip all bits
            byte = ~byte & 0xFF
            
            # Extract sign and magnitude
            sign = 1 if (byte & 0x80) else -1
            position = ((byte & 0x70) >> 4) + 5
            quantization = ((byte & 0x0F) << 1) | 1
            
            # Convert to 16-bit linear PCM
            value = sign * (quantization << position)
            
            # Convert to bytes (little-endian)
            result[i*2] = value & 0xFF
            result[i*2+1] = (value >> 8) & 0xFF
            
        return bytes(result)
    
    def _alaw_to_linear(self, a_law_data):
        """Convert G.711 A-law to linear PCM"""
        # This is a simplified conversion
        # For production, consider using specialized audio libraries
        result = bytearray(len(a_law_data) * 2)
        
        for i, byte in enumerate(a_law_data):
            # Invert every other bit
            byte ^= 0x55
            
            # Extract sign and magnitude
            sign = -1 if (byte & 0x80) else 1
            position = ((byte & 0x70) >> 4)
            quantization = (byte & 0x0F)
            
            # Convert to 16-bit linear PCM
            value = sign * (((quantization << 1) | 1) << position)
            
            # Convert to bytes (little-endian)
            result[i*2] = value & 0xFF
            result[i*2+1] = (value >> 8) & 0xFF
            
        return bytes(result)

# Function to use in client code
async def transcribe_call(client, call_id: str, api_key: str, 
                     callback: Callable[[str, bool], None],
                     language: str = "en-US", model: str = "nova-2") -> Optional[DeepgramTranscriber]:
    """
    Start transcribing audio from an active call
    
    Args:
        client: OpenSIPSClient instance
        call_id: ID of the active call
        api_key: Deepgram API key
        callback: Function to call with transcription results
                  First parameter is the transcript, second is is_final (bool)
        language: Language code
        model: Deepgram model to use
        
    Returns:
        DeepgramTranscriber instance or None if failed
    """
    try:
        # Create transcriber
        transcriber = DeepgramTranscriber(
            client=client, 
            call_id=call_id, 
            api_key=api_key,
            language=language,
            model=model
        )
        
        # Start transcription
        success = await transcriber.start_transcription(callback)
        
        if success:
            return transcriber
        return None
    except Exception as e:
        logger.error(f"Error starting call transcription: {e}")
        return None
