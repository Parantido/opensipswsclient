"""
ElevenLabs TTS Integration for OpenSIPS WebSocket SIP Client
Enables real-time streaming of TTS audio to active SIP calls
"""

import asyncio
import base64
import json
import io
import wave
import pyaudio
import time
from typing import Dict, Optional, Callable, Any
import threading
import queue

# For RTP streaming
import socket
import struct
import random

class ElevenLabsStreamer:
    """
    Class for streaming ElevenLabs TTS audio to SIP calls
    Handles real-time streaming of audio chunks to an active call
    """
    
    def __init__(self, client, call_id: str, sample_rate: int = 16000, chunk_size: int = 1024):
        self.client = client  # OpenSIPSClient instance
        self.call_id = call_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_streaming = False
        self.audio_queue = queue.Queue()
        self.stream_thread = None
        
        # RTP streaming parameters
        self.rtp_socket = None
        self.rtp_sequence = 0
        self.rtp_timestamp = 0
        self.rtp_ssrc = random.randint(0, 0xFFFFFFFF)  # Random synchronization source
        
        # Initialize PyAudio for processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    def initialize_rtp(self):
        """Initialize RTP socket for streaming"""
        if self.client.calls.get(self.call_id) is None:
            raise ValueError(f"Call with ID {self.call_id} not found")
            
        call = self.client.calls[self.call_id]
        if not call.get("answered", False):
            raise ValueError(f"Call {self.call_id} not answered yet")
        
        # Get remote IP and port from the call
        remote_ip = call.get("remote_ip")
        remote_port = call.get("remote_port")
        
        if not remote_ip or not remote_port:
            raise ValueError(f"Missing remote media information for call {self.call_id}")
        
        # Create UDP socket for RTP
        self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to local port (next port after the one used for SIP)
        local_port = call.get("local_port", 10000)
        try:
            self.rtp_socket.bind(('0.0.0.0', local_port))
        except OSError:
            # Try alternate port if binding fails
            self.rtp_socket.bind(('0.0.0.0', 0))
        
        # Store remote address for sending
        self.remote_address = (remote_ip, remote_port)
        return True
    
    async def stream_from_elevenlabs(self, text: str, voice_id: str = "gUbIduqGzBP438teh4ZA", 
                                     optimize_streaming_latency: int = 4) -> bool:
        """
        Stream audio from ElevenLabs TTS to an active call
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            optimize_streaming_latency: Latency optimization level (0-4)
            
        Returns:
            bool: Success status
        """
        if not self.client.calls.get(self.call_id):
            raise ValueError(f"Call with ID {self.call_id} not found or has ended")
        
        # Initialize RTP if not already done
        if not self.rtp_socket:
            self.initialize_rtp()
        
        # Start streaming in a separate thread to not block the event loop
        self.is_streaming = True
        
        # Create a thread for handling the streaming
        self.stream_thread = threading.Thread(
            target=self._streaming_worker,
            args=(text, voice_id, optimize_streaming_latency)
        )
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        return True
    
    def _streaming_worker(self, text: str, voice_id: str, optimize_streaming_latency: int):
        """Worker thread to handle the streaming process"""
        try:
            # Get audio stream from ElevenLabs
            audio_stream = self._get_elevenlabs_stream(
                text=text,
                voice_id=voice_id,
                optimize_streaming_latency=optimize_streaming_latency
            )
            
            # Process and stream audio chunks
            self._process_audio_stream(audio_stream)
            
        except Exception as e:
            print(f"Error in streaming worker: {e}")
            # Try to play a fallback message
            fallback_text = "Sorry, I encountered an issue while processing."
            try:
                audio_stream = self._get_elevenlabs_stream(
                    text=fallback_text,
                    voice_id=voice_id,
                    optimize_streaming_latency=optimize_streaming_latency
                )
                self._process_audio_stream(audio_stream)
            except Exception:
                pass
        finally:
            self.is_streaming = False
    
    def _get_elevenlabs_stream(self, text: str, voice_id: str, optimize_streaming_latency: int):
        """
        Get streaming audio from ElevenLabs
        
        This method should be replaced with actual implementation using ElevenLabs API
        For now, it's a placeholder that returns an iterable of audio chunks
        """
        # Import the ElevenLabs module here to avoid circular imports
        try:
            from elevenlabs import generate, stream
            from elevenlabs.api import Models
            
            # Stream audio from ElevenLabs
            audio_stream = generate(
                text=text,
                voice=voice_id,
                model=Models.ELEVEN_TURBO_V2,
                stream=True,
                latency=optimize_streaming_latency
            )
            
            return audio_stream
            
        except ImportError:
            raise ImportError("ElevenLabs library not installed. Install with: pip install elevenlabs")
        except Exception as e:
            raise Exception(f"Error getting audio from ElevenLabs: {e}")
    
    def _process_audio_stream(self, audio_stream):
        """Process incoming audio stream and send via RTP"""
        # For PCM u-law (PCMU, payload type 0)
        payload_type = 0
        
        # Process audio chunks
        for chunk in audio_stream:
            if not self.is_streaming:
                break
                
            # Process chunk for RTP
            self._send_audio_chunk_rtp(chunk, payload_type)
            
        # Close the stream when done
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def _send_audio_chunk_rtp(self, chunk, payload_type=0):
        """Send an audio chunk via RTP"""
        if not self.rtp_socket or not self.is_streaming:
            return
            
        # Convert chunk to PCM if needed
        # This is simplified - in reality you'd need to convert to the correct format
        # based on the negotiated codec (usually G.711 u-law or a-law)
        
        # Create RTP header
        # RTP header format:
        #  0                   1                   2                   3
        #  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # |V=2|P|X|  CC   |M|     PT      |       sequence number         |
        # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # |                           timestamp                           |
        # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # |           synchronization source (SSRC) identifier            |
        # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        
        version = 2
        padding = 0
        extension = 0
        cc = 0
        marker = 0
        
        # First byte: V=2, P=0, X=0, CC=0
        first_byte = (version << 6) | (padding << 5) | (extension << 4) | cc
        
        # Second byte: M=0, PT=payload_type
        second_byte = (marker << 7) | payload_type
        
        # Create RTP header
        header = struct.pack('!BBHII', 
                            first_byte, 
                            second_byte, 
                            self.rtp_sequence, 
                            self.rtp_timestamp, 
                            self.rtp_ssrc)
        
        # Increment sequence number (wrap at 16 bits)
        self.rtp_sequence = (self.rtp_sequence + 1) & 0xFFFF
        
        # Increment timestamp based on samples
        # For 8kHz audio, increment by 160 samples per 20ms
        self.rtp_timestamp += len(chunk)
        
        # Send RTP packet
        try:
            packet = header + chunk
            self.rtp_socket.sendto(packet, self.remote_address)
            
            # Small delay to control send rate
            time.sleep(0.020)  # 20ms typical for RTP packets
        except Exception as e:
            print(f"Error sending RTP packet: {e}")
    
    def stop_streaming(self):
        """Stop the audio streaming"""
        self.is_streaming = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.rtp_socket:
            self.rtp_socket.close()
            self.rtp_socket = None
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        # Wait for streaming thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)
            
        return True

# Function to use in place of your original text_to_speech
async def stream_tts_to_call(client, call_id: str, text: str, 
                           voice_id: str = "gUbIduqGzBP438teh4ZA",
                           optimize_streaming_latency: int = 4) -> bool:
    """
    Stream TTS audio directly to an active call
    
    Args:
        client: OpenSIPSClient instance
        call_id: ID of the active call
        text: Text to convert to speech
        voice_id: ElevenLabs voice ID
        optimize_streaming_latency: Latency optimization level (0-4)
        
    Returns:
        bool: Success status
    """
    try:
        # Create streamer
        streamer = ElevenLabsStreamer(client, call_id)
        
        # Stream audio
        await streamer.stream_from_elevenlabs(
            text=text,
            voice_id=voice_id,
            optimize_streaming_latency=optimize_streaming_latency
        )
        
        return True
    except Exception as e:
        print(f"Error streaming TTS to call: {e}")
        return False
