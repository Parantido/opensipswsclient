i"""
Audio processing helpers for OpenSIPsCall

Handles audio conversions and formats for SIP calls
"""

import wave
import pyaudio
import threading
import queue
import socket
import struct
import random
import time
from typing import Optional, Dict, Any

class AudioHandler:
    """Class for handling audio streams in SIP calls"""
    
    def __init__(self, local_port: int, remote_ip: str, remote_port: int):
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def play_audio_file(self, file_path: str) -> None:
        """Play an audio file to the remote endpoint"""
        # Open the audio file
        if file_path.lower().endswith('.wav'):
            try:
                wf = wave.open(file_path, 'rb')
                
                # Initialize audio stream
                self.stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                # Read data in chunks and play
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                while data:
                    self.stream.write(data)
                    data = wf.readframes(chunk_size)
                    
                # Close everything when done
                self.stream.stop_stream()
                self.stream.close()
                wf.close()
                
            except Exception as e:
                print(f"Error playing audio file: {e}")
        else:
            print("Unsupported audio file format. Only WAV is currently supported.")
                
    def stop(self) -> None:
        """Stop audio handling"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        self.audio.terminate()

def ulaw_to_linear(u_law_data):
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

def alaw_to_linear(a_law_data):
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

def create_rtp_packet(payload, sequence_number, timestamp, ssrc, payload_type=0, marker=0):
    """
    Create an RTP packet with the given parameters
    
    Args:
        payload: Audio data payload
        sequence_number: RTP sequence number (16 bits)
        timestamp: RTP timestamp (32 bits)
        ssrc: Synchronization source identifier (32 bits)
        payload_type: RTP payload type (7 bits)
        marker: RTP marker bit (1 bit)
        
    Returns:
        bytes: RTP packet
    """
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
    
    # First byte: V=2, P=0, X=0, CC=0
    first_byte = (version << 6) | (padding << 5) | (extension << 4) | cc
    
    # Second byte: M=marker, PT=payload_type
    second_byte = (marker << 7) | payload_type
    
    # Create R
