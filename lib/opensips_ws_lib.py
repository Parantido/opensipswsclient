"""
OpenSIPS WebSocket Library - A Python library for interacting with OpenSIPS via WebSockets
Pure Python implementation with no external SIP dependencies
Enhanced debug logging with configurable levels
"""

import asyncio
import json
import logging
import re
import uuid
import hashlib
import time
import wave
import pyaudio
from typing import Dict, List, Optional, Tuple, Union, Callable
import websockets
import pkg_resources
import sys

# Configure logging with a custom level system
# Debug levels from 1-9 with 9 being the most verbose
DEBUG_LEVEL = 1  # Default level, will be set by the client

class DebugLogger:
    """Custom logger with configurable debug levels"""
    
    def __init__(self, name, level=1):
        self.logger = logging.getLogger(name)
        self.debug_level = level
        
    def set_debug_level(self, level):
        """Set the debug level (1-9)"""
        self.debug_level = max(1, min(9, level))
        
    def debug(self, msg, level=1):
        """Log debug message if the current debug level is >= the message level"""
        if self.debug_level >= level:
            self.logger.debug(f"[DEBUG-L{level}] {msg}")
            
    def info(self, msg):
        """Log info message"""
        self.logger.info(msg)
        
    def warning(self, msg):
        """Log warning message"""
        self.logger.warning(msg)
        
    def error(self, msg):
        """Log error message"""
        self.logger.error(msg)
        
    def critical(self, msg):
        """Log critical message"""
        self.logger.critical(msg)
        
# Initialize logger
logger = DebugLogger("opensips_ws_lib")

# Check websockets version to use the correct parameter name
try:
    websockets_version = pkg_resources.get_distribution("websockets").version
    major_version = int(websockets_version.split('.')[0])
    logger.debug(f"Detected websockets version: {websockets_version}", level=2)
except Exception as e:
    logger.warning(f"Could not determine websockets version: {e}")
    major_version = 0  # Default to using older parameter name

class SIPMessage:
    """Class representing a SIP message"""
    
    def __init__(self, method: Optional[str] = None, 
                 status_code: Optional[int] = None, 
                 reason: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None, 
                 content: Optional[str] = None):
        self.method = method
        self.status_code = status_code
        self.reason = reason
        self.headers = headers or {}
        self.content = content or ""
        
    @classmethod
    def parse(cls, message: str) -> 'SIPMessage':
        """Parse a SIP message string into a SIPMessage object"""
        lines = message.strip().split('\r\n')
        
        # Parse first line
        first_line = lines[0]
        method = None
        status_code = None
        reason = None
        
        if first_line.startswith('SIP/2.0 '):
            # Response
            parts = first_line.split(' ', 2)
            if len(parts) >= 3:
                status_code = int(parts[1])
                reason = parts[2]
        else:
            # Request
            parts = first_line.split(' ', 1)
            if len(parts) >= 1:
                method = parts[0]
        
        # Parse headers
        headers = {}
        content_start = 0
        
        for i, line in enumerate(lines[1:], 1):
            if not line.strip():
                content_start = i + 1
                break
                
            if ':' in line:
                name, value = line.split(':', 1)
                headers[name.strip()] = value.strip()
        
        # Get content
        content = '\r\n'.join(lines[content_start:]) if content_start else ""
        
        return cls(method, status_code, reason, headers, content)
    
    def to_string(self) -> str:
        """Convert SIPMessage to a string"""
        lines = []
    
        # First line
        if self.method:
            request_uri = self.headers.get('Request-URI', 'sip:unknown')
            lines.append(f"{self.method} {request_uri} SIP/2.0")
        elif self.status_code:
            lines.append(f"SIP/2.0 {self.status_code} {self.reason or ''}")
    
        # Ensure Content-Length is always present and comes last among headers
        headers_to_add = {}
        content_length = "0"
    
        # Process headers, excluding Content-Length for now
        for name, value in self.headers.items():
            if name != 'Request-URI' and name.lower() != 'content-length':
                headers_to_add[name] = value
            elif name.lower() == 'content-length':
                content_length = value
    
        # Add all headers except Content-Length
        for name, value in headers_to_add.items():
            lines.append(f"{name}: {value}")
    
        # Always add Content-Length header (even if 0)
        lines.append(f"Content-Length: {content_length}")
    
        # Add empty line to separate headers from content (MANDATORY in SIP)
        lines.append("")
    
        # Content (if any) - but don't add content if Content-Length is 0
        if self.content and content_length != "0":
            lines.append(self.content)
    
        # Join with \r\n and add the mandatory final \r\n
        message = "\r\n".join(lines)
    
        # CRITICAL: Every SIP message must end with an additional \r\n
        # This adds the final terminating CRLF that's required by the SIP protocol
        message += "\r\n"
    
        return message

class SDPGenerator:
    """Class for generating SDP content for SIP messages"""
    
    @staticmethod
    def generate_offer(local_ip: str, audio_port: int) -> str:
        """Generate SDP offer for audio call"""
        session_id = int(time.time())
        sdp = [
            "v=0",
            f"o=pythonclient {session_id} {session_id} IN IP4 {local_ip}",
            "s=OpenSIPS WebSocket Call",
            f"c=IN IP4 {local_ip}",
            "t=0 0",
            f"m=audio {audio_port} RTP/AVP 0 8 101",
            "a=rtpmap:0 PCMU/8000",
            "a=rtpmap:8 PCMA/8000",
            "a=rtpmap:101 telephone-event/8000",
            "a=fmtp:101 0-16",
            "a=sendrecv"
        ]
        return "\r\n".join(sdp)
    
    @staticmethod
    def parse_answer(sdp_text: str) -> Dict[str, Union[str, int, List[str]]]:
        """Parse SDP answer to extract media information"""
        lines = sdp_text.strip().split('\r\n')
        result = {
            'remote_ip': '',
            'remote_port': 0,
            'codecs': []
        }
        
        for line in lines:
            if line.startswith('c=IN IP4 '):
                result['remote_ip'] = line.split(' ')[2]
            elif line.startswith('m=audio '):
                parts = line.split(' ')
                result['remote_port'] = int(parts[1])
                result['codecs'] = parts[3:]
        
        return result

class SIPAuthHelper:
    """Helper class for SIP authentication"""
    
    @staticmethod
    def generate_auth_response(username: str, password: str, realm: str, nonce: str, 
                              method: str, uri: str, qop: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
        """Generate a proper SIP digest authentication response"""
        # Log the inputs for debugging at level 9
        logger.debug(f"Auth inputs: username={username}, realm={realm}, method={method}, uri={uri}, qop={qop}", level=9)
        
        # Calculate HA1 = MD5(username:realm:password)
        ha1_input = f"{username}:{realm}:{password}"
        ha1 = hashlib.md5(ha1_input.encode()).hexdigest()
        logger.debug(f"HA1 input: {ha1_input}", level=9)
        logger.debug(f"HA1 value: {ha1}", level=9)
        
        # Calculate HA2 = MD5(method:uri)
        ha2_input = f"{method}:{uri}"
        ha2 = hashlib.md5(ha2_input.encode()).hexdigest()
        logger.debug(f"HA2 input: {ha2_input}", level=9)
        logger.debug(f"HA2 value: {ha2}", level=9)
        
        # Generate cnonce and other qop-related parameters if needed
        cnonce = None
        nc = None
        
        if qop and qop.lower() in ('auth', 'auth-int'):
            cnonce = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16]
            nc = "00000001"
            
            # Calculate response = MD5(HA1:nonce:nc:cnonce:qop:HA2)
            response_input = f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}"
            response_hash = hashlib.md5(response_input.encode()).hexdigest()
            logger.debug(f"Response hash input (with qop): {response_input}", level=9)
            logger.debug(f"Response hash value: {response_hash}", level=9)
            return response_hash, cnonce, nc
        else:
            # Calculate response = MD5(HA1:nonce:HA2)
            response_input = f"{ha1}:{nonce}:{ha2}"
            response_hash = hashlib.md5(response_input.encode()).hexdigest()
            logger.debug(f"Response hash input (no qop): {response_input}", level=9)
            logger.debug(f"Response hash value: {response_hash}", level=9)
            return response_hash, None, None

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
                logger.error(f"Error playing audio file: {e}")
        else:
            logger.error("Unsupported audio file format. Only WAV is currently supported.")
                
    def stop(self) -> None:
        """Stop audio handling"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        self.audio.terminate()

class OpenSIPSClient:
    """Main client class for interacting with OpenSIPS via WebSockets"""
    
    def __init__(self, server_uri: str, username: str, password: str, domain: str, 
             debug_level: int = 1, config: Optional[Dict[str, any]] = None):
        self.server_uri = server_uri
        self.username = username
        self.password = password
        self.domain = domain
        self.connection = None
        self.registered = False
        self.calls = {}
        self.audio_handlers = {}
        self.message_listeners = []
        self.config = config or {}  # Store configuration
        
        # Set debug level
        logger.set_debug_level(debug_level)
        
        # For message routing - stores futures for pending responses
        self.pending_responses = {}
        
        logger.debug(f"Initialized OpenSIPS client with server: {server_uri}, username: {username}, domain: {domain}", level=2)  

    async def connect(self) -> bool:
        """Connect to the OpenSIPS WebSocket server"""
        try:
            # Create headers for OpenSIPS WebSocket connection
            headers = {
                'Origin': 'https://localhost',  # Required by OpenSIPS
                'Sec-WebSocket-Protocol': 'sip'  # Protocol identifier for SIP over WebSocket
            }
            
            logger.debug(f"Connecting to {self.server_uri} with headers: {headers}", level=3)
            
            # Use the correct parameter name based on websockets version
            connect_kwargs = {
                'subprotocols': ['sip']
            }
            
            # Choose the correct parameter name for headers based on websockets version
            if major_version >= 10:
                # For newer versions of websockets
                connect_kwargs['additional_headers'] = headers
                logger.debug("Using 'additional_headers' parameter for websockets connect", level=3)
            else:
                # For older versions of websockets
                connect_kwargs['extra_headers'] = headers
                logger.debug("Using 'extra_headers' parameter for websockets connect", level=3)
            
            # Connect with proper headers and subprotocol
            self.connection = await websockets.connect(
                self.server_uri,
                **connect_kwargs
            )
            logger.info("Connected to OpenSIPS WebSocket server")
            
            # Start the message listener
            asyncio.create_task(self._message_listener())
            
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            logger.debug(f"Connection error details: {str(e)}", level=3)
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}", level=4)
            return False
            
    async def register(self) -> bool:
        """Register with the OpenSIPS server with properly formatted Via header"""
        call_id = str(uuid.uuid4())
        from_tag = self._generate_tag()
        branch = self._generate_branch()
        
        # Set a host and port for the Via header - using a sensible default
        via_host = "127.0.0.1"
        via_port = "5060"

        # Check if we have a config with a specific via_host override
        if hasattr(self, 'config') and self.config:
            # Try to get via_host from config
            via_host = self.config.get("via_host", via_host)
            via_port = self.config.get("via_port", via_port)
        
        # Initial REGISTER without auth
        register_message = SIPMessage(
            method="REGISTER",
            headers={
                'Request-URI': f"sip:{self.domain}",
                # Fix: Include host:port in the Via header
                'Via': f"SIP/2.0/WSS {via_host}:{via_port};branch=z9hG4bK{branch}",
                'From': f"<sip:{self.username}@{self.domain}>;tag={from_tag}",
                'To': f"<sip:{self.username}@{self.domain}>",
                'Call-ID': call_id,
                'CSeq': "1 REGISTER",
                'Contact': f"<sip:{self.username}@{self.domain};transport=ws>;expires=3600",
                'Allow': 'INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO, PUBLISH, MESSAGE',
                'Expires': "3600",
                'Max-Forwards': "70",
                'User-Agent': "Python OpenSIPS WebSocket Client",
                'Content-Length': "0"
            }
        )
        
        # Log the REGISTER message at debug level 5
        register_str = register_message.to_string()
        logger.debug(f"Sending initial REGISTER request:\n{register_str}", level=5)
        
        # Send initial REGISTER
        await self.connection.send(register_str)
        logger.info("Sent initial REGISTER request")
        
        # Create a future to wait for 401 response
        auth_future = asyncio.Future()
        self.pending_responses[call_id] = {
            "type": "register",
            "future": auth_future
        }
        
        # Wait for 401 response
        try:
            logger.debug(f"Waiting for 401 response to initial REGISTER", level=3)
            response = await asyncio.wait_for(auth_future, timeout=5.0)
            
            if response.status_code == 401:
                # Extract authentication info from WWW-Authenticate header
                www_auth = response.headers.get('WWW-Authenticate', '')
                logger.debug(f"Authentication challenge received: {www_auth}", level=4)
                
                # Extract authentication parameters
                nonce_match = re.search(r'nonce="([^"]+)"', www_auth)
                realm_match = re.search(r'realm="([^"]+)"', www_auth)
                
                if nonce_match and realm_match:
                    nonce = nonce_match.group(1)
                    realm = realm_match.group(1)
                    
                    logger.debug(f"Extracted auth params: nonce={nonce}, realm={realm}", level=4)
                    
                    # The URI must be exactly as shown in the example
                    uri = f"sip:{self.domain}"
                    
                    # Calculate the response digest
                    # HA1 = MD5(username:realm:password)
                    ha1 = hashlib.md5(f"{self.username}:{realm}:{self.password}".encode()).hexdigest()
                    logger.debug(f"HA1 = MD5({self.username}:{realm}:{self.password}) = {ha1}", level=8)
                    
                    # HA2 = MD5(method:uri)
                    ha2 = hashlib.md5(f"REGISTER:{uri}".encode()).hexdigest()
                    logger.debug(f"HA2 = MD5(REGISTER:{uri}) = {ha2}", level=8)
                    
                    # response = MD5(HA1:nonce:HA2)
                    response_hash = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
                    logger.debug(f"response = MD5({ha1}:{nonce}:{ha2}) = {response_hash}", level=8)
                    
                    # Build authorization header
                    auth_header = (
                        f'Digest username="{self.username}", '
                        f'realm="{realm}", '
                        f'nonce="{nonce}", '
                        f'uri="{uri}", '
                        f'response="{response_hash}", '
                        f'algorithm=MD5'
                    )
                    
                    logger.debug(f"Generated auth header: {auth_header}", level=4)
                    
                    # Create authenticated REGISTER with incremented CSeq and fixed Via header
                    auth_register = SIPMessage(
                        method="REGISTER",
                        headers={
                            'Request-URI': f"sip:{self.domain}",
                            # Fix: Include host:port in the Via header
                            'Via': f"SIP/2.0/WSS {via_host}:{via_port};branch=z9hG4bK{self._generate_branch()}",
                            'From': f"<sip:{self.username}@{self.domain}>;tag={from_tag}",
                            'To': f"<sip:{self.username}@{self.domain}>",
                            'Call-ID': call_id,
                            'CSeq': "2 REGISTER",  # Increment CSeq
                            'Contact': f"<sip:{self.username}@{self.domain};transport=ws>;expires=3600",
                            'Authorization': auth_header,
                            'Allow': 'INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO, PUBLISH, MESSAGE',
                            'Expires': "3600",
                            'Max-Forwards': "70",
                            'User-Agent': "Python OpenSIPS WebSocket Client",
                            'Content-Length': "0"
                        }
                    )
                    
                    # Log the authenticated REGISTER
                    auth_register_str = auth_register.to_string()
                    logger.debug(f"Sending authenticated REGISTER request:\n{auth_register_str}", level=5)
                    
                    # Create a future for the auth response
                    auth_resp_future = asyncio.Future()
                    self.pending_responses[call_id] = {
                        "type": "register_auth",
                        "future": auth_resp_future
                    }
                    
                    # Send authenticated REGISTER
                    await self.connection.send(auth_register_str)
                    logger.info("Sent authenticated REGISTER request")
                    
                    # Wait for 200 OK response
                    try:
                        logger.debug(f"Waiting for response to authenticated REGISTER", level=3)
                        auth_response = await asyncio.wait_for(auth_resp_future, timeout=5.0)
                        
                        logger.debug(f"Received auth response: {auth_response.status_code} {auth_response.reason}", level=3)
                        logger.debug(f"Auth response headers: {auth_response.headers}", level=4)
                        
                        if auth_response.status_code == 200:
                            self.registered = True
                            logger.info("Registration successful")
                            return True
                        else:
                            logger.error(f"Registration failed with status: {auth_response.status_code} {auth_response.reason}")
                            
                            # Detailed debug
                            logger.debug(f"Auth response headers: {auth_response.headers}", level=3)
                            logger.debug(f"Auth response content: {auth_response.content}", level=3)
                            
                            return False
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for authentication response")
                        return False
                else:
                    logger.error("Missing authentication parameters in challenge")
                    logger.debug(f"Complete WWW-Authenticate header: {www_auth}", level=3)
                    return False
            else:
                logger.error(f"Unexpected response to REGISTER: {response.status_code} {response.reason}")
                logger.debug(f"Unexpected response headers: {response.headers}", level=3)
                logger.debug(f"Unexpected response content: {response.content}", level=3)
                return False
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for registration response")
            return False
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            import traceback
            logger.debug(f"Registration error traceback: {traceback.format_exc()}", level=3)
            return False
    
    async def place_call(self, destination: str, timeout: int = 30) -> Dict[str, any]:
        """Place a call to the specified destination with timeout"""
        call_id = str(uuid.uuid4())
        from_tag = self._generate_tag()
        branch = self._generate_branch()
        
        # Create local media info (simplified)
        local_ip = "127.0.0.1"  # Default, should be overridden by config
        audio_port = 10000  # In a real implementation, find a free port

        # Set a host and port for the Via header - using a sensible default
        # Avoid using domain in Via header to prevent SIP syntax issues with special characters
        via_host = "127.0.0.1"  # Default via host
        via_port = "5060"       # Default via port
    
        # Check if we have a config with a specific via_host override
        if hasattr(self, 'config') and self.config:
            # Try to get via_host from config
            via_host = self.config.get("via_host", via_host)
            via_port = self.config.get("via_port", via_port)
            # Also try to get local_ip for media from config
            local_ip = self.config.get("local_ip", local_ip)
        
        # Generate SDP offer
        sdp_offer = SDPGenerator.generate_offer(local_ip, audio_port)
        
        # Create INVITE request
        invite_message = SIPMessage(
            method="INVITE",
            headers={
                'Request-URI': f"sip:{destination}@{self.domain}",
                'Via': f"SIP/2.0/WSS {via_host}:{via_port};branch=z9hG4bK{branch}",
                'From': f"<sip:{self.username}@{self.domain}>;tag={from_tag}",
                'To': f"<sip:{destination}@{self.domain}>",
                'Call-ID': call_id,
                'CSeq': "1 INVITE",
                'Contact': f"<sip:{self.username}@{self.domain};transport=ws>",
                'Content-Type': "application/sdp",
                'Content-Length': str(len(sdp_offer)),
                'Max-Forwards': "70",
                'User-Agent': "Python OpenSIPS WebSocket Client"
            },
            content=sdp_offer
        )
        
        # Log the INVITE message
        invite_str = invite_message.to_string()
        logger.debug(f"Sending INVITE request:\n{invite_str}", level=5)
        
        # Store call details
        self.calls[call_id] = {
            "state": "calling",
            "from_tag": from_tag,
            "to_tag": None,
            "destination": destination,
            "branch": branch,
            "sdp_offer": sdp_offer,
            "local_ip": local_ip,
            "local_port": audio_port,
            "remote_ip": None,
            "remote_port": None,
            "ringing": False,
            "answered": False,
            "ended": False
        }
        
        # Send INVITE
        await self.connection.send(invite_str)
        logger.info(f"Placed call to {destination} with Call-ID: {call_id}")
        
        # Wait for call to be answered or timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.calls[call_id]["answered"]:
                logger.info(f"Call answered: {call_id}")
                return {"success": True, "call_id": call_id, "state": "answered"}
            
            if self.calls[call_id]["ended"]:
                logger.info(f"Call ended before being answered: {call_id}")
                return {"success": False, "call_id": call_id, "state": "ended"}
                
            # Check if call is ringing
            if not self.calls[call_id]["ringing"] and time.time() - start_time > 5:
                logger.warning(f"Call not ringing after 5 seconds: {call_id}")
                
            await asyncio.sleep(0.1)
        
        # If we get here, call timed out - hang up
        logger.info(f"Call timed out after {timeout} seconds: {call_id}")
        await self.end_call(call_id)
        return {"success": False, "call_id": call_id, "state": "timeout"}
    
    async def end_call(self, call_id: str) -> bool:
        """End a call by sending BYE request"""
        if call_id not in self.calls:
            logger.error(f"Cannot end call: Call-ID {call_id} not found")
            return False
    
        call = self.calls[call_id]
    
        # Get via_host and via_port from config, with defaults
        via_host = "127.0.0.1"
        via_port = "5060"
    
        if hasattr(self, 'config') and self.config:
            via_host = self.config.get("via_host", via_host)
            via_port = self.config.get("via_port", via_port)
    
        # Only send BYE if call was answered
        if call["state"] in ["answered", "established"]:
            branch = self._generate_branch()
    
            bye_message = SIPMessage(
                method="BYE",
                headers={
                    'Request-URI': f"sip:{call['destination']}@{self.domain}",
                    # Fix: Use proper host:port format instead of domain
                    'Via': f"SIP/2.0/WSS {via_host}:{via_port};branch=z9hG4bK{branch}",
                    'From': f"<sip:{self.username}@{self.domain}>;tag={call['from_tag']}",
                    'To': f"<sip:{call['destination']}@{self.domain}>;tag={call['to_tag']}",
                    'Call-ID': call_id,
                    'CSeq': "2 BYE",
                    'Contact': f"<sip:{self.username}@{self.domain};transport=ws>",
                    'Max-Forwards': "70",
                    'User-Agent': "Python OpenSIPS WebSocket Client"
                }
            )
    
            bye_str = bye_message.to_string()
            logger.debug(f"Sending BYE request:\n{bye_str}", level=5)
    
            await self.connection.send(bye_str)
            logger.info(f"Sent BYE for call: {call_id}")
        elif call["state"] == "calling" or call["state"] == "ringing":
            # Call is still in progress but not answered, send CANCEL
            branch = call["branch"]
    
            cancel_message = SIPMessage(
                method="CANCEL",
                headers={
                    'Request-URI': f"sip:{call['destination']}@{self.domain}",
                    # Fix: Use proper host:port format instead of domain
                    'Via': f"SIP/2.0/WSS {via_host}:{via_port};branch=z9hG4bK{branch}",
                    'From': f"<sip:{self.username}@{self.domain}>;tag={call['from_tag']}",
                    'To': f"<sip:{call['destination']}@{self.domain}>",
                    'Call-ID': call_id,
                    'CSeq': "1 CANCEL",
                    'Max-Forwards': "70",
                    'User-Agent': "Python OpenSIPS WebSocket Client"
                }
            )
    
            cancel_str = cancel_message.to_string()
            logger.debug(f"Sending CANCEL request:\n{cancel_str}", level=5)
    
            await self.connection.send(cancel_str)
            logger.info(f"Sent CANCEL for call: {call_id}")
    
        # Stop audio handling if any
        if call_id in self.audio_handlers:
            self.audio_handlers[call_id].stop()
            del self.audio_handlers[call_id]
    
        # Mark call as ended
        call["state"] = "ended"
        call["ended"] = True
    
        return True

    async def play_audio_to_call(self, call_id: str, audio_file: str) -> bool:
        """Play an audio file to an active call"""
        if call_id not in self.calls:
            logger.error(f"Cannot play audio: Call-ID {call_id} not found")
            return False
            
        call = self.calls[call_id]
        
        if not call["answered"]:
            logger.error(f"Cannot play audio: Call {call_id} not answered yet")
            return False
            
        if call["ended"]:
            logger.error(f"Cannot play audio: Call {call_id} already ended")
            return False
            
        # Create audio handler if it doesn't exist
        if call_id not in self.audio_handlers:
            if call["remote_ip"] and call["remote_port"]:
                self.audio_handlers[call_id] = AudioHandler(
                    call["local_port"],
                    call["remote_ip"],
                    call["remote_port"]
                )
            else:
                logger.error(f"Cannot play audio: Missing remote media information for call {call_id}")
                return False
        
        # Play the audio file
        try:
            # Run in a separate task to not block
            asyncio.create_task(self._play_audio_async(call_id, audio_file))
            return True
        except Exception as e:
            logger.error(f"Error playing audio to call {call_id}: {e}")
            return False
    
    async def _play_audio_async(self, call_id: str, audio_file: str) -> None:
        """Play audio file asynchronously"""
        if call_id in self.audio_handlers:
            # This runs in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.audio_handlers[call_id].play_audio_file, 
                audio_file
            )
    
    async def _message_listener(self) -> None:
        """Listen for incoming messages from the WebSocket"""
        try:
            while True:
                message = await self.connection.recv()
                
                # Print raw message for debugging
                logger.debug(f"Raw message received:\n{message}", level=5)
                
                # Parse the SIP message
                try:
                    sip_message = SIPMessage.parse(message)
                    await self._route_message(sip_message)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    logger.debug(f"Raw message that caused error: {message}", level=3)
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}", level=4)
        except websockets.ConnectionClosed:
            logger.error("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            import traceback
            logger.debug(f"Message listener error traceback: {traceback.format_exc()}", level=3)
    
    async def _route_message(self, message: SIPMessage) -> None:
        """Route the message to the appropriate handler based on headers"""
        # Get Call-ID
        call_id = message.headers.get('Call-ID', '')
        
        # If it's a response (has status code)
        if message.status_code is not None:
            logger.debug(f"Routing response: {message.status_code} {message.reason} for Call-ID: {call_id}", level=3)
            
            # Check if we have a pending response for this Call-ID
            if call_id in self.pending_responses:
                pending = self.pending_responses[call_id]
                
                # If the future is not done, set its result
                if not pending["future"].done():
                    logger.debug(f"Setting future result for pending response: {pending['type']}", level=4)
                    pending["future"].set_result(message)
                    return
                else:
                    logger.debug(f"Future already done for pending response: {pending['type']}", level=4)
            else:
                logger.debug(f"No pending response for Call-ID: {call_id}", level=4)
            
            # If no pending response or future already done, handle normally
            await self._handle_response(message)
            
        # If it's a request (has method)
        elif message.method is not None:
            logger.debug(f"Routing request: {message.method} for Call-ID: {call_id}", level=3)
            await self._handle_request(message)
    
    async def _handle_response(self, response: SIPMessage) -> None:
        """Handle SIP responses that weren't caught by pending response handlers"""
        # Extract Call-ID and CSeq to identify the corresponding request
        call_id = response.headers.get('Call-ID', '')
        cseq = response.headers.get('CSeq', '')
        
        logger.info(f"Received response: {response.status_code} {response.reason} for {cseq}")
        
        # Handle responses for INVITE
        if call_id in self.calls and "INVITE" in cseq:
            call = self.calls[call_id]
            
            # 100 Trying
            if response.status_code == 100:
                logger.info(f"Call {call_id}: Trying")
                
            # 180 Ringing
            elif response.status_code == 180:
                logger.info(f"Call {call_id}: Ringing")
                call["ringing"] = True
                call["state"] = "ringing"
                
            # 200 OK (Call answered)
            elif response.status_code == 200:
                logger.info(f"Call {call_id}: Answered")
            
                # Extract To tag
                to_header = response.headers.get('To', '')
                to_tag_match = re.search(r'tag=([^;>]+)', to_header)
                if to_tag_match:
                    call["to_tag"] = to_tag_match.group(1)
            
                # Extract Contact header for ACK Request-URI
                contact_header = response.headers.get('Contact', '')
                if contact_header:
                    # Extract URI from Contact header (remove < > if present)
                    contact_match = re.search(r'<([^>]+)>', contact_header)
                    if contact_match:
                        call["contact_uri"] = contact_match.group(1)
                    else:
                        # Contact might not have < > brackets
                        call["contact_uri"] = contact_header.split(';')[0].strip()
                    logger.debug(f"Extracted Contact URI: {call['contact_uri']}", level=4)
            
                # Extract SDP answer
                if response.content:
                    # Parse SDP to get remote media info
                    media_info = SDPGenerator.parse_answer(response.content)
                    call["remote_ip"] = media_info["remote_ip"]
                    call["remote_port"] = media_info["remote_port"]
            
                # Update call state
                call["answered"] = True
                call["state"] = "answered"
            
                # Send ACK
                await self._send_ack(call_id)
                
            # Handle failure responses
            elif response.status_code >= 400:
                logger.warning(f"Call {call_id} failed: {response.status_code} {response.reason}")
                call["ended"] = True
                call["state"] = "failed"
    
    async def _handle_request(self, request: SIPMessage) -> None:
        """Handle SIP requests"""
        method = request.method
        call_id = request.headers.get('Call-ID', '')
        
        logger.info(f"Received request: {method} for Call-ID: {call_id}")
        
        # Handle BYE (call termination by remote party)
        if method == "BYE" and call_id in self.calls:
            logger.info(f"Remote party ended call: {call_id}")
            
            # Mark call as ended
            self.calls[call_id]["ended"] = True
            self.calls[call_id]["state"] = "ended"
            
            # Stop audio handling if any
            if call_id in self.audio_handlers:
                self.audio_handlers[call_id].stop()
                del self.audio_handlers[call_id]
            
            # Send 200 OK response to BYE
            await self._send_response(request, 200, "OK")
    
    async def _send_ack(self, call_id: str) -> None:
        """Send ACK for a successful INVITE response"""
        if call_id not in self.calls:
            return
    
        call = self.calls[call_id]
    
        # Get via_host and via_port from config, with defaults
        via_host = "127.0.0.1"
        via_port = "5060"
    
        if hasattr(self, 'config') and self.config:
            via_host = self.config.get("via_host", via_host)
            via_port = self.config.get("via_port", via_port)
    
        # For ACK, the Request-URI should be the Contact from the 200 OK response
        # If we have it stored, use it; otherwise fallback to the original destination
        request_uri = call.get("contact_uri", f"sip:{call['destination']}@{self.domain}")
    
        # Create ACK request
        ack_message = SIPMessage(
            method="ACK",
            headers={
                'Request-URI': request_uri,
                'Via': f"SIP/2.0/WSS {via_host}:{via_port};branch=z9hG4bK{self._generate_branch()}",
                'From': f"<sip:{self.username}@{self.domain}>;tag={call['from_tag']}",
                'To': f"<sip:{call['destination']}@{self.domain}>;tag={call['to_tag']}",
                'Call-ID': call_id,
                'CSeq': "1 ACK",
                'Contact': f"<sip:{self.username}@{self.domain};transport=ws>",
                'Max-Forwards': "70",
                'User-Agent': "Python OpenSIPS WebSocket Client",
                'Content-Length': "0"
            }
        )
    
        ack_str = ack_message.to_string()
        logger.debug(f"Sending ACK request:\n{ack_str}", level=5)
    
        try:
            await self.connection.send(ack_str)
            logger.info(f"Sent ACK for call: {call_id}")
            logger.debug(f"ACK sent successfully for Call-ID: {call_id}", level=3)
        except Exception as e:
            logger.error(f"Failed to send ACK for call {call_id}: {e}")
            raise
        
        async def _send_response(self, request: SIPMessage, status_code: int, reason: str) -> None:
            """Send a SIP response to a request"""
            response = SIPMessage(
                status_code=status_code,
                reason=reason,
                headers={
                    'Via': request.headers.get('Via', ''),
                    'From': request.headers.get('From', ''),
                    'To': request.headers.get('To', ''),
                    'Call-ID': request.headers.get('Call-ID', ''),
                    'CSeq': request.headers.get('CSeq', ''),
                    'Contact': f"<sip:{self.username}@{self.domain};transport=ws>",
                    'User-Agent': "Python OpenSIPS WebSocket Client"
                }
            )
            
            response_str = response.to_string()
            logger.debug(f"Sending response:\n{response_str}", level=5)
            
            await self.connection.send(response_str)
    
    # Helper methods
    def _generate_branch(self) -> str:
        """Generate a unique branch parameter for Via header"""
        branch = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]
        logger.debug(f"Generated branch: {branch}", level=8)
        return branch
    
    def _generate_tag(self) -> str:
        """Generate a unique tag parameter for From/To headers"""
        tag = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]
        logger.debug(f"Generated tag: {tag}", level=8)
        return tag
    
    def add_message_listener(self, callback: Callable[[SIPMessage], None]) -> None:
        """Add a callback function to be called when messages are received"""
        self.message_listeners.append(callback)
    
    async def disconnect(self) -> None:
        """Disconnect from the OpenSIPS WebSocket server"""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from OpenSIPS WebSocket server")
