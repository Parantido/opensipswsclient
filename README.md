# OpenSIPS WebSocket Client

A Python library for interacting with OpenSIPS via WebSockets. This library provides a pure Python implementation with no external SIP dependencies for establishing WebSocket connections to OpenSIPS servers, handling SIP registration, making calls, and playing audio.

## Features

- WebSocket connection to OpenSIPS server
- SIP registration with digest authentication
- Making outbound calls
- Playing audio files to active calls
- Ending calls
- Detailed debug logging with configurable levels

## Installation

1. Clone this repository or download the source code.

2. Install the required dependencies:

```bash
pip install websockets pyaudio
```

## Usage

### Basic Usage

```python
import asyncio
from opensips_ws_lib import OpenSIPSClient

async def main():
    # Create client instance
    client = OpenSIPSClient(
        server_uri="wss://your-opensips-server:6061",
        username="your_username",
        password="your_password",
        domain="your_domain"
    )
    
    # Connect to server
    connected = await client.connect()
    if not connected:
        print("Failed to connect")
        return
    
    # Register with server
    registered = await client.register()
    if not registered:
        print("Failed to register")
        return
    
    print("Successfully registered")
    
    # Place a call
    result = await client.place_call("destination_number", timeout=30)
    
    if result["success"]:
        print(f"Call answered! Call-ID: {result['call_id']}")
        
        # Play audio file
        await client.play_audio_to_call(result["call_id"], "your_audio.wav")
        
        # Wait for audio to finish
        await asyncio.sleep(10)
        
        # End the call
        await client.end_call(result["call_id"])
    else:
        print(f"Call was not answered. Status: {result['state']}")
    
    # Disconnect
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### Command Line Client

The library includes a command-line client for testing:

```bash
python call_client.py --server wss://your-opensips-server:6061 \
                      --username your_username \
                      --password your_password \
                      --domain your_domain \
                      --destination destination_number \
                      --audio-file your_audio.wav \
                      --debug-level 5
```

## Debug Levels

The client supports 9 debug levels for detailed logging:

- **Level 1**: Basic information
- **Level 2-3**: Connection details and message routing
- **Level 4**: Authentication parameters and headers
- **Level 5**: Complete SIP messages (packets)
- **Level 6-8**: Internal calculations
- **Level 9**: Most detailed debugging, including authentication hash calculations

## Core Classes

### OpenSIPSClient

Main client for interacting with OpenSIPS WebSocket server:

```python
client = OpenSIPSClient(
    server_uri="wss://your-opensips-server:6061",
    username="your_username",
    password="your_password",
    domain="your_domain",
    debug_level=1  # Optional
)
```

### SIPMessage

Represents a SIP message with methods for parsing and generating SIP messages:

```python
message = SIPMessage(
    method="REGISTER",  # For requests
    status_code=200,    # For responses
    reason="OK",        # For responses
    headers={},
    content=""
)
```

### AudioHandler

Handles playing audio to SIP calls:

```python
audio_handler = AudioHandler(
    local_port=10000,
    remote_ip="192.168.1.1",
    remote_port=10000
)
audio_handler.play_audio_file("audio.wav")
```

## SIP Authentication

The library implements SIP digest authentication according to RFC 2617, with support for:

- Authentication challenges with nonce and realm
- MD5 digest calculation
- Proper CSeq handling

## Requirements

- Python 3.8+
- websockets library
- pyaudio library
- WAV audio files for playing to calls

## Limitations

- Currently only supports WebSocket transport
- Audio handling is limited to playing WAV files
- No support for incoming calls
- Limited codec support

## License

This project is licensed under the MIT License - see the LICENSE file for details.
