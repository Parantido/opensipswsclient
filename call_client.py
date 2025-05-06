"""
Enhanced Debug OpenSIPS WebSocket Test Client:
1. Connects to OpenSIPS WebSocket
2. Registers with the server using improved authentication handling
3. Makes a call to a static phone number
4. Plays an audio file when the call is answered
5. Hangs up if not answered within 30 seconds

Supports debug levels 1-9 for detailed SIP message logging
"""

import asyncio
import argparse
import logging
import sys
import os
from opensips_ws_lib import OpenSIPSClient, SIPMessage, SIPAuthHelper, AudioHandler, SDPGenerator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for immediate visibility
)

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='OpenSIPS WebSocket Call Client')
    parser.add_argument('--server', required=True, help='OpenSIPS WebSocket server URI (e.g., wss://example.com:8080/ws)')
    parser.add_argument('--username', required=True, help='SIP username')
    parser.add_argument('--password', required=True, help='SIP password')
    parser.add_argument('--domain', required=True, help='SIP domain')
    parser.add_argument('--destination', required=True, help='Destination phone number to call')
    parser.add_argument('--audio-file', required=True, help='Audio file to play when call is answered (WAV format)')
    parser.add_argument('--timeout', type=int, default=30, help='Call timeout in seconds (default: 30)')
    parser.add_argument('--debug-level', type=int, default=1, choices=range(1, 10), 
                        help='Debug level 1-9, with 9 being most verbose (default: 1)')
    
    args = parser.parse_args()
    
    # Set log level based on debug level
    if args.debug_level >= 5:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Check if audio file is WAV format
    if not args.audio_file.lower().endswith('.wav'):
        print("Error: Only WAV audio files are currently supported")
        sys.exit(1)
    
    print(f"Starting OpenSIPS WebSocket client with debug level {args.debug_level}")
    print(f"Server: {args.server}")
    print(f"Username: {args.username}")
    print(f"Domain: {args.domain}")
    
    # Create the OpenSIPS client with the specified debug level
    client = OpenSIPSClient(
        server_uri=args.server,
        username=args.username,
        password=args.password,
        domain=args.domain,
        debug_level=args.debug_level
    )
    
    try:
        # Connect to the WebSocket server
        print(f"Connecting to OpenSIPS WebSocket server: {args.server}")
        connected = await client.connect()
        
        if not connected:
            print("Error: Failed to connect to OpenSIPS WebSocket server")
            return
        
        # Register with the SIP server
        print("Registering with SIP server...")
        registered = await client.register()
        
        if not registered:
            print("Error: Failed to register with SIP server")
            return
        
        print("Successfully registered with SIP server")
        
        # Place a call to the destination with timeout
        print(f"Placing call to {args.destination} (timeout: {args.timeout}s)...")
        result = await client.place_call(args.destination, args.timeout)
        
        if result["success"]:
            print(f"Call answered! Call-ID: {result['call_id']}")
            
            # Play audio file
            print(f"Playing audio file: {args.audio_file}")
            await client.play_audio_to_call(result["call_id"], args.audio_file)
            
            # Wait for audio to finish (this is a simplified approach)
            # In a real implementation, you'd want to monitor the call status
            # and handle various events properly
            await asyncio.sleep(10)  # Assume 10 seconds is enough for the audio
            
            # End the call
            print("Ending call...")
            await client.end_call(result["call_id"])
            
        else:
            print(f"Call was not answered. Status: {result['state']}")
            if result['state'] != 'timeout':  # If not already timed out and hung up
                print("Hanging up call...")
                await client.end_call(result["call_id"])
        
        # Wait a moment before disconnecting
        await asyncio.sleep(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Disconnect from the server
        if hasattr(client, 'disconnect') and client.connection:
            await client.disconnect()
        print("Exiting...")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
