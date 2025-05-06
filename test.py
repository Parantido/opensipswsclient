"""
Modified authentication code for OpenSIPS WebSocket
"""

import asyncio
import logging
import hashlib
import random
import string
import uuid
import re
import websockets

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenSIPSAuth:
    """Specialized class to handle OpenSIPS authentication"""

    def __init__(self, username, password, domain):
        self.username = username
        self.password = password
        self.domain = domain

    def _generate_cnonce(self, length=16):
        """Generate a random cnonce value"""
        return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

    def parse_www_authenticate(self, header):
        """Parse the WWW-Authenticate header from a SIP response"""
        auth_params = {}

        # Extract the scheme
        if header.startswith('Digest '):
            auth_params['scheme'] = 'Digest'
            header = header[7:]  # Remove 'Digest ' prefix

        # Parse parameters
        for param in re.finditer(r'([^=,\s]+)=(?:"([^"]*)"|([^,\s]*))', header):
            key = param.group(1)
            value = param.group(2) if param.group(2) is not None else param.group(3)
            auth_params[key] = value

        return auth_params

    def calculate_response(self, auth_params, method):
        """Calculate the digest response"""
        realm = auth_params.get('realm', '')
        nonce = auth_params.get('nonce', '')
        qop = 'auth'  # Use 'auth' QoP as indicated in the OpenSIPS route

        # Generate various hash components
        ha1 = hashlib.md5(f"{self.username}:{realm}:{self.password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:sip:{self.domain}".encode()).hexdigest()

        # Generate cnonce and nc
        cnonce = self._generate_cnonce()
        nc = "00000001"

        # Calculate response with QoP
        response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()).hexdigest()

        # Return the auth dictionary
        return {
            'realm': realm,
            'nonce': nonce,
            'username': self.username,
            'uri': f"sip:{self.domain}",
            'response': response,
            'algorithm': auth_params.get('algorithm', 'MD5'),
            'qop': qop,
            'cnonce': cnonce,
            'nc': nc
        }

    def create_authorization_header(self, auth_params, method):
        """Create a complete Authorization header value"""
        response_params = self.calculate_response(auth_params, method)

        # Format the header exactly as in the successful example
        auth_header = 'Digest algorithm=MD5'

        # Add the remaining parameters
        auth_header += f', username="{self.username}"'
        auth_header += f', realm="{response_params["realm"]}"'
        auth_header += f', nonce="{response_params["nonce"]}"'
        auth_header += f', uri="sip:{self.domain}"'
        auth_header += f', response="{response_params["response"]}"'
        auth_header += f', qop={response_params["qop"]}'
        auth_header += f', cnonce="{response_params["cnonce"]}"'
        auth_header += f', nc={response_params["nc"]}'

        return auth_header

    async def register_with_opensips(ws_uri, username, password, domain):
        """Handle the SIP registration with OpenSIPS"""

        # Headers needed for WebSocket connection
        headers = {
            'Origin': 'https://localhost',
            'Sec-WebSocket-Protocol': 'sip'
        }

        try:
            # Connect to the WebSocket server
            connection = await websockets.connect(
                ws_uri,
                additional_headers=headers,
                subprotocols=['sip']
            )
            logger.info("Connected to OpenSIPS WebSocket server")

            # Create auth handler
            auth_handler = OpenSIPSAuth(username, password, domain)

            # Generate unique identifiers
            call_id = str(uuid.uuid4())
            tag = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]

            # Create a random local address component for the Contact header
            local_addr = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))

            # Try multiple registration attempts (to handle nonce changes)
            max_attempts = 5
            current_attempt = 0
            cseq_number = 1

            while current_attempt < max_attempts:
                current_attempt += 1

                # Generate a new branch for each attempt
                branch = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]

                if cseq_number == 1:
                    # Initial REGISTER request (without auth)
                    register_request = f"""REGISTER sip:{domain} SIP/2.0\r
Via: SIP/2.0/WSS {domain};branch=z9hG4bK{branch}\r
From: "Test User" <sip:{username}@{domain}>;tag={tag}\r
To: <sip:{username}@{domain}>\r
Call-ID: {call_id}\r
CSeq: {cseq_number} REGISTER\r
Contact: <sip:{local_addr}@{domain};transport=ws>;expires=3600\r
Allow: ACK,CANCEL,INVITE,MESSAGE,BYE,OPTIONS,INFO,NOTIFY,REFER\r
Supported: outbound, path, gruu\r
Max-Forwards: 70\r
User-Agent: Python OpenSIPS WebSocket Client\r
Content-Length: 0\r
\r
"""
                    logger.debug(f"Sending initial REGISTER request (attempt {current_attempt}):\n{register_request}")
                else:
                    # We already have an auth challenge, build an authenticated request
                    register_request = f"""REGISTER sip:{domain} SIP/2.0\r
Via: SIP/2.0/WSS {domain};branch=z9hG4bK{branch}\r
From: "Test User" <sip:{username}@{domain}>;tag={tag}\r
To: <sip:{username}@{domain}>\r
Call-ID: {call_id}\r
CSeq: {cseq_number} REGISTER\r
Contact: <sip:{local_addr}@{domain};transport=ws>;expires=3600\r
Authorization: {auth_header}\r
Allow: ACK,CANCEL,INVITE,MESSAGE,BYE,OPTIONS,INFO,NOTIFY,REFER\r
Supported: outbound, path, gruu\r
Max-Forwards: 70\r
User-Agent: Python OpenSIPS WebSocket Client\r
Content-Length: 0\r
\r
"""
                    logger.debug(f"Sending authenticated REGISTER request (attempt {current_attempt}):\n{register_request}")

                # Send the request
                await connection.send(register_request)

                # Wait for response
                response = await connection.recv()
                logger.debug(f"Received response:\n{response}")

                # Check if it's a 200 OK (success)
                if "200 OK" in response:
                    logger.info(f"Registration successful on attempt {current_attempt}!")
                    return True

                # Check if it's a 401 Unauthorized
                elif "401 Unauthorized" in response:
                    # Extract WWW-Authenticate header for the next attempt
                    www_auth_match = re.search(r'WWW-Authenticate: (.*?)\r\n', response)
                    if www_auth_match:
                        www_auth_header = www_auth_match.group(1)
                        logger.debug(f"WWW-Authenticate header: {www_auth_header}")

                        # Parse auth params
                        auth_params = auth_handler.parse_www_authenticate(www_auth_header)
                        logger.debug(f"Parsed auth params: {auth_params}")

                        # Create authorization header for the next attempt
                        auth_header = auth_handler.create_authorization_header(auth_params, "REGISTER")
                        logger.debug(f"Generated Authorization header: {auth_header}")

                        # Increment CSeq for the next request
                        cseq_number += 1

                        # Continue to the next attempt
                        continue
                    else:
                        logger.error("No WWW-Authenticate header found in 401 response")
                        break

                # Any other response
                else:
                    logger.error(f"Unexpected response: {response.splitlines()[0]}")
                    break

            # If we got here, all attempts failed
            logger.error(f"Registration failed after {current_attempt} attempts")

            # Close connection
            await connection.close()
            return False

        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return False

async def test_register():
    """Test function to register with OpenSIPS"""
    success = await register_with_opensips(
        ws_uri="wss://ats-demo.call-matrix.com:6061",
        username="test",
        password="l3tMe.Test",
        domain="test.com"
    )

    if success:
        print("Registration successful!")
    else:
        print("Registration failed!")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_register())
