#!/usr/bin/env python3
"""
Clawdbot Gateway Client for Pibot Voice
Sends messages to Clawdbot and receives responses
"""

import asyncio
import aiohttp
import sys
from typing import Optional


class ClawdbotClient:
    """Client for Clawdbot Gateway API"""
    
    def __init__(
        self,
        gateway_url: str = "http://localhost:18789",
        timeout: int = 60
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
    
    async def send_message(self, message: str, session_key: str = None) -> Optional[str]:
        """
        Send a message to Clawdbot and get response
        
        Args:
            message: User message to send
            session_key: Optional session key (default: main session)
        
        Returns:
            Assistant response text or None on error
        """
        url = f"{self.gateway_url}/v1/chat/completions"
        
        payload = {
            "model": "default",
            "messages": [
                {"role": "user", "content": message}
            ],
            "stream": False
        }
        
        if session_key:
            payload["session_key"] = session_key
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._extract_response(data)
                    else:
                        error = await response.text()
                        print(f"❌ Gateway error {response.status}: {error}")
                        return None
                        
        except asyncio.TimeoutError:
            print(f"❌ Gateway timeout after {self.timeout}s")
            return None
        except aiohttp.ClientError as e:
            print(f"❌ Connection error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def _extract_response(self, data: dict) -> Optional[str]:
        """Extract assistant message from API response"""
        try:
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "").strip()
        except (KeyError, IndexError):
            pass
        return None
    
    async def check_health(self) -> bool:
        """Check if gateway is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.gateway_url}/health") as response:
                    return response.status == 200
        except:
            return False
    
    def send_message_sync(self, message: str, session_key: str = None) -> Optional[str]:
        """Synchronous wrapper for send_message()"""
        return asyncio.run(self.send_message(message, session_key))


async def interactive_mode(client: ClawdbotClient):
    """Interactive chat mode"""
    print("💬 Interactive mode (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            
            print("🤔 Thinking...")
            response = await client.send_message(user_input)
            
            if response:
                print(f"\n🤖 Pibot: {response}")
            else:
                print("\n❌ No response")
                
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except EOFError:
            break


async def main():
    """CLI test"""
    client = ClawdbotClient()
    
    # Check health first
    print("🔍 Checking Clawdbot Gateway...")
    healthy = await client.check_health()
    
    if not healthy:
        print("❌ Gateway not available at", client.gateway_url)
        print("   Make sure Clawdbot is running: clawdbot gateway start")
        return
    
    print("✅ Gateway is healthy")
    
    if len(sys.argv) > 1:
        # Single message mode
        message = " ".join(sys.argv[1:])
        print(f"\n📤 Sending: {message}")
        response = await client.send_message(message)
        if response:
            print(f"\n📥 Response: {response}")
        else:
            print("\n❌ No response")
    else:
        # Interactive mode
        await interactive_mode(client)


if __name__ == "__main__":
    asyncio.run(main())
