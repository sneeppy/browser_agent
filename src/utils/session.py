"""
Browser session management with encryption

Handles saving/loading browser state (cookies, localStorage) with encryption.
"""

import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages encrypted browser session storage."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize session manager.
        
        Args:
            encryption_key: Base64-encoded encryption key. If None, uses ENV var or generates new.
        """
        self.sessions_dir = Path("sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Get or generate encryption key
        if encryption_key:
            key = encryption_key.encode()
        else:
            key_str = os.getenv("ENCRYPTION_KEY")
            if key_str:
                key = key_str.encode()
            else:
                # Generate a new key (in production, this should be set via env)
                logger.warning("No encryption key found. Generating new key (not secure for production!)")
                key = Fernet.generate_key()
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'browser_agent_salt',  # In production, use random salt per session
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key))
        self.cipher = Fernet(derived_key)
    
    def save_session(self, session_id: str, storage_state: Dict[str, Any]) -> str:
        """
        Save encrypted browser session state.
        
        Args:
            session_id: Unique session identifier
            storage_state: Playwright storage state dict
            
        Returns:
            Path to saved session file
        """
        session_file = self.sessions_dir / f"{session_id}.encrypted"
        
        # Serialize and encrypt
        json_data = json.dumps(storage_state)
        encrypted_data = self.cipher.encrypt(json_data.encode())
        
        # Save to file
        session_file.write_bytes(encrypted_data)
        
        logger.info(f"Saved session {session_id} to {session_file}")
        return str(session_file)
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load and decrypt browser session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Storage state dict or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.encrypted"
        
        if not session_file.exists():
            logger.warning(f"Session file not found: {session_file}")
            return None
        
        try:
            # Read and decrypt
            encrypted_data = session_file.read_bytes()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            storage_state = json.loads(decrypted_data.decode())
            
            logger.info(f"Loaded session {session_id} from {session_file}")
            return storage_state
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def list_sessions(self) -> list[str]:
        """List all available session IDs."""
        return [f.stem for f in self.sessions_dir.glob("*.encrypted")]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session file.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        session_file = self.sessions_dir / f"{session_id}.encrypted"
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Deleted session {session_id}")
            return True
        return False
