"""
OpenClay Memory Poisoning Prevention.
Scans memory reads/writes to prevent RAG poisoning and context injection.
"""

from typing import Any, List, Optional
from .shields import Shield
from .tracing import Trace

class MemoryWriteBlocked(Exception):
    """Raised when data attempting to be written to memory contains a threat."""
    def __init__(self, trace: Trace):
        self.trace = trace
        super().__init__(f"Memory Write Blocked ({trace.layer}): {trace.reason}")

class MemoryReadBlocked(Exception):
    """Raised when data retrieved from memory contains a threat (RAG poisoning)."""
    def __init__(self, trace: Trace):
        self.trace = trace
        super().__init__(f"Memory Read Blocked ({trace.layer}): {trace.reason}")

class ClayMemory:
    """
    Secure memory storage that prevents poisoning.
    Scans data before it is written and before it is read back into the agent context.
    
    Subclass `_write` and `_read` to integrate with real vector databases like Chroma or Pinecone.
    """
    
    def __init__(self, shield: Optional[Shield] = None):
        # By default, use a balanced shield if none is provided
        self.shield = shield or Shield.balanced()
        self._store = [] # Simple in-memory list for the base implementation
        
    def save(self, data: Any, context: str = "") -> None:
        """
        Scans data before writing it to storage.
        Raises MemoryWriteBlocked if a threat is detected.
        """
        stringified_data = str(data)
        
        # Scan the data being saved as if it were user input, 
        # looking for embedded prompt injections or exploits before they enter the DB.
        result = self.shield.protect_input(
            user_input=stringified_data, 
            system_context=context
        )
        
        if result["blocked"]:
            trace = Trace(
                blocked=True,
                layer="memory_write",
                reason=result["reason"],
                threat_level=result["threat_level"],
                rule=None,
                threat_breakdown=result.get("threat_breakdown"),
                latency_ms=0.0,
                input_result=result,
                output_result=None,
                recommendation="Review the data attempting to be saved."
            )
            raise MemoryWriteBlocked(trace)
            
        # If safe, write to the underlying store
        self._write(data)
        
    def _write(self, data: Any) -> None:
        """Override this method to write to a real database."""
        self._store.append(data)
        
    def recall(self, query: str = "") -> List[Any]:
        """
        Retrieves data and sanitizes context before injecting it into the prompt.
        Raises MemoryReadBlocked if retrieved data contains a threat.
        """
        raw_results = self._read(query)
        safe_results = []
        
        # In this base class we evaluate each result individually.
        for item in raw_results:
            stringified_item = str(item)
            
            # The retrieved item acts as a potential input threat (RAG indirect injection)
            result = self.shield.protect_input(
                user_input=stringified_item,
                system_context="Data retrieved from memory."
            )
            
            if result["blocked"]:
                trace = Trace(
                    blocked=True,
                    layer="memory_read",
                    reason=result["reason"],
                    threat_level=result["threat_level"],
                    rule=None,
                    threat_breakdown=result.get("threat_breakdown"),
                    latency_ms=0.0,
                    input_result=result,
                    output_result=None,
                    recommendation="RAG poisoning detected. Data blocked from reaching agent context."
                )
                raise MemoryReadBlocked(trace)
                
            safe_results.append(item)
            
        return safe_results
        
    def _read(self, query: str) -> List[Any]:
        """Override this method to read from a real database."""
        return self._store
