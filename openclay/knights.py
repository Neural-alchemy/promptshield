"""
OpenClay Knights & Squads.
Minimal, secure-by-default autonomous entity primitives.
"""

from typing import List, Any, Callable, Optional, Dict
from .shields import Shield
from .runtime import ClayRuntime, ClayResult
from .memory import ClayMemory

class Knight:
    """
    A minimal, secure-by-default autonomous entity primitive.
    """
    def __init__(
        self, 
        name: str, 
        llm_caller: Callable, 
        tools: Optional[List[Callable]] = None, 
        shield: Optional[Shield] = None, 
        memory: Optional[ClayMemory] = None, 
        trust: str = "untrusted"
    ):
        self.name = name
        self.llm_caller = llm_caller
        self.tools = tools or []
        self.trust = trust
        
        # Trust level dictates default shield if none provided
        if not shield:
            if trust == "untrusted":
                self.shield = Shield.strict()
            else:
                self.shield = Shield.balanced()
        else:
            self.shield = shield
            
        self.memory = memory
        self.runtime = ClayRuntime(policy=self.shield)
        
    def run(self, input_text: str, context: str = "") -> ClayResult:
        """
        Executes the knight's core logic securely.
        Data flows: Input -> Memory Recall -> Runtime(LLM) -> Memory Save -> Output
        """
        # 1. Pre-execution memory recall
        retrieved_context = ""
        if self.memory:
            safe_results = self.memory.recall(input_text)
            if safe_results:
                retrieved_context = "\n".join(str(r) for r in safe_results)
                
        full_context = context
        if retrieved_context:
            full_context = f"{context}\n\n[Retrieved Memory Context]\n{retrieved_context}"
            
        # 2. Secure Execution via ClayRuntime
        # Bind the context so the user's llm_caller receives it
        def _bound_llm_caller(text: str):
            return self.llm_caller(text, context=full_context)
            
        result = self.runtime.run(
            _bound_llm_caller,
            input_text,
            context=full_context
        )
        
        # 3. Post-execution memory save
        if not result.blocked and self.memory and result.output is not None:
            self.memory.save({
                "input": input_text,
                "output": result.output,
                "knight": self.name
            })
            
        return result

class Squad:
    """
    Orchestrates multiple Knights securely.
    The entire squad operates under a master shield to prevent Knights 
    from poisoning each other during collaboration.
    """
    def __init__(self, knights: List[Knight], shield: Optional[Shield] = None):
        self.knights: Dict[str, Knight] = {k.name: k for k in knights}
        self.shield = shield or Shield.secure()
        self.runtime = ClayRuntime(policy=self.shield)
        
    def deploy(self, task: str, workflow_fn: Callable) -> ClayResult:
        """
        Deploys the squad on a task using a custom workflow function.
        workflow_fn(knights_dict, task) -> result
        """
        def _execute_squad(task_str: str):
            return workflow_fn(self.knights, task_str)
            
        # The entire multi-knight deployment is run inside the master runtime
        return self.runtime.run(_execute_squad, task)
