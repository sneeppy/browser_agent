"""
LangGraph State Schema Definition

Defines the state that flows through the graph, including:
- Task information
- Browser state
- Execution history
- Error tracking
- Token usage
"""

from typing import TypedDict, Annotated, List, Dict, Optional, Any
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class AgentState(TypedDict):
    """
    Main state schema for the browser agent graph.
    
    This state is passed between nodes and persists across graph execution.
    """
    # Task & Planning
    user_task: str  # Original user instruction
    current_plan: Optional[str]  # Current high-level plan
    next_steps: List[str]  # List of next actions to take
    current_step: Optional[str]  # Currently executing step
    
    # Browser State
    current_url: Optional[str]  # Current page URL
    last_screenshot_path: Optional[str]  # Path to last screenshot
    page_summary: Optional[str]  # Summarized page content (not full HTML)
    browser_context_id: Optional[str]  # Identifier for browser context
    
    # Execution History
    task_history: Annotated[List[Dict[str, Any]], operator.add]  # History of actions taken (use operator.add, NOT add_messages)
    messages: Annotated[List[BaseMessage], add_messages]  # LLM conversation history
    
    # Error Handling
    last_error: Optional[str]  # Last error encountered
    retry_count: int  # Number of retries for current step
    max_retries: int  # Maximum retries allowed
    
    # Security & Human-in-the-loop
    requires_confirmation: bool  # Whether action requires user confirmation
    confirmation_pending: bool  # Whether waiting for user confirmation
    pending_action: Optional[Dict[str, Any]]  # Action awaiting confirmation
    
    # Token & Resource Management
    token_usage: Dict[str, int]  # Track token usage per LLM call
    total_tokens: int  # Cumulative token count
    
    # Control Flow
    should_continue: bool  # Whether to continue execution
    should_retry: bool  # Whether to retry current step
    should_escalate: bool  # Whether to escalate to human
    
    # Node Routing
    next_node: Optional[str]  # Next node to execute (for conditional routing)
    
    # Results
    extracted_data: Dict[str, Any]  # Data extracted during execution
    final_result: Optional[str]  # Final result summary
    completion_status: Optional[str]  # "success", "failed", "interrupted", "pending"
    
    # Metadata
    session_id: str  # Unique session identifier
    start_time: Optional[float]  # Timestamp when task started
    last_update_time: Optional[float]  # Timestamp of last state update
    extraction_count: int  # Number of times extractor has been called (to prevent loops)
    step_count: int  # Total number of steps executed (to prevent infinite loops)

    # Anti-loop / progress tracking (generic)
    last_progress_signature: Optional[str]  # Hash/signature of last observed page context
    stagnation_count: int  # How many consecutive steps without progress
    last_action_key: Optional[str]  # Normalized key of last action (e.g., click label)
    repeated_action_count: int  # How many times the same action repeated consecutively