"""
LangGraph Graph Definition

Builds the complete graph using LangGraph Graph API with nodes, edges, and conditional routing.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import os

# Try to import RedisSaver, but make it optional
try:
    from langgraph.checkpoint.redis import RedisSaver
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisSaver = None

from src.graph.state import AgentState
from src.graph.nodes import (
    coordinator_node, planner_node, executor_node, extractor_node, error_handler_node, set_llms, set_observer
)
from src.graph.edges import (
    route_after_coordinator, route_after_executor, route_after_planner,
    route_after_extractor, route_after_error_handler
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_graph(primary_llm, fallback_llm, observer=None) -> StateGraph:
    """
    Build and compile the LangGraph state machine.
    
    Args:
        primary_llm: Primary LLM (unified API service)
        fallback_llm: Fallback LLM (same as primary for unified API)
        observer: ConsoleAgentObserver for detailed logging
        
    Returns:
        Compiled LangGraph graph
    """
    # Set LLMs and observer for nodes
    set_llms(primary_llm, fallback_llm)
    set_observer(observer)
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Set entry point
    workflow.set_entry_point("coordinator")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "planner": "planner",
            "executor": "executor",
            "extractor": "extractor",
            "error_handler": "error_handler",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "coordinator": "coordinator",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "coordinator": "coordinator",
            "error_handler": "error_handler",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "extractor",
        route_after_extractor,
        {
            "coordinator": "coordinator",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "error_handler",
        route_after_error_handler,
        {
            "executor": "executor",
            "planner": "planner",
            "coordinator": "coordinator",
            "END": END
        }
    )
    
    # Compile graph with checkpointing
    use_redis = os.getenv("USE_REDIS", "false").lower() == "true"
    
    if use_redis and REDIS_AVAILABLE:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            checkpointer = RedisSaver(redis_url)
            logger.info("Using Redis checkpointing")
        except Exception as e:
            logger.warning(f"Redis checkpointing failed, using memory: {e}")
            checkpointer = MemorySaver()
    else:
        if use_redis and not REDIS_AVAILABLE:
            logger.warning("Redis checkpointing requested but RedisSaver not available, using memory")
        checkpointer = MemorySaver()
        logger.info("Using memory checkpointing")
    
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Graph compiled successfully")
    return app


def create_initial_state(user_task: str, session_id: str, current_url: str = None) -> Dict[str, Any]:
    """
    Create initial state for graph execution.
    
    Args:
        user_task: User's natural language instruction
        session_id: Unique session identifier
        current_url: Optional current browser URL (for continuing from existing page)
        
    Returns:
        Initial state dict
    """
    import time
    
    return {
        "user_task": user_task,
        "current_plan": None,
        "next_steps": [],
        "current_step": None,
        "current_url": current_url,
        "last_screenshot_path": None,
        "page_summary": None,
        "browser_context_id": None,
        "task_history": [],
        "messages": [],
        "last_error": None,
        "retry_count": 0,
        "max_retries": 3,
        "requires_confirmation": False,
        "confirmation_pending": False,
        "pending_action": None,
        "token_usage": {},
        "total_tokens": 0,
        "should_continue": True,
        "should_retry": False,
        "should_escalate": False,
        "next_node": "planner",
        "extracted_data": {},
        "final_result": None,
        "completion_status": "pending",
        "session_id": session_id,
        "start_time": time.time(),
        "last_update_time": time.time(),
        "extraction_count": 0,
        "step_count": 0,

        # Anti-loop / progress tracking (generic)
        "last_progress_signature": None,
        "stagnation_count": 0,
        "last_action_key": None,
        "repeated_action_count": 0
    }
