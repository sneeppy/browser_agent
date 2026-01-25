"""
LangGraph Conditional Edges

Defines routing logic between nodes based on state.
"""

from typing import Literal
from src.graph.state import AgentState
from src.utils.logging import get_logger

logger = get_logger(__name__)


def route_after_coordinator(state: AgentState) -> Literal["planner", "executor", "extractor", "error_handler", "END"]:
    """
    Route after coordinator node based on next_node in state.
    """
    next_node = state.get("next_node")
    
    if next_node == "planner":
        return "planner"
    elif next_node == "executor":
        return "executor"
    elif next_node == "extractor":
        return "extractor"
    elif next_node == "error_handler":
        return "error_handler"
    elif state.get("completion_status") in ["success", "failed", "interrupted"]:
        return "END"
    else:
        # Default to planner
        return "planner"


def route_after_executor(state: AgentState) -> Literal["coordinator", "error_handler", "END"]:
    """
    Route after executor: check for errors or continue.
    """
    if state.get("last_error") and not state.get("should_retry"):
        return "error_handler"
    
    if state.get("confirmation_pending"):
        # Wait for confirmation - stay in coordinator
        return "coordinator"
    
    if state.get("completion_status") in ["success", "failed"]:
        return "END"
    
    # Check for infinite loops - if we've been executing too many times without progress
    task_history = state.get("task_history", [])
    step_count = state.get("step_count", 0)
    
    if step_count > 50:
        # Too many steps - force end
        logger.error(f"[EDGE] Maximum step count ({step_count}) reached, forcing END")
        return "END"
    
    if len(task_history) > 20:
        # Too many actions without completion - might be stuck
        logger.warning(f"[EDGE] Too many actions ({len(task_history)}), checking for completion")
        if state.get("completion_status") == "pending" and not state.get("next_steps"):
            # No next steps and still pending - might need to end
            logger.warning("[EDGE] No next steps and still pending, forcing END")
            return "END"
    
    # Continue to coordinator for next routing decision
    return "coordinator"


def route_after_planner(state: AgentState) -> Literal["coordinator", "END"]:
    """
    Route after planner: always go back to coordinator.
    """
    if state.get("completion_status") in ["success", "failed"]:
        return "END"
    return "coordinator"


def route_after_extractor(state: AgentState) -> Literal["coordinator", "END"]:
    """
    Route after extractor: check if task is complete.
    """
    if state.get("completion_status") == "success":
        return "END"
    
    if not state.get("should_continue"):
        return "END"
    
    return "coordinator"


def route_after_error_handler(state: AgentState) -> Literal["executor", "planner", "coordinator", "END"]:
    """
    Route after error handler based on decision.
    """
    if state.get("should_escalate"):
        return "END"
    
    if state.get("should_retry"):
        return "executor"
    
    if state.get("next_node") == "planner":
        return "planner"
    
    return "coordinator"


