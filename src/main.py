#!/usr/bin/env python3
"""
Browser Agent - Console Edition with Full LLM Reasoning Visibility

A fully autonomous AI web browser agent that shows EVERY step of LLM reasoning,
agent decision-making, and execution processes in real-time console output.

Features:
- Complete LLM reasoning visibility
- Agent activity tracking
- Step-by-step execution logging
- Tool execution details
- State transitions
- Error analysis and recovery

Usage:
    python src/main.py                    # Interactive mode
    python src/main.py "your task here"   # Single task mode
"""

import asyncio
import sys
import os
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.graph.graph import build_graph, create_initial_state
from src.tools.browser_tools import set_browser_context, set_current_task
from src.utils.session import SessionManager

# Load environment
load_dotenv()

# Create logs directory BEFORE configuring logging
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/browser_agent.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class ConsoleAgentObserver:
    """Observes and logs all agent activities with maximum detail."""

    def __init__(self):
        self.start_time = time.time()
        self.step_counter = 0
        self.current_agent = None
        self.llm_call_count = 0
        self.tool_call_count = 0

    def log_separator(self, title: str = "", char: str = "=", length: int = 80):
        """Log a visual separator."""
        if title:
            side_len = (length - len(title) - 2) // 2
            separator = char * side_len + f" {title} " + char * side_len
            if len(separator) < length:
                separator += char
        else:
            separator = char * length
        logger.info(separator)

    def log_llm_reasoning(self, messages: List, agent_name: str):
        """Log complete LLM reasoning process."""
        self.llm_call_count += 1

        self.log_separator(f"[BRAIN] LLM CALL #{self.llm_call_count} - {agent_name.upper()}", "-")

        for i, msg in enumerate(messages[-3:], 1):  # Show last 3 messages
            msg_type = "[AI]" if isinstance(msg, AIMessage) else "[USER]"
            role = "ASSISTANT" if isinstance(msg, AIMessage) else "USER"

            content = msg.content
            if len(content) > 500:
                content = content[:500] + "..."

            logger.info(f"   {msg_type} {role}: {content}")

        self.log_separator("", "-")

    def log_agent_activation(self, agent_name: str, context: Dict[str, Any]):
        """Log when an agent becomes active."""
        self.current_agent = agent_name
        self.step_counter += 1

        self.log_separator(f"[STEP {self.step_counter}] {agent_name.upper()} AGENT", "=")

        # Log context
        if "user_task" in context:
            logger.info(f"[TASK] {context['user_task']}")

        if "current_step" in context:
            logger.info(f"[ACTION] {context['current_step']}")

        if "current_plan" in context:
            plan = context["current_plan"]
            if len(plan) > 200:
                plan = plan[:200] + "..."
            logger.info(f"[PLAN] {plan}")

        if "next_steps" in context and context["next_steps"]:
            logger.info("[NEXT STEPS]:")
            for i, step in enumerate(context["next_steps"], 1):
                logger.info(f"   {i}. {step}")

    def log_tool_execution(self, tool_name: str, args: Dict, result: Dict):
        """Log tool execution with full details."""
        self.tool_call_count += 1

        status = "[SUCCESS]" if result.get("success", False) else "[FAILED]"
        logger.info(f"[TOOL #{self.tool_call_count}] {tool_name.upper()}")
        logger.info(f"   [INPUT] {json.dumps(args, indent=2, ensure_ascii=False)}")
        logger.info(f"   {status} {result}")

    def log_state_change(self, old_state: Dict, new_state: Dict):
        """Log significant state changes."""
        changes = []

        for key in ["current_url", "page_summary", "extracted_data", "last_error"]:
            if key in old_state and key in new_state and old_state[key] != new_state[key]:
                changes.append(f"{key}: {old_state[key]} → {new_state[key]}")

        if changes:
            logger.info("🔄 STATE CHANGES:")
            for change in changes:
                logger.info(f"   • {change}")

    def log_task_completion(self, status: str, result: str, duration: float):
        """Log task completion."""
        status_msg = {
            "success": "[COMPLETED SUCCESSFULLY]",
            "failed": "[FAILED]",
            "interrupted": "[INTERRUPTED]"
        }.get(status, "[UNKNOWN STATUS]")

        self.log_separator(f"TASK {status.upper()} {status_msg}", "=")

        if result:
            result_preview = result[:300] + "..." if len(result) > 300 else result
            logger.info(f"[RESULT] {result_preview}")

        logger.info(f"[DURATION] {duration:.2f}s")
        logger.info(f"[LLM CALLS] {self.llm_call_count}")
        logger.info(f"[TOOL CALLS] {self.tool_call_count}")
        logger.info(f"[TOTAL STEPS] {self.step_counter}")


class BrowserAgentConsole:
    """Console-based browser agent with maximum visibility."""
    
    def __init__(self):
        self.observer = ConsoleAgentObserver()
        self.llm = None
        self.graph = None
        self.browser = None
        self.context = None
        self.page = None
        self.session_manager = SessionManager()
        self.browser_initialized = False
        self._chrome_process = None  # For Chrome launched with CDP

        self._setup_logging()

    def _setup_logging(self):
        """Setup enhanced logging."""
        logger.info(">>> BROWSER AGENT CONSOLE EDITION")
        logger.info(">>> Maximum visibility mode: ON")
        logger.info(">>> LLM reasoning: VISIBLE")
        logger.info(">>> Agent activities: LOGGED")
        logger.info(">>> Tool executions: DETAILED")
        self.observer.log_separator("INITIALIZATION", "=")
    
    def initialize_llm(self):
        """Initialize LLM with detailed logging."""
        logger.info("[INIT] Initializing Language Model...")

        api_key = os.getenv("AGENTPLATFORM_KEY")
        model = os.getenv("LLM_MODEL", "openai/gpt-4o")
        api_base = os.getenv("LLM_API_BASE_URL", "https://litellm.tokengate.ru/v1")
        
        if not api_key:
            logger.error("[ERROR] AGENTPLATFORM_KEY not found in environment!")
            logger.error("[HINT] Please set AGENTPLATFORM_KEY in your .env file")
            sys.exit(1)

        logger.info(f"[MODEL] {model}")
        logger.info(f"[API] {api_base}")

        try:
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.7,
                api_key=api_key,
                base_url=api_base
            )

            # Test LLM
            logger.info("[TEST] Testing LLM connection...")
            test_msg = HumanMessage(content="Respond with exactly: 'LLM connection successful'")
            response = self.llm.invoke([test_msg])

            logger.info(f"[SUCCESS] LLM Response: {response.content.strip()}")
            logger.info("[SUCCESS] LLM initialized successfully")

        except Exception as e:
            logger.error(f"[ERROR] LLM initialization failed: {e}")
            sys.exit(1)

    def initialize_graph(self):
        """Initialize the agent graph."""
        logger.info("[GRAPH] Building agent graph...")

        # Mock fallback LLM (same as primary for unified API)
        fallback_llm = self.llm

        self.graph = build_graph(self.llm, fallback_llm, self.observer)
        logger.info("[SUCCESS] Agent graph compiled successfully")

        # Log agent structure
        logger.info("[AGENTS] Agent Architecture:")
        logger.info("   [COORDINATOR] - Routes tasks between agents")
        logger.info("   [PLANNER] - Creates execution plans")
        logger.info("   [EXECUTOR] - Performs browser actions")
        logger.info("   [EXTRACTOR] - Analyzes page content")
        logger.info("   [ERROR_HANDLER] - Manages failures")
    
    async def initialize_browser(self, session_id: str):
        """Initialize browser with detailed logging."""
        logger.info("[BROWSER] Initializing Playwright browser...")

        from playwright.async_api import async_playwright

        headless = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
        video_recording = os.getenv("BROWSER_VIDEO_RECORDING", "true").lower() == "true"
        use_chrome_profile = os.getenv("USE_CHROME_USER_PROFILE", "false").lower() == "true"
        chrome_profile_dir = os.getenv("CHROME_PROFILE_DIRECTORY", "Default")
        
        logger.info(f"[HEADLESS] {headless}")
        logger.info(f"[VIDEO] {video_recording}")
        logger.info(f"[CHROME PROFILE] {use_chrome_profile}")

        playwright = await async_playwright().start()

        # Try to launch Chrome first, fallback to Chromium
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Users\%USERNAME%\AppData\Local\Google\Chrome\Application\chrome.exe"
        ]

        chrome_launched = False
        chrome_executable = None
        for chrome_path in chrome_paths:
            expanded_path = os.path.expandvars(chrome_path)
            if os.path.exists(expanded_path):
                chrome_executable = expanded_path
                break

        if use_chrome_profile:
            import shutil
            
            # Use a separate profile directory for the agent
            agent_profile_dir = os.path.join(os.getcwd(), "browser_profile")
            agent_default_dir = os.path.join(agent_profile_dir, "Default")
            
            # Get Chrome's system profile path
            local_app_data = os.getenv("LOCALAPPDATA", "")
            chrome_default_dir = os.path.join(local_app_data, "Google", "Chrome", "User Data", "Default") if local_app_data else None
            
            # Copy cookies and login data from system Chrome (only once, if not already done)
            cookies_copied_flag = os.path.join(agent_profile_dir, ".cookies_copied")
            
            if chrome_default_dir and os.path.exists(chrome_default_dir) and not os.path.exists(cookies_copied_flag):
                logger.info("[BROWSER] Copying login data from your Chrome profile...")
                
                # Create agent profile directory
                os.makedirs(agent_default_dir, exist_ok=True)
                
                # Files to copy for preserving logins
                files_to_copy = ["Cookies", "Login Data", "Web Data", "Preferences"]
                
                for filename in files_to_copy:
                    src = os.path.join(chrome_default_dir, filename)
                    dst = os.path.join(agent_default_dir, filename)
                    if os.path.exists(src):
                        try:
                            shutil.copy2(src, dst)
                            logger.info(f"[BROWSER] Copied {filename}")
                        except Exception as e:
                            logger.warning(f"[BROWSER] Could not copy {filename}: {e}")
                
                # Mark as copied
                with open(cookies_copied_flag, "w") as f:
                    f.write("Cookies copied from Chrome profile")
                
                logger.info("[SUCCESS] Login data copied! You should be logged in to your accounts.")
            
            agent_profile_dir = os.path.join(os.getcwd(), "browser_profile")
            
            if not os.path.exists(agent_profile_dir):
                os.makedirs(agent_profile_dir, exist_ok=True)
                logger.info(f"[BROWSER] Created agent profile directory: {agent_profile_dir}")
            
            logger.info(f"[BROWSER] Using agent profile dir: {agent_profile_dir}")

            profile_args = [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled"
            ]
            if not headless:
                profile_args.append("--start-maximized")

            try:
                self.context = await playwright.chromium.launch_persistent_context(
                    agent_profile_dir,
                    headless=headless,
                    args=profile_args,
                    accept_downloads=True
                )
                self.browser = self.context.browser if hasattr(self.context, "browser") else None
                self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
                set_browser_context(self.browser, self.context, self.page)
                logger.info("[SUCCESS] Chromium with persistent profile launched")
                return
            except Exception as e:
                logger.warning(f"[WARNING] Failed to launch with profile: {e}")

        if chrome_executable:
            try:
                logger.info(f"[BROWSER] Attempting to launch Chrome from: {chrome_executable}")
                self.browser = await playwright.chromium.launch(
                    headless=headless,
                    executable_path=chrome_executable,
                    args=['--start-maximized', '--no-sandbox', '--disable-dev-shm-usage'] if not headless else []
                )
                logger.info("[SUCCESS] System Chrome launched successfully")
                chrome_launched = True
            except Exception as e:
                logger.warning(f"[WARNING] Failed to launch Chrome from {chrome_executable}: {e}")

        if not chrome_launched:
            logger.info("[BROWSER] Chrome not found, using Chromium (testing browser)")
            try:
                self.browser = await playwright.chromium.launch(
                    headless=headless,
                    args=['--start-maximized'] if not headless else []
                )
                logger.info("[SUCCESS] Chromium launched successfully")
            except Exception as e:
                logger.error(f"[ERROR] Failed to launch any browser: {e}")
                raise
        
        # Setup context
        context_options = {
            "viewport": {"width": 1920, "height": 1080},
            "record_video_dir": "recordings/videos" if video_recording else None,
            "record_video_size": {"width": 1920, "height": 1080} if video_recording else None
        }
        
        # Load existing session
        storage_state = self.session_manager.load_session(session_id)
        if storage_state:
            context_options["storage_state"] = storage_state
            logger.info(f"📁 Loaded session state for {session_id}")
        
        self.context = await self.browser.new_context(**context_options)
        self.page = await self.context.new_page()
        
        set_browser_context(self.browser, self.context, self.page)
        
        logger.info("[SUCCESS] Browser initialized successfully")
    
    async def run_task(self, task: str, session_id: str = None):
        """Run a single task with maximum visibility."""
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        task_start_time = time.time()

        logger.info("[EXEC] STARTING TASK EXECUTION")
        logger.info(f"[TASK] {task}")
        logger.info(f"[SESSION] {session_id}")
        self.observer.log_separator(f"TASK: {task[:50]}{'...' if len(task) > 50 else ''}", "=")
        
        # Set current task for context-aware element selection
        set_current_task(task)

        # Browser should already be initialized in run() method
        if not self.browser_initialized or not self.page:
            logger.error("[ERROR] Browser not initialized! This should not happen.")
            logger.warning("[WARNING] Attempting to initialize browser now...")
            try:
                await self.initialize_browser(session_id)
                self.browser_initialized = True
                logger.info("[SUCCESS] Browser initialized in run_task")
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize browser: {e}")
                raise

        # Create initial state with a unique task_id for each task
        # This prevents LangGraph from restoring old state between tasks
        task_id = f"{session_id}_{int(time.time())}"
        
        # Get current URL from browser so new tasks can continue from current page
        current_url = None
        if self.page:
            try:
                current_url = self.page.url
                if current_url == "about:blank":
                    current_url = None
            except Exception:
                pass
        
        initial_state = create_initial_state(task, session_id, current_url)
        config = {"configurable": {"thread_id": task_id}}

        # Run the graph with detailed observation
        async for event in self.graph.astream(initial_state, config, stream_mode="updates"):
            for node_name, state_update in event.items():

                # Log agent activation with full context
                self.observer.log_agent_activation(node_name, state_update)

                # Check for LLM messages and log reasoning
                if "messages" in state_update:
                    self.observer.log_llm_reasoning(state_update["messages"], node_name)

                # Check for tool calls and log executions
                task_history = state_update.get("task_history", [])
                if task_history:
                    for action in task_history:
                        if "tool_results" in action:
                            for result in action["tool_results"]:
                                tool_name = result.get("tool", "unknown")
                                tool_args = {}  # Would need to extract from action
                                self.observer.log_tool_execution(tool_name, tool_args, result)

                # SECURITY LAYER: Check if confirmation is required for destructive actions
                if state_update.get("confirmation_pending"):
                    pending_action = state_update.get("pending_action", {})
                    action_desc = pending_action.get("step", "Unknown action")
                    
                    logger.warning("")
                    logger.warning("=" * 60)
                    logger.warning("⚠️  CONFIRMATION REQUIRED - POTENTIALLY DESTRUCTIVE ACTION")
                    logger.warning("=" * 60)
                    logger.warning(f"Action: {action_desc}")
                    logger.warning("")
                    
                    # Ask user for confirmation
                    try:
                        user_input = input("Do you want to proceed? (yes/no): ").strip().lower()
                        if user_input in ["yes", "y", "да", "д"]:
                            logger.info("✅ User confirmed action - continuing...")
                            # Clear confirmation flags to continue
                            # The graph will continue on next iteration
                        else:
                            logger.info("❌ User rejected action - stopping task")
                            return "cancelled"
                    except (EOFError, KeyboardInterrupt):
                        logger.info("❌ Confirmation interrupted - stopping task")
                        return "cancelled"

                # Check for completion
                completion_status = state_update.get("completion_status")
                if completion_status in ["success", "failed", "interrupted"]:
                    duration = time.time() - task_start_time
                    result = state_update.get("final_result", "")

                    self.observer.log_task_completion(completion_status, result, duration)

                    # Save session
                    if self.context:
                        storage_state = await self.context.storage_state()
                        self.session_manager.save_session(session_id, storage_state)
                        logger.info(f"💾 Session saved: {session_id}")

                    return completion_status

                # Log state changes
                # Note: This would require comparing with previous state

        logger.warning("⚠️ Task completed without explicit status")

    async def interactive_mode(self):
        """Run interactive mode with detailed logging."""
        logger.info("[INTERACTIVE] ENTERING INTERACTIVE MODE")
        
        # Show current browser state if already open
        if self.page:
            try:
                current_url = self.page.url
                logger.info(f"[BROWSER] Current page: {current_url}")
            except:
                pass
        
        logger.info("[HELP] Enter your web automation tasks below")
        logger.info("[EXAMPLES]:")
        logger.info("   • Find the cheapest iPhone on Amazon")
        logger.info("   • Book a flight from Moscow to Paris")
        logger.info("   • Check weather in London")
        logger.info("   • exit - to quit (closes browser)")
        self.observer.log_separator("READY FOR COMMANDS", "-")

        # Use the session_id from browser initialization
        session_id = getattr(self, 'session_id', str(uuid.uuid4())[:8])

        while True:
            try:
                user_input = input("\n[TASK] Your task: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'q']:
                    logger.info("[EXIT] Goodbye!")
                    break

                await self.run_task(user_input, session_id)
                
                # Show ready message after task completion
                logger.info("")
                logger.info("=" * 60)
                logger.info("[READY] Task finished. Enter a new task or 'exit' to quit.")
                logger.info("=" * 60)

            except KeyboardInterrupt:
                logger.info("\n[INTERRUPT] Interrupted by user")
                break
            except Exception as e:
                logger.error(f"[ERROR] Error in interactive mode: {e}")
                continue
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("[CLEANUP] Cleaning up resources...")

        try:
            if self.page:
                await self.page.close()
        except Exception:
            pass  # Page may already be closed
            
        try:
            if self.context:
                await self.context.close()
        except Exception:
            pass  # Context may already be closed
            
        try:
            if self.browser:
                await self.browser.close()
        except Exception:
            pass  # Browser may already be closed
        
        # Close Chrome process if we launched it
        if self._chrome_process:
            try:
                self._chrome_process.terminate()
                self._chrome_process.wait(timeout=5)
            except Exception:
                pass

        logger.info("[SUCCESS] Cleanup completed")

    async def run(self, task: str = None, session_id: str = None):
        """Main run method."""
        try:
            self.initialize_llm()
            self.initialize_graph()

            # Initialize browser before running any tasks
            if not session_id:
                session_id = str(uuid.uuid4())[:8]
            self.session_id = session_id  # Store for use in interactive mode
            await self.initialize_browser(session_id)
            self.browser_initialized = True

            if task:
                # Execute the provided task first
                await self.run_task(task, session_id)
            
            # Always enter interactive mode (after initial task if provided)
            # This keeps the browser open and allows for additional tasks
            await self.interactive_mode()

        except KeyboardInterrupt:
            logger.info("\n[INTERRUPT] Interrupted by user")
        except Exception as e:
            logger.error(f"[ERROR] Fatal error: {e}", exc_info=True)
        finally:
            await self.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Browser Agent - Console Edition with Full LLM Visibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                              # Interactive mode
  python src/main.py "Find cheapest iPhone"       # Single task
  python src/main.py "Book flight Moscow-Paris" --session-id abc123
        """
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task to execute (if not provided, enters interactive mode)"
    )

    parser.add_argument(
        "--session-id",
        help="Session ID (auto-generated if not provided)"
    )

    args = parser.parse_args()
    
    # Run the agent
    agent = BrowserAgentConsole()
    asyncio.run(agent.run(args.task, args.session_id))


if __name__ == "__main__":
    main()