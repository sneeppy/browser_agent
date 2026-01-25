"""
LangGraph Node Functions

Each node represents a step in the agent workflow:
- Coordinator: Routes tasks between sub-agents
- Planner: Generates high-level plan
- Executor: Performs browser actions
- Extractor: Analyzes page content
- ErrorHandler: Handles errors and retries
"""

import time
import asyncio
import concurrent.futures
import os
import re
import hashlib
from typing import Dict, Any
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import AgentState
from src.tools.browser_tools import (
    BROWSER_TOOLS, get_page, take_screenshot_tool, extract_content_tool, get_current_url,
    navigate_tool, click_tool, type_into_tool, scroll_tool, handle_popup_tool, get_current_url_tool
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# LLM instances and observer (initialized in graph.py)
_primary_llm: ChatOpenAI = None
_fallback_llm: ChatOpenAI = None
_observer = None


def set_llms(primary: ChatOpenAI, fallback: ChatOpenAI):
    """Set LLM instances for nodes to use."""
    global _primary_llm, _fallback_llm
    _primary_llm = primary
    _fallback_llm = fallback


def set_observer(observer):
    """Set observer for detailed logging."""
    global _observer
    _observer = observer


def get_observer():
    """Get the current observer."""
    return _observer


def get_llm():
    """Get LLM instance, with fallback if primary fails."""
    if _primary_llm:
        return _primary_llm
    if _fallback_llm:
        logger.warning("Using fallback LLM")
        return _fallback_llm
    raise RuntimeError("No LLM configured")


async def coordinator_node(state: AgentState) -> Dict[str, Any]:
    """
    Coordinator node: Routes tasks between sub-agents based on current state.
    
    This coordinator implements STEP-BY-STEP planning:
    - After each executor action, goes back to planner
    - Planner sees actual page state before deciding next action
    - No blind multi-step plans
    """
    observer = get_observer()
    if observer:
        observer.log_agent_activation("COORDINATOR", {
            "user_task": state.get("user_task", ""),
            "current_step": "Analyzing task and routing to appropriate agent"
        })

    logger.info("[COORDINATOR] Analyzing current state and routing...")
    
    # Check maximum step count to prevent infinite loops
    step_count = state.get("step_count", 0) + 1
    max_steps = 50  # Reduced since we're doing step-by-step planning
    
    if step_count > max_steps:
        logger.error(f"[COORDINATOR] Maximum step count ({max_steps}) reached, forcing completion")
        return {
            "completion_status": "failed",
            "last_error": f"Maximum step count ({max_steps}) reached. Task may be stuck in a loop.",
            "should_continue": False,
            "last_update_time": time.time(),
            "next_node": None
        }
    
    updates = {
        "last_update_time": time.time(),
        "next_node": None,
        "step_count": step_count
    }
    
    # If task just started (no plan yet), go to planner
    if not state.get("current_plan"):
        updates["next_node"] = "planner"
        logger.info("Coordinator: Routing to Planner (initial)")
        return updates
    
    # If error occurred, route to error handler (unless already retrying)
    if state.get("last_error") and not state.get("should_retry"):
        updates["next_node"] = "error_handler"
        logger.info("Coordinator: Routing to ErrorHandler")
        return updates
    
    # If confirmation pending, wait
    if state.get("confirmation_pending"):
        updates["next_node"] = "coordinator"  # Stay here
        logger.info("Coordinator: Waiting for confirmation")
        return updates
    
    # Check for loops - same step failing repeatedly
    task_history = state.get("task_history", [])
    if task_history and len(task_history) >= 3:
        last_steps = [action.get("step") for action in task_history[-3:]]
        if len(set(last_steps)) == 1 and len(last_steps) == 3:
            all_failed = all(not action.get("success", False) for action in task_history[-3:])
            if all_failed:
                logger.warning(f"[COORDINATOR] Loop detected: '{last_steps[0]}' failed 3 times")
                updates["next_steps"] = []  # Clear the stuck step
                updates["next_node"] = "planner"
                updates["last_error"] = f"Action failed repeatedly: {last_steps[0]}"
                return updates

    # Check for stagnation loops - repeating successful action without page progress
    # This catches cases like "refresh/retry" loops where clicks succeed but nothing changes.
    if task_history and len(task_history) >= 4:
        recent = task_history[-4:]

        def _extract_action_key(action: Dict[str, Any]) -> str:
            tool_results = action.get("tool_results") or []
            # Prefer click label inferred from tool result message (generic)
            for tr in tool_results:
                if tr.get("tool") != "click":
                    continue
                res = tr.get("result") or {}
                msg = (res.get("message") or "").strip()
                method = (res.get("method") or "").strip()
                m = re.search(r"containing:\s*(.+)$", msg, flags=re.IGNORECASE)
                if m:
                    label = m.group(1).strip()
                    if label:
                        return f"click:{label.lower()}"
                m = re.search(r"text:\s*(.+)$", msg, flags=re.IGNORECASE)
                if m:
                    label = m.group(1).strip()
                    if label:
                        return f"click_text:{label.lower()}"
                if method:
                    return f"click:{method.lower()}"
            # Navigate key
            for tr in tool_results:
                if tr.get("tool") != "navigate":
                    continue
                res = tr.get("result") or {}
                url = (res.get("current_url") or "").strip()
                if url:
                    return f"nav:{url}"
            # Scroll key
            for tr in tool_results:
                if tr.get("tool") != "scroll":
                    continue
                res = tr.get("result") or {}
                args = tr.get("args") or {}
                direction = (res.get("direction") or args.get("direction") or "").strip().lower()
                amount = res.get("amount") or args.get("amount") or ""
                if direction:
                    return f"scroll:{direction}:{amount}"
            # Fallback to step text (normalized)
            step = (action.get("step") or "").strip().lower()
            step = re.sub(r"\s+", " ", step)
            return f"step:{step[:80]}" if step else "step:unknown"

        keys = [_extract_action_key(a) for a in recent]
        urls = [(a.get("url") or "").strip() for a in recent]
        all_success = all(a.get("success", False) for a in recent)
        same_key = len(set(keys)) == 1
        same_url = all(urls) and len(set(urls)) == 1

        if all_success and same_key and same_url:
            logger.warning(f"[COORDINATOR] Stagnation loop detected: repeated {keys[0]} on {urls[0]}")
            updates["last_error"] = f"Stuck loop detected: repeated action without progress ({keys[0]})"
            # If we're just scrolling without new signals, bounce back to planner to pick a different action
            # (e.g., Extract content) instead of hard-failing.
            if keys[0].startswith("scroll:"):
                updates["next_steps"] = []
                updates["next_node"] = "planner"
                return updates
            updates["should_retry"] = False
            updates["next_steps"] = []
            updates["next_node"] = "error_handler"
            return updates
    
    # STEP-BY-STEP LOGIC:
    # If we have a pending step to execute, go to executor
    if state.get("next_steps") and len(state.get("next_steps", [])) > 0:
        updates["next_node"] = "executor"
        logger.info("Coordinator: Routing to Executor")
        return updates
    
    # If no pending steps, check if task might be complete
    # Look at recent history for meaningful progress
    if task_history:
        last_action = task_history[-1] if task_history else None
        
        # If last action was successful and there are no more steps,
        # go back to planner to decide next action (or complete)
        if last_action:
            logger.info("Coordinator: Step completed, returning to Planner for next decision")
            updates["next_node"] = "planner"
            return updates
    
    # Default: go to planner
    updates["next_node"] = "planner"
    logger.info("Coordinator: Routing to Planner")
    return updates


async def get_page_context(max_length: int = 2000) -> str:
    """
    Get current page context for the planner to make informed decisions.
    Returns a summary of visible elements and page content.
    """
    try:
        page = get_page()
        # Wait briefly for DOM to be ready (avoids "empty snapshots" right after clicks/navigations).
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=2000)
        except Exception:
            pass
        try:
            await page.wait_for_timeout(150)
        except Exception:
            pass

        current_url = page.url

        # Get page title
        title = await page.title()
        
        # Get visible text content (prioritize what's actually on screen)
        visible_text = await page.evaluate("""
            () => {
                const texts = [];
                const seen = new Set();

                const isVisibleElement = (el) => {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') return false;
                    if (style.opacity === '0') return false;
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return false;
                    // Must be near viewport (avoid offscreen/screen-reader-only boilerplate)
                    if (rect.bottom < 0) return false;
                    if (rect.top > window.innerHeight * 5) return false;
                    return true;
                };

                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode: (node) => {
                            const parent = node.parentElement;
                            if (!parent) return NodeFilter.FILTER_REJECT;
                            if (!isVisibleElement(parent)) return NodeFilter.FILTER_REJECT;
                            const text = (node.textContent || '').trim();
                            if (text.length < 2) return NodeFilter.FILTER_REJECT;
                            // Skip JSON-like content
                            if (text.includes('{"') || text.includes('widgets')) return NodeFilter.FILTER_REJECT;
                            return NodeFilter.FILTER_ACCEPT;
                        }
                    }
                );

                let node;
                while ((node = walker.nextNode()) && texts.length < 60) {
                    let text = (node.textContent || '').trim();
                    if (text.length < 3) continue;
                    if (text.length > 250) text = text.slice(0, 250);
                    text = text.replace(/\\s+/g, ' ');
                    const key = text.toLowerCase();
                    if (seen.has(key)) continue;
                    seen.add(key);
                    texts.push(text);
                }

                return texts.slice(0, 35).join('\\n');
            }
        """)
        
                # Get interactive elements summary (include list rows/items, dedupe repeated labels)
        elements = await page.evaluate("""
            () => {
                const candidates = [];
                const seenCount = new Map();
                const allowedRoles = new Set([
                    'button', 'link', 'tab', 'menuitem', 'option', 'checkbox', 'radio',
                    'listitem', 'row'
                ]);

                const selectors = [
                    'a[href]',
                    'button',
                    'input',
                    '[role]',
                    '[onclick]',
                    '[tabindex]'
                ].join(', ');

                const els = Array.from(document.querySelectorAll(selectors)).slice(0, 900);

                const getLabel = (el) => {
                    let text = (el.innerText || el.textContent || '').trim();
                    text = text.replace(/\\s+/g, ' ').substring(0, 120);
                    if (!text) text = el.getAttribute('placeholder') || el.getAttribute('aria-label') || el.getAttribute('title') || '';
                    if (!text) {
                        const img = el.querySelector && el.querySelector('img');
                        if (img) text = img.getAttribute('alt') || '';
                    }
                    text = (text || '').trim();
                    return text;
                };

                for (const el of els) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
                    if (rect.bottom < 0) continue;
                    // capture more than just the very top to include listings below the fold
                    if (rect.top > window.innerHeight * 8) continue;

                    const tag = el.tagName.toLowerCase();
                    const role = (el.getAttribute('role') || '').toLowerCase();
                    if (role && !allowedRoles.has(role) && tag !== 'a' && tag !== 'button' && tag !== 'input') {
                        continue;
                    }

                    const typeAttr = (el.getAttribute('type') || '').toLowerCase();
                    const isCheckbox = role === 'checkbox' || (tag === 'input' && (typeAttr === 'checkbox' || typeAttr === 'radio'));

                    let kind = 'element';
                    if (tag === 'input') kind = 'input';
                    else if (tag === 'button' || role === 'button') kind = 'button';
                    else if (tag === 'a' || role === 'link') kind = 'link';
                    else if (role) kind = role;
                    if (isCheckbox) kind = 'checkbox';

                    const text = getLabel(el);
                    if (!text || text.includes('{')) continue;

                    // De-dupe by kind+label, but keep a few duplicates for repeated controls
                    // so the planner can refer to "second/third" where needed.
                    const key = `${kind}:${text.toLowerCase()}`;
                    const prev = seenCount.get(key) || 0;
                    const next = prev + 1;
                    seenCount.set(key, next);
                    if (kind === 'checkbox') {
                        if (next > 1) continue; // checkboxes are extremely spammy
                    } else {
                        if (next > 3) continue; // allow up to 3 duplicates for buttons/links/etc.
                    }
                    const disambiguatedText = (next > 1 && (kind === 'button' || kind === 'link')) ? `${text} (#${next})` : text;

                    // Score candidates to surface main content actions ahead of repetitive controls.
                    let score = 0;
                    if (kind === 'link') score += 2;
                    if (kind === 'button') score += 1;
                    if (text.length >= 18) score += 2;
                    if (text.includes(' ')) score += 1;
                    if (kind === 'checkbox') score -= 1;
                    if (/^\\d+$/.test(text)) score -= 2;

                    candidates.push({ kind, tag, role, typeAttr, text: disambiguatedText.substring(0, 120), rectTop: rect.top, score });
                }

                candidates.sort((a, b) => (b.score - a.score) || (a.rectTop - b.rectTop));

                const results = [];
                const counts = { link: 0, button: 0, input: 0, checkbox: 0, other: 0 };
                const limits = { link: 18, button: 12, input: 6, checkbox: 12, other: 12 };
                const maxTotal = 55;

                const push = (label) => {
                    if (results.length < maxTotal) results.push(label);
                };

                for (const c of candidates) {
                    if (results.length >= maxTotal) break;
                    const kind = c.kind;
                    const text = c.text;
                    if (!text) continue;

                    if (kind === 'checkbox') {
                        if (counts.checkbox >= limits.checkbox) continue;
                        counts.checkbox += 1;
                        push(`[checkbox] ${text}`);
                        continue;
                    }
                    if (kind === 'input') {
                        if (counts.input >= limits.input) continue;
                        counts.input += 1;
                        push(`[input${c.typeAttr ? ':' + c.typeAttr : ''}] ${text || 'text field'}`);
                        continue;
                    }
                    if (kind === 'button') {
                        if (counts.button >= limits.button) continue;
                        counts.button += 1;
                        push(`[button] ${text}`);
                        continue;
                    }
                    if (kind === 'link') {
                        if (counts.link >= limits.link) continue;
                        counts.link += 1;
                        push(`[link] ${text}`);
                        continue;
                    }

                    if (counts.other >= limits.other) continue;
                    counts.other += 1;
                    push(`[${kind}] ${text}`);
                }

                return results;
            }
        """)
        
        context = f"""Page: {title}
URL: {current_url}

Visible content:
{visible_text[:800] if visible_text else 'No visible text'}

Interactive elements:
{chr(10).join(elements[:55]) if elements else 'No elements found'}"""
        
        return context[:max_length + 1000]  # Allow more context for planner
        
    except Exception as e:
        return f"Could not analyze page: {str(e)}"


async def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner node: Generates ONE next step based on current page state.
    
    This is a step-by-step planner that sees the actual page before deciding
    what to do next, rather than planning many steps blindly.
    """
    observer = get_observer()
    if observer:
        observer.log_agent_activation("PLANNER", {
            "user_task": state.get("user_task", ""),
            "current_step": "Analyzing page and deciding next action"
        })

    logger.info("[PLANNER] Analyzing current page state and deciding next step...")
    
    try:
        llm = get_llm().bind_tools(BROWSER_TOOLS)
        
        user_task = state.get("user_task", "")
        current_url = state.get("current_url", "Unknown")
        page_summary = state.get("page_summary", "")
        task_history = state.get("task_history", [])
        
        # Get real-time page context for informed decision making
        page_context = await get_page_context()
        
        # Build history of what we've done, emphasizing failures
        history_text = ""
        failed_actions = []
        if task_history:
            recent_actions = task_history[-10:]  # Last 10 actions
            history_items = []
            for action in recent_actions:
                step = action.get("step", "unknown")
                success = action.get("success", False)
                status = "✓" if success else "✗ FAILED"
                history_items.append(f"  {status} {step}")
                if not success:
                    failed_actions.append(step)
            history_text = "Actions taken:\n" + "\n".join(history_items)
            
            # Add explicit warning about failed actions
            if failed_actions:
                history_text += "\n\n⚠️ WARNING: The following actions FAILED - do NOT repeat them:\n"
                for fa in failed_actions[-3:]:  # Last 3 failures
                    history_text += f"  - {fa}\n"
                history_text += "Try a DIFFERENT approach!"
        
        # Build context message
        context = f"""USER TASK: {user_task}

CURRENT PAGE STATE:
{page_context}

{history_text if history_text else "No actions taken yet."}

Based on the current page state, what is the SINGLE NEXT ACTION to take?
"""
        
        messages = [
            SystemMessage(content="""You are a step-by-step browser automation planner.
Your job is to decide the ONE NEXT ACTION based on the CURRENT page state.

IMPORTANT: You can SEE the actual page content and elements. Use this information!

RESPONSE FORMAT:
1. Brief analysis (1-2 sentences): What do you see? What's the current state?
2. Next step: ONE specific browser action

RULES FOR THE NEXT STEP:
- Must be a SINGLE browser action (one click OR one type OR one navigation OR one extract)
- Must reference ACTUAL elements you can see on the page
- Be SPECIFIC: use exact button text, link text, or field names from the page

AVAILABLE ACTIONS:
- Navigate to [URL] - go to a specific URL
- Type '[text]' into [field description] - type text into a visible input field
- Click [element description] - click a visible button, link, or element
- Extract [what to extract] - read content from the page
- Scroll down/up - scroll to see more content

WHEN TASK IS COMPLETE:
If the user's task has been accomplished, respond with:
"TASK COMPLETE: [brief summary of what was done/found]"

WHEN STUCK:
If you can't proceed (page not loading, element not found), try:
- Scroll down to see more content on the page
- Navigate to a different page
- Extract content to understand current state

CRITICAL - AVOID REPEATING THE SAME ACTION:
If you already tried the same action (e.g., clicking the same button/link) 2+ times recently
and the page state is not improving, do NOT keep repeating it.
Pick a DIFFERENT action type (scroll, extract, navigate) or a different element.

IMPORTANT - SCROLL TO FIND ELEMENTS:
If you're on a page with search results or listings but don't see the elements you need to interact with,
scroll down to load and reveal more content before giving up.

CRITICAL - FAILED ACTIONS (marked with ✗):
If a previous action FAILED (shown with ✗ in history), DO NOT repeat it!
Try a DIFFERENT approach:
- If clicking a link failed, try using the search input directly
- If a button wasn't found, look for alternative elements
- If one method doesn't work, use another way to achieve the same goal
"""), 
            HumanMessage(content=context)
        ]

        # Log LLM reasoning process
        if observer:
            observer.log_llm_reasoning(messages, "PLANNER")
        
        response = await llm.ainvoke(messages)

        # Log LLM response
        if observer and hasattr(response, 'content'):
            response_messages = messages + [AIMessage(content=response.content)]
            observer.log_llm_reasoning(response_messages, "PLANNER")
        
        # Extract plan from response
        plan_text = response.content if hasattr(response, 'content') else str(response)
        
        # Check if task is complete
        plan_lower = plan_text.lower()
        if "task complete" in plan_lower or "задача выполнена" in plan_lower:
            logger.info("[PLANNER] Task marked as complete")
            return {
                "completion_status": "success",
                "final_result": plan_text[:500],
                "should_continue": False,
                "current_plan": plan_text[:1000],
                "next_steps": [],
                "messages": [AIMessage(content=plan_text)] if plan_text else [],
                "last_update_time": time.time()
            }
        
        # Parse the ONE next step
        next_step = None
        lines = plan_text.split("\n")
        
        # First pass: look specifically for "Next step:" pattern (may have number prefix like "2. Next step:")
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Skip analysis/brief lines - these are NOT actions
            if "brief analysis" in line_lower or "analysis:" in line_lower:
                continue
            
            # Look for "Next step:" anywhere in line (handles "2. Next step: ..." format)
            if "next step" in line_lower:
                # Extract text after "Next step:"
                import re
                match = re.search(r'next step[:\s]+(.+)$', line, re.IGNORECASE)
                if match:
                    next_step = match.group(1).strip()
                    break
        
        # Second pass: look for numbered action items (but skip analysis)
        if not next_step:
            for line in lines:
                line = line.strip()
                line_lower = line.lower()
                
                # Skip analysis lines
                if "brief analysis" in line_lower or "analysis:" in line_lower:
                    continue
                
                # Look for numbered or bulleted action
                if line and (line[0].isdigit() or line.startswith(("-", "*", "•"))):
                    step = line.lstrip("0123456789.-*•) ").strip()
                    step = step.replace("**", "").replace("__", "")
                    step_lower = step.lower()
                    
                    # Must contain action keywords AND not be analysis
                    if any(kw in step_lower for kw in ["navigate", "click", "type", "extract", "scroll", "go to", "open"]):
                        # Double check it's not analysis text containing these words
                        if not any(skip in step_lower for skip in ["analysis", "the page is", "previous attempts", "error occurred"]):
                            next_step = step
                            break
        
        # Third pass: try to extract any action-like sentence
        if not next_step:
            for line in lines:
                line = line.strip()
                line_lower = line.lower()
                
                # Skip analysis lines
                if "brief analysis" in line_lower or "analysis:" in line_lower or "the page is" in line_lower:
                    continue
                
                if any(kw in line_lower for kw in ["navigate to", "click on", "click the", "type '", "extract", "scroll"]):
                    next_step = line.lstrip("0123456789.-*•) ").strip()
                    next_step = next_step.replace("**", "").replace("__", "")
                    break
        
        # Fallback if still no step - let LLM decide based on task
        if not next_step:
            if not current_url or current_url == "Unknown" or current_url == "about:blank":
                # No specific URL - LLM should determine from task context
                next_step = f"Navigate to the website mentioned in the task: {user_task}"
            else:
                next_step = f"Extract content related to: {user_task}"
        
        updates = {
            "current_plan": plan_text[:2000],
            "next_steps": [next_step] if next_step else [],  # Only ONE step!
            "messages": [AIMessage(content=plan_text)] if plan_text and plan_text.strip() else [],
            "last_update_time": time.time(),
        }
        
        logger.info(f"[PLANNER] Next step: {next_step}")
        return updates
        
    except Exception as e:
        logger.error(f"Planner error: {e}")
        return {
            "last_error": f"Planner failed: {str(e)}",
            "should_continue": False,
            "last_update_time": time.time()
        }


async def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Executor node: Performs browser actions using tools.
    """
    observer = get_observer()
    if observer:
        observer.log_agent_activation("EXECUTOR", {
            "user_task": state.get("user_task", ""),
            "current_step": f"Executing: {state.get('current_step', 'Unknown step')}",
            "next_steps": state.get("next_steps", [])
        })

    logger.info("[EXECUTOR] Performing browser actions...")
    
    try:
        next_steps = state.get("next_steps", [])
        if not next_steps:
            return {
                "should_continue": True,
                "next_steps": [],
                "last_update_time": time.time()
            }
        
        current_step = next_steps[0]
        remaining_steps = next_steps[1:]
        
        logger.info(f"Executor: Executing step: {current_step}")

        # Fast-path execution for obvious actions to avoid LLM round-trips
        step_text = current_step or ""
        step_lower = step_text.lower()
        tool_calls = []

        url_match = re.search(r"(https?://[^\s`\"'\]\[()]+)", step_text)
        if url_match:
            url = url_match.group(1).strip("`.,)\"'][")
            tool_calls.append({"name": "navigate", "args": {"url": url}})

        if not tool_calls and ("type" in step_lower or "введ" in step_lower or "напечат" in step_lower):
            quote_match = re.search(r"[\"'“”«»]([^\"'“”«»]+)[\"'“”«»]", step_text)
            if quote_match:
                text_value = quote_match.group(1).strip()
                field_desc = "search field"
                into_match = re.search(r"(?:into|in|в)\s+(.+)$", step_text, re.IGNORECASE)
                if into_match:
                    field_desc = into_match.group(1).strip().strip(".")
                tool_calls.append({
                    "name": "type_into",
                    "args": {"field_description": field_desc, "text": text_value}
                })

        if not tool_calls and ("click" in step_lower or "нажм" in step_lower or "клик" in step_lower):
            element_desc = "button"  # Generic description, let tool figure it out
            click_match = re.search(r"(?:click|нажм(?:и|ите)|клик(?:ни|ните))\s+(.+)$", step_text, re.IGNORECASE)
            if click_match:
                element_desc = click_match.group(1).strip().strip(".")
            tool_calls.append({"name": "click", "args": {"element_description": element_desc}})
        
        # Fast-path for extract content
        if not tool_calls and ("extract" in step_lower or "извлеч" in step_lower or "прочитай" in step_lower or "read" in step_lower):
            query = "main content"
            extract_match = re.search(r"(?:extract|извлеч|прочитай|read)\s+(?:content\s+)?(?:related to:|about|о|об)?\s*(.+)$", step_text, re.IGNORECASE)
            if extract_match:
                query = extract_match.group(1).strip().strip(".")
            tool_calls.append({"name": "extract_content", "args": {"query": query}})

        response = None
        if tool_calls:
            logger.info("[EXECUTOR] Using fast-path tool execution")
        else:
            # Use LLM to decide which tool to use and how
            llm = get_llm().bind_tools(BROWSER_TOOLS)

            messages = [
                SystemMessage(content="""You are an execution agent that controls a web browser.
Your job is to EXECUTE browser actions by calling the appropriate tools.

CRITICAL: You MUST call at least one tool for every step. Do not just describe what to do - actually call the tool!

Available tools:
- navigate(url): Go to a URL - use this to navigate to a website
- click(element_description): Click an element - describe what to click (e.g., "Search button", "Login link", "first item in the list")
- type_into(field_description, text): Type text into a field - describe the field and what text to type
- extract_content(query): Get page content - use to read and analyze page content
- take_screenshot(): Capture page image - use to see the current page
- scroll(direction, amount): Scroll the page - direction: 'up', 'down', 'left', 'right'
- get_current_url(): Get current URL - use to check where you are

IMPORTANT GUIDELINES:
1. For steps that say "click on first/second/third item" - use click() with description like "first item in the list"
2. For steps that require reading content - use extract_content() first to understand what's on the page
3. For steps that require finding something specific - use extract_content() with a specific query
4. If the step seems vague or complex, break it down: first extract_content() to see what's available, then click() on the specific element

Remember: ALWAYS call a tool. Never just describe what should be done."""),
                HumanMessage(content=f"Execute this step: {current_step}\n\nCurrent URL: {state.get('current_url', 'Unknown')}\n\nIMPORTANT: You must call at least one tool to execute this step. Do not just describe what to do - actually call the tool!")
            ]

            llm_timeout = float(os.getenv("EXECUTOR_LLM_TIMEOUT", "30"))
            try:
                response = await asyncio.wait_for(llm.ainvoke(messages), timeout=llm_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[EXECUTOR] LLM call timed out after {llm_timeout}s")
                return {
                    "last_error": f"Executor LLM timeout after {llm_timeout}s",
                    "should_retry": True,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "should_continue": True,
                    "last_update_time": time.time(),
                    "messages": [AIMessage(content="Executor LLM timeout, retrying")]
                }
            except asyncio.CancelledError:
                logger.warning("[EXECUTOR] LLM call was cancelled")
                return {
                    "last_error": "Executor LLM call was cancelled",
                    "should_retry": True,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "should_continue": True,
                    "last_update_time": time.time(),
                    "messages": [AIMessage(content="Executor LLM call cancelled, retrying")]
                }

            # Log LLM response
            if observer and hasattr(response, 'content'):
                response_messages = messages + [AIMessage(content=response.content)]
                observer.log_llm_reasoning(response_messages, "EXECUTOR")

            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = response.tool_calls

        # Execute tool calls if any
        tool_results = []
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                logger.info(f"🔧 EXECUTOR: Calling tool {tool_name} with args {tool_args}")

                # Map tool names to async functions
                tool_function_map = {
                    "navigate": navigate_tool,
                    "click": click_tool,
                    "type_into": type_into_tool,
                    "extract_content": extract_content_tool,
                    "take_screenshot": take_screenshot_tool,
                    "scroll": scroll_tool,
                    "handle_popup": handle_popup_tool,
                    "get_current_url": get_current_url_tool
                }
                
                # Find and execute tool
                if tool_name in tool_function_map:
                    try:
                        # Filter out None values from tool_args
                        clean_args = {k: v for k, v in tool_args.items() if v is not None}
                        
                        # Call async function directly
                        tool_func = tool_function_map[tool_name]
                        result = await tool_func(**clean_args)
                        
                        # Ensure result is a dict
                        if not isinstance(result, dict):
                            result = {"success": True, "result": result}
                        
                        tool_results.append({
                            "tool": tool_name,
                            "args": clean_args,
                            "result": result,
                            "success": result.get("success", False) if isinstance(result, dict) else True
                        })

                        # Log tool execution
                        if observer:
                            observer.log_tool_execution(tool_name, clean_args, result)

                        # Update state based on tool result
                        if tool_name == "navigate" and isinstance(result, dict):
                            if result.get("success"):
                                old_url = state.get("current_url")
                                new_url = result.get("current_url")
                                state["current_url"] = new_url
                                logger.info(f"[STATE] URL updated: {old_url} -> {new_url}")
                            else:
                                logger.warning(f"[STATE] Navigate failed: {result}")
                        elif tool_name == "take_screenshot" and isinstance(result, dict):
                            if result.get("success"):
                                state["last_screenshot_path"] = result.get("path")
                            
                        logger.info(f"[SUCCESS] Tool {tool_name} executed successfully")
                        
                    except Exception as e:
                        logger.error(f"Tool execution error for {tool_name}: {e}", exc_info=True)
                        tool_results.append({
                            "tool": tool_name,
                            "args": clean_args,
                            "result": {"success": False, "error": str(e)},
                            "success": False
                        })
                else:
                    logger.error(f"Tool {tool_name} not found in tool_function_map")
                    tool_results.append({
                        "tool": tool_name,
                        "args": clean_args if "clean_args" in locals() else tool_args,
                        "result": {"success": False, "error": f"Tool {tool_name} not found"},
                        "success": False
                    })
        
        # If no tool calls were made, this is a problem - we need to execute something
        if not tool_results:
            logger.warning(f"[EXECUTOR] No tools were called for step: {current_step}")
            logger.warning(f"[EXECUTOR] LLM response: {response.content if response and hasattr(response, 'content') else 'No content'}")
            
            # Track failed attempts for this step to break infinite loops
            existing_task_history = state.get("task_history", [])
            failed_attempts = sum(1 for action in existing_task_history[-5:] if action.get("step") == current_step and not action.get("success"))
            
            # Create new history entry (operator.add will append this to existing task_history)
            new_history_entry = {
                "step": current_step,
                "tool_results": [],
                "timestamp": time.time(),
                "success": False,
                "llm_response": response.content if response and hasattr(response, 'content') else None
            }
            
            if failed_attempts >= 2:
                # Too many failed attempts for this step - skip it and move on
                logger.error(f"[EXECUTOR] Step '{current_step}' failed {failed_attempts + 1} times, skipping")
                return {
                    "last_error": f"Step '{current_step}' failed too many times, skipping",
                    "should_continue": True,
                    "next_steps": remaining_steps,  # Don't re-add the failed step
                    "task_history": [new_history_entry],  # Return as list for operator.add
                    "last_update_time": time.time(),
                    "messages": [AIMessage(content=f"Skipped step after {failed_attempts + 1} failures: {current_step}")]
                }
            
            # Mark this as needing replanning - the step might be too vague
            # Put the step back in next_steps
            remaining_steps = [current_step] + remaining_steps
            return {
                "last_error": f"Could not execute step '{current_step}': LLM did not call any tools. Step may be too vague.",
                "should_continue": True,
                "next_steps": remaining_steps,
                "task_history": [new_history_entry],  # Return as list for operator.add
                "last_update_time": time.time(),
                "messages": [AIMessage(content=f"Executor could not run step: {current_step}")]
            }
        
        url_before = state.get("current_url")

        # Check for security-sensitive actions (optional)
        require_confirmation = os.getenv("REQUIRE_CONFIRMATION", "false").lower() == "true"
        current_step_lower = current_step.lower()
        user_task_lower = state.get("user_task", "").lower()
        payment_keywords = ["pay", "payment", "purchase", "buy", "card"]
        destructive_keywords = ["delete", "remove", "trash", "erase", "cancel account", "удал", "очист"]
        user_requested_deletion = any(keyword in user_task_lower for keyword in destructive_keywords)

        requires_confirmation = False
        if require_confirmation:
            requires_confirmation = any(
                keyword in current_step_lower for keyword in payment_keywords + destructive_keywords
            )
            if requires_confirmation and user_requested_deletion:
                # Let explicitly requested deletions proceed without blocking
                requires_confirmation = any(keyword in current_step_lower for keyword in payment_keywords)
        
        # Create new history entry (operator.add will append this to existing task_history)
        new_history_entry = {
            "step": current_step,
            "tool_results": tool_results,
            "timestamp": time.time(),
            "success": all(r.get("success", False) for r in tool_results) if tool_results else False,
            "llm_response": response.content if hasattr(response, 'content') else None,
            "url_before": url_before
        }
        
        current_url = state.get("current_url")
        last_error = None
        # Get current URL - use direct function call instead of tool
        # Note: get_current_url_tool and get_page are imported at the top of this file
        try:
            # Verify page is available
            page = get_page()
            url_result = await get_current_url_tool()
            if isinstance(url_result, dict) and url_result.get("success"):
                current_url = url_result.get("url")
        except RuntimeError as e:
            # Browser not initialized
            logger.error(f"[EXECUTOR] Browser not initialized: {e}")
            last_error = f"Browser not initialized: {str(e)}"
        except Exception as e:
            logger.warning(f"Failed to get current URL: {e}")
            last_error = f"Failed to get current URL: {str(e)}"
        
        updates = {
            "current_step": current_step,
            "next_steps": remaining_steps,
            "current_url": current_url,
            "requires_confirmation": requires_confirmation,
            "confirmation_pending": requires_confirmation,
            "pending_action": {
                "step": current_step,
                "tool_results": tool_results
            } if requires_confirmation else None,
            "last_error": last_error or (None if all(r.get("success", False) for r in tool_results) else "Some tools failed"),
            "should_continue": True,
            "last_update_time": time.time(),
            "messages": [AIMessage(content=f"Executed: {current_step}")] if current_step and current_step.strip() else [],
            "task_history": [new_history_entry]  # Return as list for operator.add
        }

        # Attach URL after to history (helps detect "successful but no-progress" loops)
        new_history_entry["url"] = current_url
        
        # If confirmation required, don't continue automatically
        if requires_confirmation:
            updates["should_continue"] = False
        
        logger.info(f"Executor: Completed step, {len(remaining_steps)} steps remaining")
        return updates
        
    except Exception as e:
        logger.error(f"Executor error: {e}")
        return {
            "last_error": f"Executor failed: {str(e)}",
            "should_retry": True,
            "retry_count": state.get("retry_count", 0) + 1,
            "last_update_time": time.time()
        }


async def extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Extractor node: Analyzes page content and extracts relevant information.
    """
    observer = get_observer()
    if observer:
        observer.log_agent_activation("EXTRACTOR", {
            "user_task": state.get("user_task", ""),
            "current_step": "Analyzing page content and extracting information"
        })

    logger.info("[EXTRACTOR] Analyzing page content and extracting relevant data...")
    
    try:
        page = get_page()
        
        # Take screenshot for vision analysis
        screenshot_result = await take_screenshot_tool()
        screenshot_path = None
        if isinstance(screenshot_result, dict) and screenshot_result.get("success"):
            screenshot_path = screenshot_result.get("path")
        
        # Extract content
        extract_result = await extract_content_tool(state.get("user_task", "all"))
        page_content = ""
        if isinstance(extract_result, dict) and extract_result.get("success"):
            page_content = extract_result.get("content", "")
        
        # Use LLM to summarize and extract relevant info
        llm = get_llm()
        
        task_history = state.get("task_history", [])
        actions_taken = "\n".join([
            f"- {action.get('step', 'unknown')}: {len(action.get('tool_results', []))} tool(s) executed"
            for action in task_history[-5:]
        ]) if task_history else "No actions taken yet"
        
        summary_prompt = f"""
Analyze this webpage content and extract information relevant to the user's task.

User Task: {state.get('user_task', '')}
Current URL: {state.get('current_url', 'Unknown')}
Actions Taken So Far:
{actions_taken}

Page Content (first 3000 chars): {page_content[:3000]}

IMPORTANT: Be very strict about task completion. The task is ONLY complete if:
- The user's specific request has been FULLY satisfied
- All required actions have been performed (not just navigation or reading lists)
- For tasks involving multiple items:
  * If the task requires analyzing or acting on specific items, ensure those items were actually opened/processed as needed
  * If the task requires comparing or choosing, clearly justify the result using evidence visible in the page content

Provide:
1. A brief summary of the current page
2. Key information relevant to the task (if found)
3. Whether the task is COMPLETE or what actions are still MISSING
   - If task is complete, explicitly state "TASK IS COMPLETE" and explain what was accomplished
   - If task is incomplete, explicitly state "TASK IS NOT COMPLETE" and list what specific actions need to be done

Be honest: 
- If we've only navigated to a page but haven't performed the actual task, it's NOT complete
- If we've only read a list but haven't analyzed individual items, it's NOT complete
- If the task requires deleting/modifying something and we only read data, it's NOT complete
"""
        
        messages = [HumanMessage(content=summary_prompt)]
        if screenshot_path:
            # In production, use vision API here
            messages[0].content += f"\n\nScreenshot available at: {screenshot_path}"
        
        # Log LLM analysis request
        if observer:
            observer.log_llm_reasoning(messages, "EXTRACTOR")

        # Use timeout for LLM call to prevent hanging
        import asyncio
        llm_failed = False  # Track if LLM call failed
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=30.0  # 30 second timeout
            )
            summary = response.content if hasattr(response, 'content') else str(response)
        except asyncio.TimeoutError:
            logger.warning("[EXTRACTOR] LLM call timed out, using fallback summary")
            llm_failed = True
            # Create a simple summary based on page content - but don't claim task is complete
            if page_content:
                summary = f"Page content extracted ({len(page_content)} chars). LLM analysis timed out - cannot determine if task is complete."
            else:
                summary = "Unable to analyze page content (timeout). Task status unknown."
            response = None  # Mark that we don't have a real response
        except asyncio.CancelledError:
            logger.warning("[EXTRACTOR] LLM call was cancelled, using fallback summary")
            llm_failed = True
            summary = "Analysis was interrupted. Task status unknown - need to continue."
            response = None

        # Log LLM analysis response (only if we got a response)
        if response and observer and hasattr(response, 'content'):
            response_messages = messages + [AIMessage(content=response.content)]
            observer.log_llm_reasoning(response_messages, "EXTRACTOR")
        elif observer:
            logger.info("[EXTRACTOR] Using fallback summary (LLM call failed or timed out)")
        
        # Check if task is complete - be strict and avoid domain-specific heuristics.
        task_history = state.get("task_history", [])
        
        # Check LLM response for completion indicators
        summary_lower = summary.lower()
        completion_keywords = ["task is complete", "task completed", "successfully completed", 
                              "all done", "finished successfully", "task finished"]
        incomplete_keywords = ["not complete", "still need", "missing", "need to", 
                               "should", "must", "requires", "incomplete"]
        
        llm_says_complete = any(keyword in summary_lower for keyword in completion_keywords)
        llm_says_incomplete = any(keyword in summary_lower for keyword in incomplete_keywords)
        
        # Count "meaningful" successful actions (exclude pure navigation/meta).
        meaningful_actions = len(
            [
                a
                for a in task_history
                if a.get("tool_results")
                and any(
                    r.get("tool") not in ["navigate", "get_current_url", "take_screenshot"]
                    and r.get("success", False)
                    for r in a.get("tool_results", [])
                )
            ]
        )
        
        # Count how many times we've extracted
        extraction_count = state.get("extraction_count", 0) + 1
        
        # CRITICAL: If LLM call failed, we cannot reliably determine completion
        # Default to NOT complete so the agent continues working
        if llm_failed:
            logger.warning("[EXTRACTOR] LLM failed - cannot determine task completion, defaulting to NOT complete")
            task_complete = False
            # Force continuation
            return {
                "current_plan": None,
                "next_steps": [],
                "page_summary": summary[:1000],
                "last_screenshot_path": screenshot_path,
                "extracted_data": {
                    "summary": summary,
                    "content_length": len(page_content),
                    "screenshot_path": screenshot_path,
                    "llm_failed": True
                },
                "completion_status": "pending",
                "should_continue": True,  # Always continue when LLM failed
                "extraction_count": extraction_count,
                "last_update_time": time.time(),
                "messages": [AIMessage(content=summary)]
            }
        
        # Task completion logic:
        # Only mark complete if the model explicitly says complete, does not indicate missing work,
        # and we have evidence of meaningful successful actions.
        task_complete = bool(llm_says_complete) and (not llm_says_incomplete) and (meaningful_actions > 0)

        # If we keep extracting without any meaningful actions, force replanning (avoid "declare done" fallback).
        if extraction_count > 6 and meaningful_actions == 0:
            logger.warning(f"[EXTRACTOR] Too many extractions ({extraction_count}) without meaningful actions; forcing replan")
            task_complete = False
        
        # If task seems incomplete, continue
        if not task_complete:
            logger.info(f"[EXTRACTOR] Task not complete: LLM says complete={llm_says_complete}, "
                       f"meaningful_actions={meaningful_actions}, summary mentions incomplete={llm_says_incomplete}, "
                       f"extraction_count={extraction_count}")
        
        updates = {
            "page_summary": summary[:1000],
            "last_screenshot_path": screenshot_path,
            "extracted_data": {
                "summary": summary,
                "content_length": len(page_content),
                "screenshot_path": screenshot_path
            },
            "completion_status": "success" if task_complete else "pending",
            "should_continue": not task_complete,
            "extraction_count": extraction_count,  # Track extraction count
            "last_update_time": time.time(),
            "messages": [AIMessage(content=summary)]
        }
        
        # If task is not complete, keep the loop moving: trigger replanning after repeated extractions.
        if not task_complete and extraction_count > 3:
            updates["current_plan"] = None
            updates["next_steps"] = []
            updates["should_continue"] = True
        
        logger.info("Extractor: Analysis complete")
        return updates
        
    except Exception as e:
        logger.error(f"Extractor error: {e}")
        return {
            "last_error": f"Extractor failed: {str(e)}",
            "should_continue": True,
            "last_update_time": time.time()
        }


async def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    ErrorHandler node: Analyzes failures and decides on retry/alternative/escalation.
    """
    observer = get_observer()
    if observer:
        observer.log_agent_activation("ERROR_HANDLER", {
            "user_task": state.get("user_task", ""),
            "current_step": f"Analyzing error: {state.get('last_error', 'Unknown error')}",
            "retry_count": state.get("retry_count", 0)
        })

    logger.info("[ERROR_HANDLER] Analyzing failure and determining recovery strategy...")
    
    error = state.get("last_error", "Unknown error")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # Use LLM to analyze error and suggest action
    llm = get_llm()
    
    error_analysis_prompt = f"""
An error occurred during browser automation:

Error: {error}
Retry Count: {retry_count}/{max_retries}
Current URL: {state.get('current_url', 'Unknown')}
Last Step: {state.get('current_step', 'Unknown')}

Analyze the error and suggest:
1. What went wrong
2. Whether to retry (and how)
3. Alternative approach
4. Whether human intervention is needed

IMPORTANT:
- If the error indicates a loop/stagnation (same action repeated without progress), do NOT recommend repeating the same action again.
  Prefer ALTERNATIVE or ESCALATE.

Respond with: RETRY, ALTERNATIVE, or ESCALATE
"""
    
    try:
        messages = [HumanMessage(content=error_analysis_prompt)]

        # Log error analysis request
        if observer:
            observer.log_llm_reasoning(messages, "ERROR_HANDLER")

        response = await llm.ainvoke(messages)
        analysis = response.content if hasattr(response, 'content') else str(response)

        # Log error analysis response
        if observer and hasattr(response, 'content'):
            response_messages = messages + [AIMessage(content=response.content)]
            observer.log_llm_reasoning(response_messages, "ERROR_HANDLER")
        
        # Determine action
        if "RETRY" in analysis.upper() and retry_count < max_retries:
            action = "retry"
            next_node = "executor"
        elif "ALTERNATIVE" in analysis.upper():
            action = "alternative"
            next_node = "planner"  # Replan with alternative approach
        else:
            action = "escalate"
            next_node = None  # Stop and wait for human
        
        updates = {
            "last_error": f"{error} | Analysis: {analysis[:200]}",
            "should_retry": action == "retry",
            "should_escalate": action == "escalate",
            "should_continue": action != "escalate",
            "next_node": next_node,
            "retry_count": retry_count + 1 if action == "retry" else retry_count,
            "last_update_time": time.time(),
            "messages": [AIMessage(content=f"Error analysis: {analysis}")]
        }
        
        logger.info(f"ErrorHandler: Action = {action}")
        return updates
        
    except Exception as e:
        logger.error(f"ErrorHandler failed: {e}")
        return {
            "should_escalate": True,
            "should_continue": False,
            "completion_status": "failed",
            "last_update_time": time.time()
        }
