"""
Browser Automation Tools using Playwright

All tools are defined as LangChain StructuredTool instances for use with LLM function calling.
"""

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from playwright.async_api import Page, BrowserContext, Browser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# Global browser context (initialized in main)
_browser: Optional[Browser] = None
_browser_context: Optional[BrowserContext] = None
_current_page: Optional[Page] = None
_current_task: Optional[str] = None  # Current user task for context-aware element selection


def set_browser_context(browser: Optional[Browser], context: BrowserContext, page: Page):
    """Set the global browser context for tools to use."""
    global _browser, _browser_context, _current_page
    _browser = browser
    _browser_context = context
    _current_page = page


def set_current_task(task: str):
    """Set the current user task for context-aware element selection."""
    global _current_task
    _current_task = task


def get_current_task() -> Optional[str]:
    """Get the current user task."""
    return _current_task


def get_page() -> Page:
    """Get the current page instance."""
    if _current_page is None:
        raise RuntimeError("Browser context not initialized. Call set_browser_context first.")
    return _current_page


# Tool Input Schemas
class NavigateInput(BaseModel):
    url: str = Field(description="The URL to navigate to")


class ClickInput(BaseModel):
    element_description: str = Field(description="Natural language description of the element to click (e.g., 'Submit button', 'Login link', 'Search field')")


class TypeInput(BaseModel):
    field_description: str = Field(description="Natural language description of the input field")
    text: str = Field(description="Text to type into the field")


class ExtractContentInput(BaseModel):
    query: str = Field(description="What content to extract (e.g., 'product prices', 'article text', 'form fields')")


class ScrollInput(BaseModel):
    direction: str = Field(description="Direction to scroll: 'up', 'down', 'left', 'right'")
    amount: int = Field(default=500, description="Number of pixels to scroll")


class HandlePopupInput(BaseModel):
    action: str = Field(description="Action to take: 'accept', 'dismiss', 'ignore'")


# Tool Implementations
async def navigate_tool(url: str) -> Dict[str, Any]:
    """
    Navigate to a URL.
    
    Args:
        url: URL to navigate to
        
    Returns:
        Dict with success status and current URL
    """
    try:
        page = get_page()
        
        # Verify page is ready
        if not page:
            return {
                "success": False,
                "error": "Browser page not initialized",
                "message": "Browser page is not available. Please ensure browser is initialized."
            }
        
        # Use "load" instead of "networkidle" for faster page loads
        await page.goto(url, wait_until="load", timeout=15000)
        current_url = page.url
        
        return {
            "success": True,
            "current_url": current_url,
            "message": f"Navigated to {current_url}"
        }
    except RuntimeError as e:
        # Browser not initialized
        return {
            "success": False,
            "error": f"Browser not initialized: {str(e)}",
            "message": f"Browser is not initialized. Error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to navigate to {url}: {str(e)}"
        }


async def wait_for_page_stable(page, timeout: int = 2000) -> bool:
    """
    Wait for the page to become stable (no network activity, content loaded).
    Returns True if page became stable, False on timeout.
    Reduced timeout for faster interaction.
    """
    try:
        # Wait for DOM content first (faster)
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)
        return True
    except:
        pass
    
    try:
        # Then try network idle
        await page.wait_for_load_state("networkidle", timeout=timeout)
        return True
    except:
        pass
    
    # Last resort: just wait a bit
    await asyncio.sleep(0.3)
    return False


async def get_interactive_elements(page, max_elements: int = 100, wait_for_load: bool = True) -> List[Dict[str, Any]]:
    """
    Extract interactive elements from the page for LLM analysis.
    Returns a list of all clickable elements - the LLM decides which to interact with.
    
    Args:
        page: Playwright page object
        max_elements: Maximum number of elements to return
        wait_for_load: Whether to wait for page to stabilize first
    """
    elements = []
    
    # Wait for page to stabilize before searching for elements
    if wait_for_load:
        await wait_for_page_stable(page, timeout=2000)
    
    # Get all interactive elements using JavaScript
    # Only generic selectors - no site-specific ones
    js_code = """
    () => {
        const results = [];
        const seenCount = new Map();
        
        // Generic selectors for any interactive elements (no site-specific selectors!)
        const selectors = [
            'a[href]',
            'button',
            '[role="button"]',
            '[role="link"]',
            '[role="tab"]',
            '[role="menuitem"]',
            '[role="option"]',
            '[role="listitem"]',
            '[role="row"]',
            '[onclick]',
            'input[type="submit"]',
            'input[type="button"]',
            '[tabindex]'
        ].join(', ');
        
        const allElements = document.querySelectorAll(selectors);
        
        for (const el of allElements) {
            // Skip hidden elements
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) continue;
            
            const style = window.getComputedStyle(el);
            if (style.display === 'none') continue;
            if (style.visibility === 'hidden') continue;
            if (style.opacity === '0') continue;
            
            // Include elements up to 5x viewport height (to capture content below fold)
            if (rect.top > window.innerHeight * 5) continue;
            
            // Get element text
            let text = (el.innerText || el.textContent || '').trim();
            text = text.replace(/\\s+/g, ' ').substring(0, 100);
            
            if (!text) text = el.getAttribute('aria-label') || el.getAttribute('title') || '';
            if (!text) {
                const img = el.querySelector('img');
                if (img) text = img.getAttribute('alt') || '';
            }
            if (!text) continue;
            
            // Skip JSON-like content
            if (text.includes('{"') || text.includes('widgets')) continue;
            
            // De-duplicate but allow a few duplicates for common repeated labels (e.g., many identical buttons).
            // Without this, the agent can't click "second/third" of identical buttons.
            const key = text.substring(0, 50).toLowerCase();
            const prev = seenCount.get(key) || 0;
            const next = prev + 1;
            seenCount.set(key, next);
            // Keep up to 6 occurrences per label (enough for "second" selection, but avoids huge lists)
            if (next > 6) continue;
            
            const tagName = el.tagName.toLowerCase();
            const href = el.getAttribute('href') || '';
            const role = el.getAttribute('role') || '';
            
            results.push({
                index: results.length,
                tag: tagName,
                text: text.substring(0, 80),
                href: href ? href.substring(0, 100) : '',
                role: role,
                y: Math.round(rect.top)
            });
            
            if (results.length >= """ + str(max_elements) + """) break;
        }
        
        // Sort by vertical position (top to bottom)
        results.sort((a, b) => a.y - b.y);
        results.forEach((el, idx) => el.index = idx);
        
        return results;
    }
    """
    
    try:
        elements = await page.evaluate(js_code)
    except Exception as e:
        pass
    
    return elements


async def ask_llm_for_element(element_description: str, elements: List[Dict], page_url: str, task_context: str = None) -> Optional[int]:
    """
    Ask LLM to identify which element matches the description.
    Returns the index of the matching element, or None if not found.
    
    Args:
        element_description: What element to click
        elements: List of available elements
        page_url: Current page URL
        task_context: Original user task for context (helps LLM pick relevant items)
    """
    from src.graph.nodes import get_llm
    
    if not elements:
        return None
    
    # Format elements for LLM
    elements_text = "\n".join([
        f"{el['index']}: [{el['tag']}] {el['text']}" + (f" -> {el['href']}" if el['href'] else "")
        for el in elements
    ])
    
    # Build context-aware prompt (generic, no domain/site-specific hints)
    context_section = ""
    if task_context:
        context_section = f"""
IMPORTANT TASK CONTEXT: The user's original task is: "{task_context}"
Prefer elements that are most likely to advance the task on the current page.
"""
    
    prompt = f"""You are helping a browser automation agent find the right element to click.

Page URL: {page_url}
{context_section}
The user wants to click on: "{element_description}"

NOTE: The elements below are ordered from top to bottom on the screen.

Here are the available clickable elements on the page:
{elements_text}

Which element (by index number) best matches what the user wants to click?

RULES:
1. If the description contains quoted text (like "..." / '...' / «...»), prefer an element whose visible text closely matches that quoted text.
2. If looking for "first/second/third item" (e.g., first email/result), choose that ordinal item among the main list/content items (often rows/list items), not global navigation.
3. Avoid random/unrelated elements; if nothing reasonably matches, respond with -1.

If no element matches the criteria, respond with -1.

Respond with ONLY the index number, nothing else."""

    try:
        llm = get_llm()
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse the response to get the index
        response_text = response.content.strip()
        # Extract first number from response
        match = re.search(r'-?\d+', response_text)
        if match:
            return int(match.group())
    except Exception as e:
        pass
    
    return None


async def click_tool(element_description: str) -> Dict[str, Any]:
    """
    Click an element based on natural language description.
    
    Uses LLM to analyze the page and determine which element to click.
    No hardcoded selectors - the LLM decides based on page content.
    
    Args:
        element_description: Natural language description of element
        
    Returns:
        Dict with success status and action taken
    """
    try:
        page = get_page()
        desc_lower = element_description.lower()

        def _ordinal_index(text: str) -> Optional[int]:
            t = (text or "").lower()
            # English
            if any(x in t for x in [" first", "1st", " first "]) or t.startswith("first") or "перв" in t:
                return 0
            if any(x in t for x in [" second", "2nd", " second "]) or t.startswith("second") or "втор" in t:
                return 1
            if any(x in t for x in [" third", "3rd", " third "]) or t.startswith("third") or "трет" in t:
                return 2
            return None

        ord_idx = _ordinal_index(desc_lower)
        
        # Special case: browser back navigation
        if any(kw in desc_lower for kw in ["back", "назад", "вернуться", "return"]):
            try:
                old_url = page.url
                await page.go_back(timeout=5000)
                # Wait for new page to load
                await wait_for_page_stable(page, timeout=3000)
                new_url = page.url
                
                # Validate that we actually went back (URL changed)
                if new_url == old_url:
                    return {
                        "success": False, 
                        "method": "browser_back",
                        "error": "URL did not change after going back",
                        "message": f"Browser back did not change URL (still on {old_url})"
                    }
                
                return {
                    "success": True, 
                    "method": "browser_back", 
                    "message": f"Used browser back navigation: {old_url} -> {new_url}",
                    "old_url": old_url,
                    "new_url": new_url
                }
            except Exception as e:
                return {
                    "success": False,
                    "method": "browser_back",
                    "error": str(e),
                    "message": f"Browser back failed: {str(e)}"
                }

        # Step 0: If the description contains a quoted label, try that label directly (fast, generic)
        # Supports ", ', and «» quotes.
        try:
            quoted = []
            quoted += re.findall(r'["“”](.{1,80}?)["“”]', element_description)
            quoted += re.findall(r"['](.{1,80}?)[']", element_description)
            quoted += re.findall(r"[«](.{1,80}?)[»]", element_description)
            quoted = [q.strip() for q in quoted if q and q.strip()]
            # De-duplicate while preserving order
            seen_q = set()
            quoted = [q for q in quoted if not (q.lower() in seen_q or seen_q.add(q.lower()))]

            for q in quoted[:3]:
                # Try common interactive roles first
                for role in ["button", "link", "tab", "menuitem", "option", "checkbox"]:
                    try:
                        loc = page.get_by_role(role, name=q, exact=False)
                        cnt = await loc.count()
                        if cnt > 0:
                            target = loc.nth(ord_idx) if (ord_idx is not None and cnt > ord_idx) else loc.first
                            await target.click(timeout=5000)
                            if ord_idx is not None and cnt > ord_idx:
                                return {"success": True, "method": f"quoted_role_{role}_nth", "message": f"Clicked {role} #{ord_idx+1} containing: {q}"}
                            return {"success": True, "method": f"quoted_role_{role}", "message": f"Clicked {role} containing: {q}"}
                    except:
                        continue

                # Fallback: any element by text
                try:
                    loc = page.get_by_text(q, exact=False)
                    cnt = await loc.count()
                    if cnt > 0:
                        target = loc.nth(ord_idx) if (ord_idx is not None and cnt > ord_idx) else loc.first
                        await target.click(timeout=5000)
                        if ord_idx is not None and cnt > ord_idx:
                            return {"success": True, "method": "quoted_text_nth", "message": f"Clicked element #{ord_idx+1} containing text: {q}"}
                        return {"success": True, "method": "quoted_text", "message": f"Clicked element containing text: {q}"}
                except:
                    pass
        except:
            pass
        
        # Step 1: Try simple text matching first (fast path)
        try:
            loc = page.get_by_text(element_description, exact=False)
            cnt = await loc.count()
            if cnt > 0:
                target = loc.nth(ord_idx) if (ord_idx is not None and cnt > ord_idx) else loc.first
                await target.click(timeout=5000)
                if ord_idx is not None and cnt > ord_idx:
                    return {"success": True, "method": "text_match_nth", "message": f"Clicked element #{ord_idx+1} with text: {element_description}"}
                return {"success": True, "method": "text_match", "message": f"Clicked element with text: {element_description}"}
        except:
            pass
        
        # Step 2: Try role-based selection
        for role in ["button", "link"]:
            try:
                loc = page.get_by_role(role, name=element_description, exact=False)
                cnt = await loc.count()
                if cnt > 0:
                    target = loc.nth(ord_idx) if (ord_idx is not None and cnt > ord_idx) else loc.first
                    await target.click(timeout=5000)
                    if ord_idx is not None and cnt > ord_idx:
                        return {"success": True, "method": f"role_{role}_nth", "message": f"Clicked {role} #{ord_idx+1}: {element_description}"}
                    return {"success": True, "method": f"role_{role}", "message": f"Clicked {role}: {element_description}"}
            except:
                continue
        
        # Step 3: Use LLM to analyze page and find the element
        current_url = page.url
        # Try without waiting first (faster); if too few elements, retry after a short stabilization wait.
        elements = await get_interactive_elements(page, wait_for_load=False)
        if not elements or len(elements) < 5:
            elements = await get_interactive_elements(page, wait_for_load=True)
        
        if elements:
            # Ask LLM which element to click, with task context for relevance
            task_context = get_current_task()
            element_index = await ask_llm_for_element(element_description, elements, current_url, task_context)
            
            if element_index is not None and element_index >= 0 and element_index < len(elements):
                selected = elements[element_index]
                
                # Click the element by its text
                try:
                    element = page.get_by_text(selected['text'], exact=False).first
                    if await element.count() > 0:
                        await element.click(timeout=5000)
                        return {
                            "success": True, 
                            "method": "llm_selected", 
                            "message": f"LLM selected element {element_index}: {selected['text'][:50]}"
                        }
                except:
                    pass
                
                # Fallback: try by tag and text
                try:
                    tag = selected['tag']
                    text = selected['text']
                    element = page.locator(f"{tag}").filter(has_text=text[:30]).first
                    if await element.count() > 0:
                        await element.click(timeout=5000)
                        return {
                            "success": True,
                            "method": "llm_selected_tag",
                            "message": f"LLM selected [{tag}]: {text[:50]}"
                        }
                except:
                    pass

            # Generic ordinal fallback: if user asked for "first/second/third" and LLM couldn't decide,
            # click the first plausible list/content item (row/listitem/option) visible on the page.
            if element_index is not None and element_index < 0:
                ordinals = ["first", "second", "third", "1st", "2nd", "3rd", "перв", "втор", "трет"]
                if any(o in desc_lower for o in ordinals):
                    candidates = [
                        el for el in elements
                        if (el.get("role") in {"row", "listitem", "option"} or el.get("tag") in {"tr", "li"})
                        and el.get("text") and len(el.get("text", "")) >= 10
                    ]
                    if candidates:
                        selected = candidates[0]
                        try:
                            element = page.get_by_text(selected["text"], exact=False).first
                            if await element.count() > 0:
                                await element.click(timeout=5000)
                                return {
                                    "success": True,
                                    "method": "ordinal_fallback",
                                    "message": f"Clicked first list item: {selected['text'][:50]}"
                                }
                        except:
                            pass

        return {
            "success": False,
            "error": "Element not found",
            "message": f"Could not find element matching: {element_description}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to click element: {str(e)}"
        }


async def get_input_fields(page, max_fields: int = 20) -> List[Dict[str, Any]]:
    """
    Extract input fields from the page for LLM analysis.
    """
    js_code = """
    () => {
        const results = [];
        const inputs = document.querySelectorAll('input, textarea, [contenteditable="true"]');
        
        for (const el of inputs) {
            // Skip hidden elements
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) continue;
            if (window.getComputedStyle(el).display === 'none') continue;
            
            const type = el.getAttribute('type') || 'text';
            if (['hidden', 'submit', 'button', 'checkbox', 'radio'].includes(type)) continue;
            
            // Get field description
            const placeholder = el.getAttribute('placeholder') || '';
            const ariaLabel = el.getAttribute('aria-label') || '';
            const name = el.getAttribute('name') || '';
            const id = el.getAttribute('id') || '';
            
            // Try to find label
            let label = '';
            if (id) {
                const labelEl = document.querySelector(`label[for="${id}"]`);
                if (labelEl) label = labelEl.innerText || '';
            }
            
            const description = placeholder || ariaLabel || label || name || `${el.tagName} field`;
            
            results.push({
                index: results.length,
                tag: el.tagName.toLowerCase(),
                type: type,
                description: description.substring(0, 80),
                placeholder: placeholder.substring(0, 50)
            });
            
            if (results.length >= """ + str(max_fields) + """) break;
        }
        return results;
    }
    """
    
    try:
        return await page.evaluate(js_code)
    except:
        return []


async def ask_llm_for_input_field(field_description: str, fields: List[Dict], page_url: str) -> Optional[int]:
    """
    Ask LLM to identify which input field matches the description.
    """
    from src.graph.nodes import get_llm
    
    if not fields:
        return None
    
    fields_text = "\n".join([
        f"{f['index']}: [{f['tag']}] {f['description']}" + (f" (placeholder: {f['placeholder']})" if f['placeholder'] else "")
        for f in fields
    ])
    
    prompt = f"""You are helping a browser automation agent find the right input field.

Page URL: {page_url}

The user wants to type into: "{field_description}"

Here are the available input fields on the page:
{fields_text}

Which field (by index number) best matches what the user wants?
For "search field", look for fields with search-related descriptions.
If no field matches, respond with -1.

Respond with ONLY the index number, nothing else."""

    try:
        llm = get_llm()
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        match = re.search(r'-?\d+', response.content.strip())
        if match:
            return int(match.group())
    except:
        pass
    
    return None


async def type_into_tool(field_description: str, text: str) -> Dict[str, Any]:
    """
    Type text into an input field.
    
    Uses LLM to analyze the page and determine which field to use.
    No hardcoded selectors.
    
    Args:
        field_description: Description of the input field
        text: Text to type
        
    Returns:
        Dict with success status
    """
    try:
        page = get_page()
        
        # Step 1: Try semantic locators first (fast path)
        # By placeholder
        try:
            element = page.get_by_placeholder(field_description, exact=False).first
            if await element.count() > 0:
                await element.fill(text)
                return {"success": True, "message": f"Typed into field: {field_description}"}
        except:
            pass
        
        # By label
        try:
            element = page.get_by_label(field_description, exact=False).first
            if await element.count() > 0:
                await element.fill(text)
                return {"success": True, "message": f"Typed into field: {field_description}"}
        except:
            pass
        
        # By role
        try:
            element = page.get_by_role("textbox", name=field_description, exact=False).first
            if await element.count() > 0:
                await element.fill(text)
                return {"success": True, "message": f"Typed into textbox: {field_description}"}
        except:
            pass
        
        # By role (searchbox)
        try:
            element = page.get_by_role("searchbox").first
            if await element.count() > 0:
                await element.fill(text)
                return {"success": True, "message": f"Typed into searchbox: {field_description}"}
        except:
            pass

        # Step 2: Use LLM to analyze page and find the right field
        current_url = page.url
        fields = await get_input_fields(page)
        
        if fields:
            field_index = await ask_llm_for_input_field(field_description, fields, current_url)
            
            if field_index is not None and field_index >= 0 and field_index < len(fields):
                selected = fields[field_index]
                
                # Try to find and fill the field
                try:
                    # Use placeholder if available
                    if selected['placeholder']:
                        element = page.get_by_placeholder(selected['placeholder'], exact=False).first
                        if await element.count() > 0:
                            await element.click(timeout=2000)
                            await element.fill(text)
                            return {"success": True, "method": "llm_selected", "message": f"LLM selected field: {selected['description'][:50]}"}
                except:
                    pass
                
                # Fallback: find visible inputs and use index
                try:
                    inputs = page.locator("input:visible, textarea:visible")
                    count = await inputs.count()
                    if count > field_index:
                        element = inputs.nth(field_index)
                        await element.click(timeout=2000)
                        await element.fill(text)
                        return {"success": True, "method": "llm_selected_index", "message": f"LLM selected input #{field_index}"}
                except:
                    pass
        
        # Step 3: Last resort - first visible input
        try:
            element = page.locator("input:visible, textarea:visible").first
            if await element.count() > 0:
                await element.click(timeout=2000)
                await element.fill(text)
                return {"success": True, "message": f"Typed into first visible input: {field_description}"}
        except:
            pass
        
        return {
            "success": False,
            "error": "Field not found",
            "message": f"Could not find input field: {field_description}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to type into field: {str(e)}"
        }


async def extract_content_tool(query: str) -> Dict[str, Any]:
    """
    Extract and summarize relevant content from the current page.
    
    Args:
        query: What content to extract
        
    Returns:
        Dict with extracted content
    """
    try:
        page = get_page()

        # IMPORTANT:
        # Do NOT return full HTML (or a markdownified full page) to the LLM.
        # We only return a bounded "visible text" snapshot, optionally filtered by query keywords.
        # This enforces token constraints and prevents leaking entire pages into context.

        q = (query or "").strip()
        if not q or q.lower() == "all":
            q = "main content"

        # Get a bounded snapshot of visible text from the page (viewport + a few screens below).
        snapshot = await page.evaluate(
            """
            (args) => {
              const { maxLines, maxLineLen, maxScreens } = args;
              const isVisibleElement = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') return false;
                if (style.opacity === '0') return false;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return false;
                if (rect.bottom < 0) return false;
                if (rect.top > window.innerHeight * maxScreens) return false;
                return true;
              };

              const lines = [];
              const seen = new Set();

              const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                {
                  acceptNode: (node) => {
                    const parent = node.parentElement;
                    if (!parent) return NodeFilter.FILTER_REJECT;
                    if (!isVisibleElement(parent)) return NodeFilter.FILTER_REJECT;
                    let t = (node.textContent || '').trim();
                    if (t.length < 2) return NodeFilter.FILTER_REJECT;
                    // Skip obvious non-human blobs (very long uninterrupted strings)
                    if (t.length > 500 && !t.includes(' ')) return NodeFilter.FILTER_REJECT;
                    return NodeFilter.FILTER_ACCEPT;
                  }
                }
              );

              let node;
              while ((node = walker.nextNode()) && lines.length < maxLines) {
                let t = (node.textContent || '').trim();
                t = t.replace(/\\s+/g, ' ');
                if (!t) continue;
                if (t.length > maxLineLen) t = t.slice(0, maxLineLen);
                const key = t.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                lines.push(t);
              }

              return lines.join('\\n');
            }
            """,
            {"maxLines": 120, "maxLineLen": 220, "maxScreens": 8},
        )

        text = (snapshot or "").strip()
        if not text:
            return {
                "success": True,
                "content": "",
                "length": 0,
                "query": q,
                "message": "No visible text found on the current page."
            }

        # Lightweight query filtering (no HTML, only text).
        # Keep lines that match at least one keyword; if nothing matches, return the unfiltered snapshot.
        keywords = [k for k in re.split(r"\\s+", q.lower()) if len(k) >= 3]
        if keywords:
            kept = []
            for line in text.splitlines():
                lower = line.lower()
                if any(k in lower for k in keywords):
                    kept.append(line)
                if len(kept) >= 80:
                    break
            if kept:
                text = "\n".join(kept)

        # Hard cap to protect token budget (characters, not tokens).
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Truncated]"
        
        return {
            "success": True,
            "content": text,
            "length": len(text),
            "query": q
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to extract content: {str(e)}"
        }


async def take_screenshot_tool() -> Dict[str, Any]:
    """
    Take a screenshot of the current page.
    
    Returns:
        Dict with screenshot path and base64 data
    """
    try:
        page = get_page()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = Path("recordings/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        screenshot_path = screenshot_dir / f"screenshot_{timestamp}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        
        # Also get base64 for vision API
        screenshot_bytes = await page.screenshot(full_page=True)
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        return {
            "success": True,
            "path": str(screenshot_path),
            "base64": screenshot_b64,
            "message": f"Screenshot saved to {screenshot_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to take screenshot: {str(e)}"
        }


async def scroll_tool(direction: str, amount: int = 500) -> Dict[str, Any]:
    """
    Scroll the page in a direction.
    
    Args:
        direction: 'up', 'down', 'left', 'right'
        amount: Pixels to scroll
        
    Returns:
        Dict with success status
    """
    try:
        page = get_page()
        
        if direction.lower() == "down":
            await page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction.lower() == "up":
            await page.evaluate(f"window.scrollBy(0, -{amount})")
        elif direction.lower() == "right":
            await page.evaluate(f"window.scrollBy({amount}, 0)")
        elif direction.lower() == "left":
            await page.evaluate(f"window.scrollBy(-{amount}, 0)")
        else:
            return {"success": False, "error": f"Invalid direction: {direction}"}
        
        await asyncio.sleep(0.5)  # Wait for scroll to complete
        
        return {
            "success": True,
            "direction": direction,
            "amount": amount,
            "message": f"Scrolled {direction} by {amount}px"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to scroll: {str(e)}"
        }


async def handle_popup_tool(action: str) -> Dict[str, Any]:
    """
    Handle browser dialogs/popups.
    
    Args:
        action: 'accept', 'dismiss', 'ignore'
        
    Returns:
        Dict with success status
    """
    try:
        page = get_page()
        
        # Set up dialog handler
        dialog_handled = False
        
        def handle_dialog(dialog):
            nonlocal dialog_handled
            if action == "accept":
                dialog.accept()
            elif action == "dismiss":
                dialog.dismiss()
            # else ignore
            dialog_handled = True
        
        page.on("dialog", handle_dialog)
        
        # Wait a bit to see if dialog appears
        await asyncio.sleep(1)
        
        return {
            "success": True,
            "action": action,
            "handled": dialog_handled,
            "message": f"Popup handler set to {action}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to handle popup: {str(e)}"
        }


async def get_current_url_tool() -> Dict[str, Any]:
    """
    Get the current page URL.
    
    Returns:
        Dict with current URL
    """
    try:
        page = get_page()
        url = page.url
        return {
            "success": True,
            "url": url,
            "message": f"Current URL: {url}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to get URL: {str(e)}"
        }


# Create LangChain tools
# Using async functions directly - LangChain's StructuredTool automatically handles async via ainvoke()

navigate = StructuredTool.from_function(
    func=navigate_tool,
    name="navigate",
    description="Navigate to a URL. Use this to go to a new webpage.",
    args_schema=NavigateInput
)

click = StructuredTool.from_function(
    func=click_tool,
    name="click",
    description="Click an element on the page. Provide a natural language description of what to click (e.g., 'Submit button', 'Login link', 'Search button').",
    args_schema=ClickInput
)

type_into = StructuredTool.from_function(
    func=type_into_tool,
    name="type_into",
    description="Type text into an input field. Provide a description of the field (e.g., 'Email field', 'Search box', 'Search field').",
    args_schema=TypeInput
)

extract_content = StructuredTool.from_function(
    func=extract_content_tool,
    name="extract_content",
    description="Extract and summarize content from the current page. Specify what to extract (e.g., 'product prices', 'article text').",
    args_schema=ExtractContentInput
)

take_screenshot = StructuredTool.from_function(
    func=take_screenshot_tool,
    name="take_screenshot",
    description="Take a screenshot of the current page. Returns path and base64 data for vision analysis."
)

scroll = StructuredTool.from_function(
    func=scroll_tool,
    name="scroll",
    description="Scroll the page. Direction: 'up', 'down', 'left', 'right'. Amount: pixels to scroll.",
    args_schema=ScrollInput
)

handle_popup = StructuredTool.from_function(
    func=handle_popup_tool,
    name="handle_popup",
    description="Handle browser dialogs/popups. Action: 'accept', 'dismiss', 'ignore'.",
    args_schema=HandlePopupInput
)

get_current_url = StructuredTool.from_function(
    func=get_current_url_tool,
    name="get_current_url",
    description="Get the current page URL."
)


# Export all tools
BROWSER_TOOLS = [
    navigate,
    click,
    type_into,
    extract_content,
    take_screenshot,
    scroll,
    handle_popup,
    get_current_url
]
