### This Toolkit is used to interact with the browser using the Browser Use project. 
### You may find more about the project here: https://github.com/browser-use/browser-use
### Documentation: https://docs.browser-use.com/quickstart
### 
### Requirements:
### - Python 3.11+: pip install browser-use
### - Python 3.10: pip install browser-use-py310x

import asyncio
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from .tool import Tool, Toolkit
from ..core.module import BaseModule
from ..core.logging import logger

# Load environment variables
load_dotenv()


class BrowserUseBase(BaseModule):
    """
    Base class for Browser Use interactions.
    Handles LLM setup, browser configuration, and async agent execution.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = os.getenv("OPENAI_API_KEY"), 
                 browser_type: str = "chromium", headless: bool = True, **kwargs):
        """
        Initialize the BrowserUse base.
        
        Args:
            model: LLM model to use (gpt-4o-mini, claude-3-5-sonnet, etc.)
            api_key: API key for the LLM (if not in environment)
            browser_type: Browser type (chromium, firefox, webkit)
            headless: Whether to run browser in headless mode
        """
        super().__init__(**kwargs)
        
        try:
            # Try importing from the standard browser-use package (Python 3.11+)
            from browser_use import Agent, BrowserProfile
            from browser_use.llm import ChatOpenAI, ChatAnthropic
            self.Agent = Agent
            self.BrowserProfile = BrowserProfile
            self.ChatOpenAI = ChatOpenAI
            self.ChatAnthropic = ChatAnthropic
        except ImportError:
            try:
                # Try importing from browser-use-py310x package (Python 3.10)
                from browser_use_py310x import Agent
                from browser_use_py310x.llm import ChatOpenAI, ChatAnthropic
                self.Agent = Agent
                self.BrowserProfile = None
                self.ChatOpenAI = ChatOpenAI
                self.ChatAnthropic = ChatAnthropic
            except ImportError as e:
                logger.error("browser-use package not installed. For Python 3.11+: pip install browser-use, For Python 3.10: pip install browser-use-py310x")
                raise ImportError(f"browser-use package required: {e}")
        
        self.model = model
        self.api_key = api_key
        self.browser_type = browser_type
        self.headless = headless

        if self.BrowserProfile is not None and browser_type != "chromium":
            raise ValueError("browser-use supports Chromium only; set browser_type='chromium'")
        
        # Initialize LLM based on model type
        self.llm = self._setup_llm()
        
        # Browser configuration
        self.browser_config = {"browser_type": browser_type, "headless": headless}
    
    def _setup_llm(self):
        """Setup the appropriate LLM based on model name."""
        try:
            if "gpt" in self.model.lower() or "openai" in self.model.lower():
                # OpenAI models
                kwargs = {"model": self.model}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                return self.ChatOpenAI(**kwargs)
            
            elif "claude" in self.model.lower() or "anthropic" in self.model.lower():
                # Anthropic models
                kwargs = {"model": self.model}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                return self.ChatAnthropic(**kwargs)
            
            else:
                # Default to OpenAI
                logger.warning(f"Unknown model {self.model}, defaulting to OpenAI")
                return self.ChatOpenAI(model=self.model)
                
        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}")
            raise
    
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """
        Execute a browser task using the Browser Use agent.
        
        Args:
            task: The task description for the browser agent
            
        Returns:
            Dictionary containing task results
        """
        try:
            # Create agent with configuration
            agent_kwargs = {"task": task, "llm": self.llm}
            if self.BrowserProfile is not None:
                # Current browser-use configures the browser through BrowserProfile.
                # Extra Agent kwargs are accepted but ignored, which previously made
                # headless=True silently launch a visible browser.
                agent_kwargs["browser_profile"] = self.BrowserProfile(headless=self.headless)
            else:
                # Keep the legacy Python 3.10 compatibility package on its old API.
                agent_kwargs.update(self.browser_config)

            agent = self.Agent(**agent_kwargs)
            
            # Execute the task
            logger.info(f"Executing browser task: {task}")
            result = await agent.run()
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Browser task failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_task_sync(self, task: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for execute_task.
        
        Args:
            task: The task description for the browser agent
            
        Returns:
            Dictionary containing task results
        """
        try:
            # Run the async task in a new event loop
            return asyncio.run(self.execute_task(task))
        except RuntimeError:
            # If we're already in an event loop, create a new task
            loop = asyncio.get_event_loop()
            task_coro = self.execute_task(task)
            return loop.run_until_complete(task_coro)


class BrowserUseTool(Tool):
    """Tool for executing browser automation tasks using natural language."""
    
    name: str = "browser_use"
    description: str = "Execute web browser automation tasks using natural language instructions"
    inputs: Dict[str, Dict[str, str]] = {
        "task": {
            "type": "string",
            "description": "Natural language description of the browser task to execute"
        }
    }
    required: Optional[List[str]] = ["task"]
    
    def __init__(self, browser_base: BrowserUseBase = None):
        super().__init__()
        self.browser_base = browser_base
    
    def __call__(self, task: str) -> Dict[str, Any]:
        """
        Execute a browser automation task.
        
        Args:
            task: Natural language task description
            
        Returns:
            Dictionary with task execution results
        """
        if not task.strip():
            return {
                "success": False,
                "error": "Task description cannot be empty"
            }
        
        return self.browser_base.execute_task_sync(task)


class BrowserUseToolkit(Toolkit):
    """Toolkit for browser automation using Browser Use."""
    
    def __init__(self, name: str = "BrowserUseToolkit", model: str = "gpt-4o-mini", 
                 api_key: str = None, browser_type: str = "chromium", 
                 headless: bool = True):
        """
        Initialize the BrowserUse toolkit.
        
        Args:
            name: Toolkit name
            model: LLM model to use
            api_key: API key for the LLM
            browser_type: Browser type (chromium, firefox, webkit)
            headless: Whether to run browser in headless mode
        """
        # Create the shared browser base instance
        browser_base = BrowserUseBase(
            model=model,
            api_key=api_key,
            browser_type=browser_type,
            headless=headless
        )
        
        # Create tools with the shared base
        tools = [
            BrowserUseTool(browser_base=browser_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store browser_base as instance variable
        self.browser_base = browser_base
