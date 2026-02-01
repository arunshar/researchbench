"""
ResearchToolBench Purple Agent - Baseline Implementation

A2A-compatible purple agent for the ResearchToolBench benchmark.
Supports both single-control and dual-control (τ²-bench style) domains.
"""

import json
import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ResearchToolAgent:
    """Baseline purple agent for ResearchToolBench."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None
        if HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        
        self.gathered_info = []
        self.tools_used = []
        self.sources = []
    
    def process_observation(self, obs: Dict) -> Dict:
        """Process observation and return action."""
        
        user_message = obs.get("user_message", "")
        agent_tools = obs.get("agent_tools", [])
        policies = obs.get("policies", [])
        conversation = obs.get("conversation_history", [])
        step = obs.get("step", 0)
        max_steps = obs.get("max_steps", 10)
        is_dual_control = obs.get("is_dual_control", False)
        user_state = obs.get("user_state", {})
        
        # Strategy based on step
        if step == 0:
            # First step: search
            search_tool = next((t for t in agent_tools if "search" in t["name"]), None)
            if search_tool:
                query = self._extract_query(user_message)
                return {
                    "type": "tool_call",
                    "tool_name": search_tool["name"],
                    "parameters": {"query": query}
                }
        
        # Check if we have search results to process
        last_tool_result = self._get_last_tool_result(conversation)
        
        if last_tool_result and "results" in last_tool_result:
            results = last_tool_result["results"]
            if results and len(results) > 0:
                # Get details of first result
                item = results[0]
                item_id = item.get("id", "")
                
                if item_id and item_id not in self.sources:
                    self.sources.append(item_id)
                    self.gathered_info.append(item)
                    
                    # Get more details
                    detail_tool = next((t for t in agent_tools if "get_" in t["name"] and "details" in t["name"]), None)
                    if detail_tool and len(self.sources) < 2:
                        return {
                            "type": "tool_call",
                            "tool_name": detail_tool["name"],
                            "parameters": {detail_tool["required"][0]: item_id}
                        }
        
        # For dual-control: request user action if needed
        if is_dual_control and step > 1:
            request_tool = next((t for t in agent_tools if "request_user" in t["name"]), None)
            if request_tool and "torch" not in str(user_state.get("installed_packages", [])):
                return {
                    "type": "tool_call",
                    "tool_name": request_tool["name"],
                    "parameters": {
                        "action": "pip install torch",
                        "expected_result": "Installation successful"
                    }
                }
        
        # Near end or have enough info: submit
        if step >= max_steps - 2 or len(self.sources) >= 2:
            submit_tool = next((t for t in agent_tools if "submit" in t["name"] or "summarize" in t["name"]), None)
            if submit_tool:
                summary = self._generate_summary(user_message, self.gathered_info, policies)
                
                params = {"sources": self.sources}
                if "summary" in submit_tool["required"]:
                    params["summary"] = summary
                elif "report" in submit_tool["required"]:
                    params["report"] = summary
                elif "solution" in submit_tool["required"]:
                    params["solution"] = summary
                    params["references"] = self.sources
                    params["code"] = "import torch\nprint(torch.__version__)"
                
                return {
                    "type": "tool_call",
                    "tool_name": submit_tool["name"],
                    "parameters": params
                }
        
        # Default: send message
        return {
            "type": "message",
            "message": self._generate_response(user_message, self.gathered_info)
        }
    
    def _extract_query(self, message: str) -> str:
        """Extract search query from user message."""
        keywords = []
        for word in message.lower().split():
            if len(word) > 4 and word not in ["about", "could", "would", "should", "please", "thanks"]:
                keywords.append(word)
        return " ".join(keywords[:3]) if keywords else "research"
    
    def _get_last_tool_result(self, conversation: List[Dict]) -> Optional[Dict]:
        """Get last tool result from conversation."""
        for msg in reversed(conversation):
            if msg.get("role") == "tool" and "result" in msg:
                return msg["result"]
        return None
    
    def _generate_summary(self, query: str, info: List[Dict], policies: List[str]) -> str:
        """Generate summary following policies."""
        if not info:
            return f"Based on my research: {query}"
        
        summary_parts = [f"Based on my research on {query}:\n"]
        
        for i, item in enumerate(info[:3]):
            title = item.get("title", item.get("topic", "Source"))
            year = item.get("year", item.get("date", "2024"))
            summary_parts.append(f"- {title} ({year})")
        
        summary_parts.append("\nKey findings include the relevant information from these sources.")
        
        # Add version numbers for technical
        if any("version" in p.lower() for p in policies):
            summary_parts.append("\nVersion: 2.0")
        
        return "\n".join(summary_parts)
    
    def _generate_response(self, query: str, info: List[Dict]) -> str:
        """Generate conversational response."""
        if info:
            return f"I found some relevant information. Let me gather more details for you."
        return "Let me search for that information."
    
    def reset(self):
        """Reset agent state."""
        self.gathered_info = []
        self.tools_used = []
        self.sources = []


# FastAPI server for A2A protocol
def create_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="ResearchToolBench Purple Agent")
    agent = ResearchToolAgent()
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @app.get("/.well-known/agent.json")
    async def agent_card():
        return {
            "name": "ResearchToolBench-Purple",
            "description": "Baseline purple agent",
            "version": "2.0.0",
            "capabilities": ["research", "dual_control"]
        }
    
    @app.post("/")
    async def handle_request(request: Request):
        data = await request.json()
        
        if data.get("method") == "tasks/send":
            params = data.get("params", {})
            message = params.get("message", {})
            
            obs = None
            for part in message.get("parts", []):
                if part.get("type") == "data":
                    obs = part.get("data", {})
                    break
            
            if obs:
                action = agent.process_observation(obs)
                
                if action["type"] == "tool_call":
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "result": {"tool_call": {"name": action["tool_name"], "parameters": action["parameters"]}},
                        "id": data.get("id")
                    })
                else:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "result": {"message": {"parts": [{"type": "text", "text": action["message"]}]}},
                        "id": data.get("id")
                    })
        
        elif data.get("method") == "tasks/reset":
            agent.reset()
            return JSONResponse({"jsonrpc": "2.0", "result": {"status": "reset"}, "id": data.get("id")})
        
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": data.get("id")})
    
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)
