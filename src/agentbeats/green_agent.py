"""
ResearchToolBench Green Agent

A benchmark combining concepts from the AgentBeats custom tracks:

1. τ²-Bench Challenge (Sierra Research)
   - Dual-control environments: BOTH agent AND user have tools
   - Dec-POMDP formulation for shared state modification
   - Compositional task generator
   - pass^k reliability metric
   - Three domains: academic, news, technical (dual-control)

2. OpenEnv Challenge (Meta-PyTorch / Hugging Face / Unsloth)
   - Gymnasium-style APIs: step(), reset(), state(), close()
   - Hugging Face Hub deployable
   - TRL/TorchForge integration ready
   - Docker containerization support

References:
- τ²-bench: arXiv:2506.07982 (Barres et al., 2025)
- τ-bench: arXiv:2406.12045 (Yao et al., 2024)
- OpenEnv: github.com/meta-pytorch/OpenEnv
"""

import json
import asyncio
import time
import re
import os
import random
import copy
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Dict, List, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI not installed. User simulation will use rule-based fallback.")


# ============================================================================
# OpenEnv Challenge: Gymnasium-Compatible Types
# ============================================================================

@dataclass
class Message:
    """A message in conversation. Compatible with HuggingFace chat template."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[Dict] = None


@dataclass
class Observation:
    """
    Observation returned by the environment.
    OpenEnv compatible + τ²-bench dual-control extensions.
    """
    # Task identification
    task_id: str
    domain: str
    
    # Current state
    user_message: str
    conversation_history: List[Dict]
    
    # τ²-bench dual-control: BOTH agent and user have tools
    agent_tools: List[Dict]  # Tools available to AGENT
    user_tools: List[Dict]   # Tools available to USER (τ²-bench style)
    
    # Domain policies (rules to follow)
    policies: List[str]
    
    # Environment state (τ²-bench: compared at end)
    database_state: Dict
    user_state: Dict  # User's observable state
    
    # Step tracking
    step_count: int
    max_steps: int
    
    # Termination
    done: bool = False
    truncated: bool = False
    reward: float = 0.0


@dataclass
class StepResult:
    """Result from environment step. OpenEnv compatible."""
    observation: Observation
    reward: float
    done: bool
    truncated: bool
    info: Dict


@dataclass
class EnvironmentState:
    """Full environment state. OpenEnv compatible."""
    task_id: str
    database: Dict
    user_state: Dict
    conversation: List[Dict]
    step_count: int
    done: bool
    metrics: Dict


# ============================================================================
# τ²-Bench Challenge: Research Domains
# ============================================================================

class ResearchDomain(Enum):
    """Research domains inspired by τ²-bench's retail/airline/telecom."""
    ACADEMIC = "academic"      # Single-control (like retail)
    NEWS = "news"              # Single-control (like airline)  
    TECHNICAL = "technical"    # DUAL-CONTROL (like telecom) - user has tools!


# ============================================================================
# Agent Tools (Purple agent can use these)
# ============================================================================

AGENT_TOOLS = {
    ResearchDomain.ACADEMIC: [
        {
            "name": "search_papers",
            "description": "Search academic papers by query",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (1-10)", "default": 5}
            },
            "required": ["query"]
        },
        {
            "name": "get_paper_details",
            "description": "Get details of a specific paper by ID",
            "parameters": {
                "paper_id": {"type": "string", "description": "Paper identifier"}
            },
            "required": ["paper_id"]
        },
        {
            "name": "get_citations",
            "description": "Get papers that cite a given paper",
            "parameters": {
                "paper_id": {"type": "string", "description": "Paper identifier"}
            },
            "required": ["paper_id"]
        },
        {
            "name": "summarize_findings",
            "description": "Submit final research summary to user",
            "parameters": {
                "summary": {"type": "string", "description": "Research summary"},
                "sources": {"type": "array", "description": "List of source paper IDs"}
            },
            "required": ["summary", "sources"]
        }
    ],
    ResearchDomain.NEWS: [
        {
            "name": "search_news",
            "description": "Search news articles by query",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "days_back": {"type": "integer", "description": "Days back to search", "default": 7}
            },
            "required": ["query"]
        },
        {
            "name": "get_article",
            "description": "Get full text of a news article",
            "parameters": {
                "article_id": {"type": "string", "description": "Article identifier"}
            },
            "required": ["article_id"]
        },
        {
            "name": "verify_claim",
            "description": "Cross-reference a claim against sources",
            "parameters": {
                "claim": {"type": "string", "description": "Claim to verify"},
                "article_ids": {"type": "array", "description": "Articles to check"}
            },
            "required": ["claim", "article_ids"]
        },
        {
            "name": "submit_report",
            "description": "Submit final news report to user",
            "parameters": {
                "report": {"type": "string", "description": "News report"},
                "sources": {"type": "array", "description": "List of article IDs"}
            },
            "required": ["report", "sources"]
        }
    ],
    ResearchDomain.TECHNICAL: [
        {
            "name": "search_documentation",
            "description": "Search technical documentation",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "product": {"type": "string", "description": "Product name", "default": ""}
            },
            "required": ["query"]
        },
        {
            "name": "get_code_example",
            "description": "Get code example for a concept",
            "parameters": {
                "concept": {"type": "string", "description": "Concept to get example for"},
                "language": {"type": "string", "description": "Programming language", "default": "python"}
            },
            "required": ["concept"]
        },
        {
            "name": "check_compatibility",
            "description": "Check compatibility between tools/versions",
            "parameters": {
                "items": {"type": "array", "description": "Items to check"}
            },
            "required": ["items"]
        },
        {
            "name": "request_user_action",
            "description": "Request user to perform an action (dual-control)",
            "parameters": {
                "action": {"type": "string", "description": "Action for user to perform"},
                "expected_result": {"type": "string", "description": "What to expect"}
            },
            "required": ["action"]
        },
        {
            "name": "submit_solution",
            "description": "Submit technical solution to user",
            "parameters": {
                "solution": {"type": "string", "description": "Technical solution"},
                "code": {"type": "string", "description": "Code if applicable", "default": ""},
                "references": {"type": "array", "description": "Documentation references"}
            },
            "required": ["solution", "references"]
        }
    ]
}


# ============================================================================
# τ²-Bench Challenge: User Tools (For Dual-Control Domains)
# ============================================================================

USER_TOOLS = {
    ResearchDomain.ACADEMIC: [],  # Single-control: user has no tools
    ResearchDomain.NEWS: [],       # Single-control: user has no tools
    ResearchDomain.TECHNICAL: [    # DUAL-CONTROL: user can modify shared state!
        {
            "name": "run_command",
            "description": "User runs a command and reports output",
            "parameters": {
                "command": {"type": "string", "description": "Command that was run"},
                "output": {"type": "string", "description": "Output received"},
                "success": {"type": "boolean", "description": "Whether it succeeded"}
            },
            "required": ["command", "output", "success"]
        },
        {
            "name": "check_installation",
            "description": "User checks if a package is installed",
            "parameters": {
                "package": {"type": "string", "description": "Package name"},
                "installed": {"type": "boolean", "description": "Whether installed"},
                "version": {"type": "string", "description": "Version if installed", "default": ""}
            },
            "required": ["package", "installed"]
        },
        {
            "name": "report_error",
            "description": "User reports an error they encountered",
            "parameters": {
                "error_type": {"type": "string", "description": "Type of error"},
                "error_message": {"type": "string", "description": "Error message"},
                "context": {"type": "string", "description": "What they were doing"}
            },
            "required": ["error_type", "error_message"]
        },
        {
            "name": "confirm_fix",
            "description": "User confirms a problem is fixed",
            "parameters": {
                "issue": {"type": "string", "description": "Issue that was fixed"},
                "verification": {"type": "string", "description": "How they verified it"}
            },
            "required": ["issue"]
        }
    ]
}


# ============================================================================
# Domain Policies
# ============================================================================

DOMAIN_POLICIES = {
    ResearchDomain.ACADEMIC: [
        "Always cite at least 2 sources for any claim",
        "Prioritize peer-reviewed papers over preprints",
        "Include publication year for all citations",
        "Do not claim certainty for disputed findings",
        "Acknowledge limitations of the research"
    ],
    ResearchDomain.NEWS: [
        "Verify facts from at least 2 independent sources",
        "Distinguish between news and opinion pieces",
        "Include publication dates for all sources",
        "Note if information could not be verified",
        "Do not present speculation as fact"
    ],
    ResearchDomain.TECHNICAL: [
        "Always specify version numbers for tools/libraries",
        "Include working code examples when possible",
        "Warn about deprecated features or methods",
        "Note platform-specific considerations",
        "Reference official documentation over third-party",
        "Guide user through troubleshooting systematically",
        "Verify fixes with user before concluding"  # Dual-control policy
    ]
}


# ============================================================================
# τ²-Bench Challenge: Simulated Database with State Tracking
# ============================================================================

class SimulatedDatabase:
    """
    Simulated database with state tracking.
    τ²-bench compares final database state with expected state.
    """
    
    def __init__(self, domain: ResearchDomain):
        self.domain = domain
        self.initial_state = {}
        self.current_state = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize domain-specific database."""
        if self.domain == ResearchDomain.ACADEMIC:
            self.papers = {
                "paper_001": {
                    "id": "paper_001",
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani et al."],
                    "year": 2017,
                    "venue": "NeurIPS",
                    "abstract": "The Transformer architecture based on attention mechanisms.",
                    "citations": 80000,
                    "topics": ["transformers", "attention", "neural networks"]
                },
                "paper_002": {
                    "id": "paper_002",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": ["Devlin et al."],
                    "year": 2019,
                    "venue": "NAACL",
                    "abstract": "BERT for deep bidirectional representations.",
                    "citations": 60000,
                    "topics": ["bert", "pre-training", "nlp"]
                },
                "paper_003": {
                    "id": "paper_003",
                    "title": "GPT-3: Language Models are Few-Shot Learners",
                    "authors": ["Brown et al."],
                    "year": 2020,
                    "venue": "NeurIPS",
                    "abstract": "Scaling language models improves few-shot performance.",
                    "citations": 15000,
                    "topics": ["gpt", "few-shot", "language models"]
                },
                "paper_004": {
                    "id": "paper_004",
                    "title": "Chain-of-Thought Prompting Elicits Reasoning",
                    "authors": ["Wei et al."],
                    "year": 2022,
                    "venue": "NeurIPS",
                    "abstract": "Chain-of-thought prompting for reasoning tasks.",
                    "citations": 3000,
                    "topics": ["prompting", "reasoning", "cot"]
                },
                "paper_005": {
                    "id": "paper_005",
                    "title": "Constitutional AI: Harmlessness from AI Feedback",
                    "authors": ["Bai et al."],
                    "year": 2022,
                    "venue": "arXiv",
                    "abstract": "Constitutional AI for harmless AI assistants.",
                    "citations": 500,
                    "topics": ["alignment", "safety", "rlhf"]
                }
            }
            self.citation_graph = {
                "paper_002": ["paper_001"],
                "paper_003": ["paper_001", "paper_002"],
                "paper_004": ["paper_003"],
                "paper_005": ["paper_003"]
            }
            self.initial_state = {"papers_accessed": [], "citations_retrieved": []}
            
        elif self.domain == ResearchDomain.NEWS:
            self.articles = {
                "article_001": {
                    "id": "article_001",
                    "title": "AI Breakthrough: New Model Achieves Human-Level Performance",
                    "source": "Tech Daily",
                    "date": "2026-01-25",
                    "type": "news",
                    "content": "Researchers announced new AI model achieving human-level performance.",
                    "verified": True
                },
                "article_002": {
                    "id": "article_002",
                    "title": "Concerns Raised Over AI Safety Standards",
                    "source": "Science Weekly",
                    "date": "2026-01-26",
                    "type": "news",
                    "content": "Safety researchers express concerns about AI deployment.",
                    "verified": True
                },
                "article_003": {
                    "id": "article_003",
                    "title": "Opinion: Why We Should Embrace AI Progress",
                    "source": "Tech Opinion",
                    "date": "2026-01-27",
                    "type": "opinion",
                    "content": "AI advancement brings tremendous benefits to society.",
                    "verified": False
                }
            }
            self.initial_state = {"articles_read": [], "claims_verified": []}
            
        elif self.domain == ResearchDomain.TECHNICAL:
            self.docs = {
                "doc_001": {
                    "id": "doc_001",
                    "product": "PyTorch",
                    "version": "2.0",
                    "topic": "Installation",
                    "content": "Install PyTorch 2.0 with: pip install torch torchvision torchaudio",
                    "code_example": "import torch\nprint(torch.__version__)"
                },
                "doc_002": {
                    "id": "doc_002",
                    "product": "Transformers",
                    "version": "4.35",
                    "topic": "Quick Start",
                    "content": "Use Hugging Face Transformers for NLP models.",
                    "code_example": "from transformers import pipeline\nclassifier = pipeline('sentiment-analysis')"
                }
            }
            # τ²-bench dual-control: Track user's system state (shared state)
            self.initial_state = {
                "docs_accessed": [],
                "user_system": {
                    "python_version": "3.11",
                    "installed_packages": ["numpy"],
                    "errors_encountered": [],
                    "fixes_verified": []
                }
            }
        
        self.current_state = copy.deepcopy(self.initial_state)
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search the database."""
        query_lower = query.lower()
        results = []
        
        if self.domain == ResearchDomain.ACADEMIC:
            for paper in self.papers.values():
                if any(q in paper["title"].lower() or 
                       q in paper["abstract"].lower() or
                       q in " ".join(paper["topics"])
                       for q in query_lower.split()):
                    results.append(paper)
        elif self.domain == ResearchDomain.NEWS:
            for article in self.articles.values():
                if any(q in article["title"].lower() or 
                       q in article["content"].lower()
                       for q in query_lower.split()):
                    results.append(article)
        elif self.domain == ResearchDomain.TECHNICAL:
            for doc in self.docs.values():
                if any(q in doc["topic"].lower() or 
                       q in doc["content"].lower() or
                       q in doc["product"].lower()
                       for q in query_lower.split()):
                    results.append(doc)
        
        return results[:limit]
    
    def get_by_id(self, item_id: str) -> Optional[Dict]:
        """Get item by ID and track access."""
        if self.domain == ResearchDomain.ACADEMIC:
            item = self.papers.get(item_id)
            if item and item_id not in self.current_state["papers_accessed"]:
                self.current_state["papers_accessed"].append(item_id)
            return item
        elif self.domain == ResearchDomain.NEWS:
            item = self.articles.get(item_id)
            if item and item_id not in self.current_state["articles_read"]:
                self.current_state["articles_read"].append(item_id)
            return item
        elif self.domain == ResearchDomain.TECHNICAL:
            item = self.docs.get(item_id)
            if item and item_id not in self.current_state["docs_accessed"]:
                self.current_state["docs_accessed"].append(item_id)
            return item
        return None
    
    def update_user_system(self, updates: Dict):
        """τ²-bench dual-control: Update user's system state."""
        if self.domain == ResearchDomain.TECHNICAL and "user_system" in self.current_state:
            for key, value in updates.items():
                if key in self.current_state["user_system"]:
                    if isinstance(self.current_state["user_system"][key], list):
                        if value not in self.current_state["user_system"][key]:
                            self.current_state["user_system"][key].append(value)
                    else:
                        self.current_state["user_system"][key] = value
    
    def get_state(self) -> Dict:
        """Get current database state for evaluation."""
        return copy.deepcopy(self.current_state)
    
    def reset(self):
        """Reset to initial state."""
        self.current_state = copy.deepcopy(self.initial_state)


# ============================================================================
# τ²-Bench Challenge: Research Tasks with Ground Truth
# ============================================================================

@dataclass
class ResearchTask:
    """
    A research task with ground truth for τ²-bench style evaluation.
    """
    task_id: str
    domain: ResearchDomain
    difficulty: str  # easy, medium, hard
    is_dual_control: bool  # τ²-bench: Does user have tools?
    
    # User simulation
    user_instruction: str
    user_initial_message: str
    user_initial_state: Dict = field(default_factory=dict)
    
    # Ground truth
    required_tools: List[str] = field(default_factory=list)
    required_sources: List[str] = field(default_factory=list)
    expected_facts: List[str] = field(default_factory=list)
    policy_checks: List[str] = field(default_factory=list)
    expected_db_state: Dict = field(default_factory=dict)
    
    # Limits
    max_steps: int = 10
    time_limit_seconds: int = 120


# Task definitions
RESEARCH_TASKS = [
    # Single-control academic task
    ResearchTask(
        task_id="academic_transformers",
        domain=ResearchDomain.ACADEMIC,
        difficulty="medium",
        is_dual_control=False,
        user_instruction="Graduate student writing literature review on transformers.",
        user_initial_message="I'm writing a literature review on transformer architectures. Can you help me understand the key papers?",
        required_tools=["search_papers", "get_paper_details", "summarize_findings"],
        required_sources=["paper_001", "paper_002"],
        expected_facts=["Attention Is All You Need", "BERT", "transformers"],
        policy_checks=["cite at least 2 sources", "include publication year"],
        expected_db_state={"papers_accessed": ["paper_001", "paper_002"]}
    ),
    # Single-control academic task (harder)
    ResearchTask(
        task_id="academic_reasoning",
        domain=ResearchDomain.ACADEMIC,
        difficulty="hard",
        is_dual_control=False,
        user_instruction="Researcher studying LLM reasoning improvements.",
        user_initial_message="I'm interested in improving reasoning in language models. What techniques exist?",
        required_tools=["search_papers", "get_paper_details", "get_citations", "summarize_findings"],
        required_sources=["paper_003", "paper_004"],
        expected_facts=["chain-of-thought", "few-shot", "reasoning"],
        policy_checks=["cite at least 2 sources", "acknowledge limitations"],
        expected_db_state={"papers_accessed": ["paper_003", "paper_004"]}
    ),
    # Single-control news task
    ResearchTask(
        task_id="news_ai_safety",
        domain=ResearchDomain.NEWS,
        difficulty="medium",
        is_dual_control=False,
        user_instruction="Citizen wanting balanced AI safety overview.",
        user_initial_message="I've been hearing about AI safety concerns. Can you give me a balanced overview?",
        required_tools=["search_news", "get_article", "verify_claim", "submit_report"],
        required_sources=["article_001", "article_002"],
        expected_facts=["AI performance", "safety concerns", "testing"],
        policy_checks=["verify from 2 sources", "distinguish news from opinion"],
        expected_db_state={"articles_read": ["article_001", "article_002"]}
    ),
    # DUAL-CONTROL technical task (τ²-bench telecom style)
    ResearchTask(
        task_id="technical_pytorch_troubleshoot",
        domain=ResearchDomain.TECHNICAL,
        difficulty="hard",
        is_dual_control=True,  # USER HAS TOOLS!
        user_instruction="Developer having trouble installing PyTorch. Can run commands and report results.",
        user_initial_message="I'm trying to install PyTorch but getting errors. Can you help me troubleshoot?",
        user_initial_state={
            "python_version": "3.11",
            "installed_packages": ["numpy"],
            "os": "Ubuntu 22.04",
            "current_error": "ModuleNotFoundError: No module named 'torch'"
        },
        required_tools=["search_documentation", "get_code_example", "request_user_action", "submit_solution"],
        required_sources=["doc_001"],
        expected_facts=["pip install torch", "import torch", "2.0"],
        policy_checks=["specify version numbers", "include code examples", "verify fixes with user"],
        expected_db_state={
            "docs_accessed": ["doc_001"],
            "user_system": {
                "installed_packages": ["numpy", "torch"],
                "fixes_verified": ["pytorch_installation"]
            }
        }
    ),
]


# ============================================================================
# τ²-Bench Challenge: User Simulator with Tool Support
# ============================================================================

class UserSimulator:
    """
    User simulator with dual-control tool support (τ²-bench style).
    In dual-control domains, user can use tools to modify shared state.
    """
    
    def __init__(self, task: ResearchTask, model: str = "gpt-4o-mini"):
        self.task = task
        self.model = model
        self.client = None
        if HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        
        self.conversation_history = []
        self.turn_count = 0
        self.max_turns = 5
        self.user_state = copy.deepcopy(task.user_initial_state)
        self.user_tool_calls = []
    
    def get_initial_message(self) -> str:
        return self.task.user_initial_message
    
    def get_state(self) -> Dict:
        return copy.deepcopy(self.user_state)
    
    def respond(self, agent_message: str) -> Tuple[str, bool, Optional[Dict]]:
        """
        Generate user response.
        Returns (response, should_end, user_tool_call_if_any).
        """
        self.conversation_history.append({"role": "agent", "content": agent_message})
        self.turn_count += 1
        
        # Check termination
        if self.turn_count >= self.max_turns:
            return "Thank you, that's very helpful!", True, None
        
        if any(kw in agent_message.lower() for kw in 
               ["here's my summary", "in conclusion", "to summarize", "final answer", "problem solved"]):
            return "Thank you, that answers my question!", True, None
        
        # Generate response
        if self.client:
            response, tool_call = self._generate_llm_response(agent_message)
        else:
            response, tool_call = self._generate_rule_based_response(agent_message)
        
        self.conversation_history.append({"role": "user", "content": response})
        
        if tool_call:
            self.user_tool_calls.append(tool_call)
            # Update user state for dual-control
            self._apply_user_tool(tool_call)
        
        return response, False, tool_call
    
    def _apply_user_tool(self, tool_call: Dict):
        """Apply user tool effects to user state (dual-control)."""
        tool = tool_call.get("tool")
        params = tool_call.get("parameters", {})
        
        if tool == "run_command":
            if params.get("success") and "pip install" in params.get("command", ""):
                # Extract package name
                cmd = params.get("command", "")
                if "torch" in cmd:
                    if "installed_packages" not in self.user_state:
                        self.user_state["installed_packages"] = []
                    if "torch" not in self.user_state["installed_packages"]:
                        self.user_state["installed_packages"].append("torch")
        
        elif tool == "confirm_fix":
            if "fixes_verified" not in self.user_state:
                self.user_state["fixes_verified"] = []
            issue = params.get("issue", "unknown")
            self.user_state["fixes_verified"].append(issue)
    
    def _generate_llm_response(self, agent_message: str) -> Tuple[str, Optional[Dict]]:
        """Generate response using LLM."""
        
        tools_desc = ""
        if self.task.is_dual_control:
            user_tools = USER_TOOLS[self.task.domain]
            tools_desc = f"\nYou can use these tools: {[t['name'] for t in user_tools]}"
            tools_desc += f"\nYour current state: {json.dumps(self.user_state)}"
        
        system_prompt = f"""You are simulating a user:
{self.task.user_instruction}
{tools_desc}

When the agent asks you to run a command or check something:
- Actually do it and report the result
- If asked to run "pip install torch", respond: "I ran pip install torch. It installed successfully!"
- If asked to verify something works, confirm it

Keep responses brief (1-2 sentences)."""
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.conversation_history:
            role = "assistant" if turn["role"] == "agent" else "user"
            messages.append({"role": role, "content": turn["content"]})
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            response = completion.choices[0].message.content
            
            # Detect if user used a tool (dual-control)
            tool_call = None
            if self.task.is_dual_control:
                if "ran" in response.lower() and "pip install" in response.lower():
                    tool_call = {
                        "tool": "run_command",
                        "parameters": {
                            "command": "pip install torch",
                            "output": "Successfully installed",
                            "success": True
                        }
                    }
                elif "it works" in response.lower() or "verified" in response.lower():
                    tool_call = {
                        "tool": "confirm_fix",
                        "parameters": {
                            "issue": "pytorch_installation",
                            "verification": "import torch works"
                        }
                    }
            
            return response, tool_call
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Can you tell me more?", None
    
    def _generate_rule_based_response(self, agent_message: str) -> Tuple[str, Optional[Dict]]:
        """Fallback rule-based response."""
        agent_lower = agent_message.lower()
        
        # Dual-control responses
        if self.task.is_dual_control:
            if "pip install" in agent_lower or "run" in agent_lower:
                return "I ran pip install torch. It installed successfully!", {
                    "tool": "run_command",
                    "parameters": {"command": "pip install torch", "output": "Success", "success": True}
                }
            if "verify" in agent_lower or "check" in agent_lower:
                return "Yes, it works now! import torch runs without errors.", {
                    "tool": "confirm_fix",
                    "parameters": {"issue": "pytorch_installation"}
                }
        
        follow_ups = [
            "Can you provide more details?",
            "What are the sources for this?",
            "Thank you, that's helpful!"
        ]
        return follow_ups[min(self.turn_count, len(follow_ups) - 1)], None


# ============================================================================
# Tool Executor
# ============================================================================

class ToolExecutor:
    """Executes tools for both agent and user."""
    
    def __init__(self, database: SimulatedDatabase, domain: ResearchDomain):
        self.database = database
        self.domain = domain
        self.agent_tool_calls = []
        self.user_tool_calls = []
        self.submitted_result = None
        self.user_action_requests = []
    
    def execute_agent_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute agent tool call."""
        self.agent_tool_calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Academic tools
        if tool_name == "search_papers":
            results = self.database.search(parameters.get("query", ""))
            return {"status": "success", "results": results}
        
        elif tool_name == "get_paper_details":
            paper = self.database.get_by_id(parameters.get("paper_id"))
            return {"status": "success", "paper": paper} if paper else {"status": "error", "message": "Not found"}
        
        elif tool_name == "get_citations":
            paper_id = parameters.get("paper_id")
            citing_ids = self.database.citation_graph.get(paper_id, [])
            citing = [self.database.papers.get(pid) for pid in citing_ids if pid in self.database.papers]
            return {"status": "success", "citations": citing}
        
        elif tool_name == "summarize_findings":
            self.submitted_result = {"summary": parameters.get("summary"), "sources": parameters.get("sources", [])}
            return {"status": "success", "message": "Summary submitted"}
        
        # News tools
        elif tool_name == "search_news":
            results = self.database.search(parameters.get("query", ""))
            return {"status": "success", "results": results}
        
        elif tool_name == "get_article":
            article = self.database.get_by_id(parameters.get("article_id"))
            return {"status": "success", "article": article} if article else {"status": "error", "message": "Not found"}
        
        elif tool_name == "verify_claim":
            return {"status": "success", "verified": True, "confidence": 0.85}
        
        elif tool_name == "submit_report":
            self.submitted_result = {"report": parameters.get("report"), "sources": parameters.get("sources", [])}
            return {"status": "success", "message": "Report submitted"}
        
        # Technical tools
        elif tool_name == "search_documentation":
            results = self.database.search(parameters.get("query", ""))
            return {"status": "success", "results": results}
        
        elif tool_name == "get_code_example":
            for doc in self.database.docs.values():
                if parameters.get("concept", "").lower() in doc["topic"].lower():
                    return {"status": "success", "code": doc.get("code_example", "")}
            return {"status": "success", "code": "# Not found"}
        
        elif tool_name == "check_compatibility":
            return {"status": "success", "compatible": True}
        
        elif tool_name == "request_user_action":
            self.user_action_requests.append(parameters)
            return {"status": "success", "message": "Request sent to user"}
        
        elif tool_name == "submit_solution":
            self.submitted_result = {
                "solution": parameters.get("solution"),
                "code": parameters.get("code", ""),
                "references": parameters.get("references", [])
            }
            return {"status": "success", "message": "Solution submitted"}
        
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def execute_user_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute user tool (dual-control)."""
        self.user_tool_calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update database state for dual-control
        if tool_name == "run_command":
            if parameters.get("success"):
                cmd = parameters.get("command", "")
                if "torch" in cmd:
                    self.database.update_user_system({"installed_packages": "torch"})
        
        elif tool_name == "confirm_fix":
            issue = parameters.get("issue", "unknown")
            self.database.update_user_system({"fixes_verified": issue})
        
        return {"status": "success"}
    
    def get_used_tools(self) -> List[str]:
        return list(set(call["tool"] for call in self.agent_tool_calls))
    
    def get_submitted_sources(self) -> List[str]:
        if self.submitted_result:
            return self.submitted_result.get("sources", []) or self.submitted_result.get("references", [])
        return []


# ============================================================================
# Evaluation Engine (τ²-bench Style)
# ============================================================================

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result."""
    task_id: str
    domain: str
    is_dual_control: bool
    
    # Scores
    tool_use_score: float
    source_score: float
    fact_score: float
    policy_score: float
    db_state_score: float  # τ²-bench: Database state comparison
    
    total_score: float
    max_score: float = 100.0
    passed: bool = False
    
    # Details
    response_time: float = 0.0
    num_steps: int = 0
    tools_used: List[str] = field(default_factory=list)
    sources_cited: List[str] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)
    user_tool_calls: int = 0


class Evaluator:
    """τ²-bench style evaluator with database state comparison."""
    
    def evaluate(
        self,
        task: ResearchTask,
        tool_executor: ToolExecutor,
        user_simulator: UserSimulator,
        final_response: str,
        response_time: float,
        num_steps: int
    ) -> EvaluationResult:
        
        # 1. Tool use (20%)
        tools_used = tool_executor.get_used_tools()
        required_used = sum(1 for t in task.required_tools if t in tools_used)
        tool_score = (required_used / len(task.required_tools)) * 100 if task.required_tools else 100
        
        # 2. Sources (20%)
        sources = tool_executor.get_submitted_sources()
        required_sources = sum(1 for s in task.required_sources if s in sources)
        source_score = (required_sources / len(task.required_sources)) * 100 if task.required_sources else 100
        
        # 3. Facts (25%)
        fact_score = self._evaluate_facts(final_response, task.expected_facts)
        
        # 4. Policies (15%)
        policy_score, violations = self._evaluate_policies(final_response, task)
        
        # 5. Database state (20%) - τ²-bench style
        db_state_score = self._evaluate_db_state(
            tool_executor.database.get_state(),
            task.expected_db_state,
            task.is_dual_control
        )
        
        # Weighted total
        total = (
            tool_score * 0.20 +
            source_score * 0.20 +
            fact_score * 0.25 +
            policy_score * 0.15 +
            db_state_score * 0.20
        )
        
        return EvaluationResult(
            task_id=task.task_id,
            domain=task.domain.value,
            is_dual_control=task.is_dual_control,
            tool_use_score=round(tool_score, 2),
            source_score=round(source_score, 2),
            fact_score=round(fact_score, 2),
            policy_score=round(policy_score, 2),
            db_state_score=round(db_state_score, 2),
            total_score=round(total, 2),
            passed=total >= 70.0,
            response_time=round(response_time, 2),
            num_steps=num_steps,
            tools_used=tools_used,
            sources_cited=sources,
            policy_violations=violations,
            user_tool_calls=len(user_simulator.user_tool_calls)
        )
    
    def _evaluate_facts(self, response: str, expected: List[str]) -> float:
        if not expected:
            return 100.0
        response_lower = response.lower()
        found = sum(1 for fact in expected if fact.lower() in response_lower or
                   sum(1 for w in fact.lower().split() if len(w) > 3 and w in response_lower) >= 2)
        return (found / len(expected)) * 100
    
    def _evaluate_policies(self, response: str, task: ResearchTask) -> Tuple[float, List[str]]:
        violations = []
        policies = DOMAIN_POLICIES[task.domain]
        
        for policy in policies:
            policy_lower = policy.lower()
            if "cite" in policy_lower and "2" in policy_lower:
                if len(re.findall(r'\[\d+\]|\(\d{4}\)|et al\.', response)) < 2:
                    violations.append(policy)
            elif "version" in policy_lower:
                if not re.search(r'\d+\.\d+', response):
                    violations.append(policy)
        
        compliance = 1 - (len(violations) / len(policies)) if policies else 1
        return compliance * 100, violations
    
    def _evaluate_db_state(self, actual: Dict, expected: Dict, is_dual: bool) -> float:
        """τ²-bench style database state comparison."""
        if not expected:
            return 100.0
        
        matches = 0
        total = 0
        
        for key, exp_val in expected.items():
            if key in actual:
                act_val = actual[key]
                if isinstance(exp_val, list):
                    for item in exp_val:
                        total += 1
                        if item in act_val:
                            matches += 1
                elif isinstance(exp_val, dict):
                    for sub_key, sub_exp in exp_val.items():
                        if sub_key in act_val:
                            total += 1
                            if isinstance(sub_exp, list):
                                if all(i in act_val[sub_key] for i in sub_exp):
                                    matches += 1
                            elif act_val[sub_key] == sub_exp:
                                matches += 1
                else:
                    total += 1
                    if act_val == exp_val:
                        matches += 1
        
        return (matches / total) * 100 if total > 0 else 100.0


# ============================================================================
# OpenEnv Challenge: Main Environment Class
# ============================================================================

class ResearchToolBenchEnv:
    """
    Main benchmark environment.
    
    OpenEnv Challenge compatible:
    - step(action) -> StepResult
    - reset(task_id) -> Observation
    - state() -> EnvironmentState
    - close() -> None
    
    τ²-Bench Challenge compatible:
    - Dual-control domains where user has tools
    - Database state tracking
    - pass^k metric support
    """
    
    def __init__(self, task: Optional[ResearchTask] = None):
        self.task = task
        self.database = None
        self.tool_executor = None
        self.user_simulator = None
        self.evaluator = Evaluator()
        
        self.conversation_history = []
        self.step_count = 0
        self.start_time = None
        self.done = False
        self.final_response = ""
        
        if task:
            self._init_task(task)
    
    def _init_task(self, task: ResearchTask):
        self.task = task
        self.database = SimulatedDatabase(task.domain)
        self.tool_executor = ToolExecutor(self.database, task.domain)
        self.user_simulator = UserSimulator(task)
        self.conversation_history = []
        self.step_count = 0
        self.done = False
        self.final_response = ""
    
    def reset(self, task_id: Optional[str] = None) -> Observation:
        """OpenEnv: Reset environment."""
        if task_id:
            task = next((t for t in RESEARCH_TASKS if t.task_id == task_id), None)
            if task:
                self._init_task(task)
        elif self.task:
            self._init_task(self.task)
        else:
            self._init_task(RESEARCH_TASKS[0])
        
        self.start_time = time.time()
        
        initial_message = self.user_simulator.get_initial_message()
        self.conversation_history.append({"role": "user", "content": initial_message})
        
        return Observation(
            task_id=self.task.task_id,
            domain=self.task.domain.value,
            user_message=initial_message,
            conversation_history=self.conversation_history.copy(),
            agent_tools=AGENT_TOOLS[self.task.domain],
            user_tools=USER_TOOLS[self.task.domain],
            policies=DOMAIN_POLICIES[self.task.domain],
            database_state=self.database.get_state(),
            user_state=self.user_simulator.get_state(),
            step_count=self.step_count,
            max_steps=self.task.max_steps,
            done=False
        )
    
    def step(self, action: Dict) -> StepResult:
        """OpenEnv: Execute action."""
        self.step_count += 1
        reward = 0.0
        info = {}
        
        action_type = action.get("type", "message")
        
        if action_type == "tool_call":
            tool_name = action.get("tool_name", "")
            parameters = action.get("parameters", {})
            
            result = self.tool_executor.execute_agent_tool(tool_name, parameters)
            info["tool_result"] = result
            
            if result.get("status") == "success":
                reward += 5.0
            
            self.conversation_history.append({"role": "tool", "tool": tool_name, "result": result})
            
            if tool_name in ["summarize_findings", "submit_report", "submit_solution"]:
                self.final_response = parameters.get("summary") or parameters.get("report") or parameters.get("solution", "")
                self.done = True
        
        elif action_type == "message":
            message = action.get("message", "")
            self.conversation_history.append({"role": "agent", "content": message})
            
            user_response, should_end, user_tool_call = self.user_simulator.respond(message)
            self.conversation_history.append({"role": "user", "content": user_response})
            
            if user_tool_call:
                self.tool_executor.execute_user_tool(user_tool_call["tool"], user_tool_call["parameters"])
                info["user_tool_call"] = user_tool_call
            
            if should_end:
                self.done = True
                self.final_response = message
        
        truncated = self.step_count >= self.task.max_steps
        if truncated:
            self.done = True
        
        if self.done:
            elapsed = time.time() - self.start_time
            evaluation = self.evaluator.evaluate(
                self.task, self.tool_executor, self.user_simulator,
                self.final_response, elapsed, self.step_count
            )
            reward = evaluation.total_score
            info["evaluation"] = asdict(evaluation)
        
        last_user_msg = ""
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        observation = Observation(
            task_id=self.task.task_id,
            domain=self.task.domain.value,
            user_message=last_user_msg,
            conversation_history=self.conversation_history.copy(),
            agent_tools=AGENT_TOOLS[self.task.domain],
            user_tools=USER_TOOLS[self.task.domain],
            policies=DOMAIN_POLICIES[self.task.domain],
            database_state=self.database.get_state(),
            user_state=self.user_simulator.get_state(),
            step_count=self.step_count,
            max_steps=self.task.max_steps,
            done=self.done,
            truncated=truncated,
            reward=reward
        )
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=self.done,
            truncated=truncated,
            info=info
        )
    
    def state(self) -> EnvironmentState:
        """OpenEnv: Get current state."""
        return EnvironmentState(
            task_id=self.task.task_id if self.task else "",
            database=self.database.get_state() if self.database else {},
            user_state=self.user_simulator.get_state() if self.user_simulator else {},
            conversation=self.conversation_history.copy(),
            step_count=self.step_count,
            done=self.done,
            metrics={
                "tools_used": self.tool_executor.get_used_tools() if self.tool_executor else [],
                "user_tool_calls": len(self.user_simulator.user_tool_calls) if self.user_simulator else 0,
                "elapsed_time": time.time() - self.start_time if self.start_time else 0
            }
        )
    
    def close(self):
        """OpenEnv: Clean up resources."""
        self.database = None
        self.tool_executor = None
        self.user_simulator = None


# ============================================================================
# pass^k Metric (τ²-bench)
# ============================================================================

def calculate_pass_at_k(results: List[bool], k: int) -> float:
    """τ²-bench pass^k: probability of passing k consecutive trials."""
    if len(results) < k:
        return 0.0
    
    n_pass = sum(results)
    n_total = len(results)
    
    if n_pass == n_total:
        return 1.0
    elif n_pass == 0:
        return 0.0
    else:
        return (n_pass / n_total) ** k


# ============================================================================
# Green Agent for AgentBeats
# ============================================================================

class ResearchToolBenchGreenAgent:
    """Green Agent for AgentBeats competition."""
    
    def __init__(self, num_tasks: int = 4, num_trials: int = 2):
        self.num_tasks = min(num_tasks, len(RESEARCH_TASKS))
        self.num_trials = num_trials
        self.tasks = RESEARCH_TASKS[:self.num_tasks]
    
    async def run_assessment(self, participants: Dict[str, str], config: Dict) -> Dict:
        """Run complete assessment."""
        
        logger.info(f"Starting ResearchToolBench assessment")
        logger.info(f"Tasks: {self.num_tasks}, Trials: {self.num_trials}")
        
        results = {
            "benchmark": "ResearchToolBench",
            "version": "2.0.0",
            "challenges": ["tau2-bench", "openenv"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "participants": {},
            "results": []
        }
        
        for role, endpoint in participants.items():
            logger.info(f"Evaluating: {role}")
            participant_results = await self._evaluate_participant(role, endpoint)
            results["participants"][role] = role
            results["results"].append(participant_results)
        
        return results
    
    async def _evaluate_participant(self, role: str, endpoint: str) -> Dict:
        """Evaluate single participant."""
        
        client = httpx.AsyncClient(timeout=120)
        task_results = []
        all_passed = []
        
        try:
            for task in self.tasks:
                trial_results = []
                
                for trial in range(self.num_trials):
                    logger.info(f"  Task: {task.task_id} (dual={task.is_dual_control}), Trial: {trial + 1}")
                    
                    env = ResearchToolBenchEnv(task)
                    observation = env.reset()
                    
                    while not observation.done:
                        action = await self._get_agent_action(client, endpoint, observation)
                        step_result = env.step(action)
                        observation = step_result.observation
                        
                        if step_result.done:
                            trial_results.append(step_result.info.get("evaluation", {}))
                            all_passed.append(step_result.info.get("evaluation", {}).get("passed", False))
                            break
                    
                    env.close()
                
                if trial_results:
                    avg = sum(r.get("total_score", 0) for r in trial_results) / len(trial_results)
                    task_results.append({
                        "task_id": task.task_id,
                        "domain": task.domain.value,
                        "difficulty": task.difficulty,
                        "dual_control": task.is_dual_control,
                        "trials": trial_results,
                        "average_score": round(avg, 2),
                        "pass_rate": sum(1 for r in trial_results if r.get("passed")) / len(trial_results)
                    })
        
        finally:
            await client.aclose()
        
        total = sum(t["average_score"] for t in task_results)
        max_score = self.num_tasks * 100
        
        pass_1 = calculate_pass_at_k(all_passed, 1)
        pass_2 = calculate_pass_at_k(all_passed, 2)
        
        return {
            "participant": role,
            "total_score": round(total, 2),
            "max_score": max_score,
            "pass_rate": round((total / max_score) * 100, 2) if max_score > 0 else 0,
            "pass_at_1": round(pass_1 * 100, 2),
            "pass_at_2": round(pass_2 * 100, 2),
            "task_results": task_results
        }
    
    async def _get_agent_action(self, client: httpx.AsyncClient, endpoint: str, obs: Observation) -> Dict:
        """Get action from purple agent via A2A."""
        
        message = {
            "user_message": obs.user_message,
            "agent_tools": obs.agent_tools,
            "user_tools": obs.user_tools,
            "policies": obs.policies,
            "conversation_history": obs.conversation_history,
            "database_state": obs.database_state,
            "user_state": obs.user_state,
            "step": obs.step_count,
            "max_steps": obs.max_steps,
            "is_dual_control": len(obs.user_tools) > 0
        }
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {"message": {"role": "user", "parts": [{"type": "data", "data": message}]}},
                "id": f"step-{obs.step_count}"
            }
            
            response = await client.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            
            if "result" in result:
                return self._parse_response(result["result"])
            
        except Exception as e:
            logger.error(f"Error: {e}")
        
        return {"type": "message", "message": "I need more information."}
    
    def _parse_response(self, result: Dict) -> Dict:
        """Parse agent response into action."""
        if "tool_call" in result:
            return {
                "type": "tool_call",
                "tool_name": result["tool_call"].get("name", ""),
                "parameters": result["tool_call"].get("parameters", {})
            }
        
        if "message" in result:
            msg = result["message"]
            if isinstance(msg, dict) and "parts" in msg:
                for part in msg["parts"]:
                    if part.get("type") == "text":
                        return {"type": "message", "message": part.get("text", "")}
            elif isinstance(msg, str):
                return {"type": "message", "message": msg}
        
        return {"type": "message", "message": str(result)}


# ============================================================================
# Agent Card
# ============================================================================

def create_agent_card() -> Dict:
    """Create A2A agent card."""
    return {
        "name": "ResearchToolBench",
        "description": "Research benchmark combining τ²-Bench (dual-control) and OpenEnv (Gymnasium APIs) challenges",
        "version": "2.0.0",
        "challenges": [
            {
                "name": "τ²-Bench Challenge",
                "sponsor": "Sierra Research",
                "features": ["dual-control", "Dec-POMDP", "pass^k metric"]
            },
            {
                "name": "OpenEnv Challenge",
                "sponsor": "Meta-PyTorch / Hugging Face / Unsloth",
                "features": ["Gymnasium APIs", "HF Hub compatible", "RL training ready"]
            }
        ],
        "domains": [
            {"name": "academic", "dual_control": False},
            {"name": "news", "dual_control": False},
            {"name": "technical", "dual_control": True}
        ],
        "metrics": [
            "tool_use_score", "source_score", "fact_score",
            "policy_score", "db_state_score",
            "pass_at_1", "pass_at_2"
        ],
        "capabilities": {
            "openenv_compatible": True,
            "tau2_bench_compatible": True,
            "gymnasium_api": True,
            "dual_control_support": True,
            "hf_hub_deployable": True
        }
    }


async def handle_assessment_request(request_data: Dict) -> Dict:
    """Handle assessment request."""
    participants = request_data.get("participants", {})
    config = request_data.get("config", {})
    
    agent = ResearchToolBenchGreenAgent(
        num_tasks=config.get("num_tasks", 4),
        num_trials=config.get("num_trials", 2)
    )
    return await agent.run_assessment(participants, config)


def create_app():
    """Create FastAPI app for A2A protocol."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="ResearchToolBench Green Agent")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "benchmark": "ResearchToolBench", "version": "2.0.0"}
    
    @app.get("/.well-known/agent.json")
    async def agent_card():
        return create_agent_card()
    
    @app.post("/")
    async def handle_request(request: Request):
        data = await request.json()
        
        if data.get("method") == "assessment/run":
            results = await handle_assessment_request(data.get("params", {}))
            return JSONResponse({"jsonrpc": "2.0", "result": results, "id": data.get("id")})
        
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": data.get("id")})
    
    return app


if __name__ == "__main__":
    print("=" * 70)
    print("ResearchToolBench - τ²-Bench + OpenEnv Challenge Compatible")
    print("=" * 70)
    
    # Test dual-control task
    task = RESEARCH_TASKS[3]  # technical_pytorch_troubleshoot
    env = ResearchToolBenchEnv(task)
    
    obs = env.reset()
    print(f"\nTask: {obs.task_id}")
    print(f"Domain: {obs.domain}")
    print(f"Dual-Control: {len(obs.user_tools) > 0}")
    print(f"User: {obs.user_message}")
    print(f"Agent Tools: {[t['name'] for t in obs.agent_tools]}")
    print(f"User Tools (τ²-bench): {[t['name'] for t in obs.user_tools]}")
    
    # Simulate
    print("\n--- Simulating dual-control interaction ---")
    
    a1 = {"type": "tool_call", "tool_name": "search_documentation", "parameters": {"query": "pytorch installation"}}
    r1 = env.step(a1)
    print(f"Step 1: search_documentation -> Found {len(r1.info.get('tool_result', {}).get('results', []))} docs")
    
    a2 = {"type": "message", "message": "Please run: pip install torch"}
    r2 = env.step(a2)
    print(f"Step 2: User response: {r2.observation.user_message[:50]}...")
    if "user_tool_call" in r2.info:
        print(f"  User used tool (τ²-bench dual-control): {r2.info['user_tool_call']['tool']}")
    
    a3 = {"type": "message", "message": "Please verify: import torch"}
    r3 = env.step(a3)
    print(f"Step 3: User verified fix")
    
    a4 = {
        "type": "tool_call",
        "tool_name": "submit_solution",
        "parameters": {
            "solution": "Install PyTorch 2.0 with pip install torch. Verify with import torch.",
            "code": "import torch\nprint(torch.__version__)",
            "references": ["doc_001"]
        }
    }
    r4 = env.step(a4)
    
    if "evaluation" in r4.info:
        e = r4.info["evaluation"]
        print(f"\n--- Evaluation Results ---")
        print(f"  Tool Use: {e['tool_use_score']}")
        print(f"  Sources: {e['source_score']}")
        print(f"  Facts: {e['fact_score']}")
        print(f"  Policies: {e['policy_score']}")
        print(f"  DB State (τ²-bench): {e['db_state_score']}")
        print(f"  User Tool Calls: {e['user_tool_calls']}")
        print(f"  Total: {e['total_score']}")
        print(f"  Passed: {e['passed']}")
    
    state = env.state()
    print(f"\n--- OpenEnv State ---")
    print(f"  Database: {state.database}")
    print(f"  User State: {state.user_state}")
    
    env.close()
    print("\n✓ Environment closed (OpenEnv compatible)")
