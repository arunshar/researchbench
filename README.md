# ResearchToolBench

A research agent benchmark combining concepts from the **AgentBeats Custom Tracks**:

## 🏆 Challenge Compatibility

### τ²-Bench Challenge (Sierra Research)
- **Dual-control environments**: Both agent AND user have tools (Dec-POMDP)
- **Database state tracking**: Final state compared with expected state
- **pass^k reliability metric**: Measures consistency across trials
- **Three domains**: academic, news, technical (dual-control)

### OpenEnv Challenge (Meta-PyTorch / Hugging Face / Unsloth)
- **Gymnasium-style APIs**: `step()`, `reset()`, `state()`, `close()`
- **HuggingFace Hub compatible**: Ready for deployment
- **RL training ready**: TRL/TorchForge integration

## 📊 Evaluation Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| Tool Use | 20% | Required tools called correctly |
| Source Citation | 20% | Required sources cited |
| Fact Accuracy | 25% | Expected facts present |
| Policy Compliance | 15% | Domain policies followed |
| DB State (τ²-bench) | 20% | Database state matches expected |

**Additional Metrics:**
- `pass@1`: Single-trial success rate
- `pass@2`: Two-trial consistency (τ²-bench style)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ResearchToolBench                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  Academic   │    │    News     │    │  Technical  │ │
│  │  (single)   │    │  (single)   │    │(DUAL-CTRL)  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│  OpenEnv APIs: step() | reset() | state() | close()    │
├─────────────────────────────────────────────────────────┤
│  τ²-bench: User Tools | DB State | pass^k Metric       │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### OpenEnv-Style Usage

```python
from src.green_agent import ResearchToolBenchEnv, RESEARCH_TASKS

# Create environment
env = ResearchToolBenchEnv()

# Reset with task
obs = env.reset(task_id="technical_pytorch_troubleshoot")

# Run episode
while not obs.done:
    action = your_agent.act(obs)  # Your purple agent
    result = env.step(action)
    obs = result.observation
    
    if result.done:
        print(f"Score: {result.info['evaluation']['total_score']}")

# Get final state (τ²-bench style)
state = env.state()
print(f"DB State: {state.database}")

env.close()
```

### Dual-Control Example (τ²-bench style)

```python
# In technical domain, USER has tools too!
obs = env.reset(task_id="technical_pytorch_troubleshoot")

print(f"Agent tools: {[t['name'] for t in obs.agent_tools]}")
print(f"User tools: {[t['name'] for t in obs.user_tools]}")  # τ²-bench!

# Agent requests user action
action = {
    "type": "tool_call",
    "tool_name": "request_user_action",
    "parameters": {"action": "pip install torch"}
}
result = env.step(action)

# User responds with tool usage
if "user_tool_call" in result.info:
    print(f"User used: {result.info['user_tool_call']['tool']}")
```

## 📁 Project Structure

```
researchbench/
├── src/
│   ├── green_agent.py    # Benchmark environment + evaluator
│   └── purple_agent.py   # Baseline agent
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── SUBMISSION_ANSWERS.md
```

## 🐳 Docker Deployment

```bash
# Build (IMPORTANT: use linux/amd64 for AgentBeats)
docker build --platform linux/amd64 -t researchbench-green .

# Run green agent
docker run -p 8000:8000 researchbench-green

# Run purple agent
docker run -p 8001:8001 researchbench-purple
```

## 📚 References

- **τ²-bench**: Barres et al., 2025 ([arXiv:2506.07982](https://arxiv.org/abs/2506.07982))
- **τ-bench**: Yao et al., 2024 ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045))
- **OpenEnv**: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **AgentBeats**: [rdi.berkeley.edu/agentx-agentbeats](https://rdi.berkeley.edu/agentx-agentbeats)

## 📄 License

MIT License
