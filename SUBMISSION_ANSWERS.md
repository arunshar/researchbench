# ResearchToolBench - Submission Form Answers

## Basic Information

**Benchmark Name:** ResearchToolBench

**GitHub Repository:** [TO BE FILLED - your GitHub repo URL]

**AgentBeats Green Agent Link:** [TO BE FILLED - agentbeats.dev link]

**Demo Video:** [TO BE FILLED - YouTube/Loom link]

---

## Submission Questions

### 1. Abstract (Brief description of tasks your green agent evaluates)

ResearchToolBench evaluates research agents across three domains (academic, news, technical) using concepts from both the **τ²-Bench Challenge** and **OpenEnv Challenge**. The benchmark features:

- **Dual-control environments** (τ²-bench): In the technical domain, BOTH agent AND user have tools, requiring coordination
- **Gymnasium-style APIs** (OpenEnv): `step()`, `reset()`, `state()`, `close()` for RL compatibility
- **Multi-dimensional evaluation**: Tool use, source citation, fact accuracy, policy compliance, and database state comparison
- **pass^k reliability metric** (τ²-bench): Measures agent consistency across multiple trials

---

### 2. How does your benchmark expand upon existing coverage?

**Comparison with τ²-bench:**
| Aspect | τ²-bench | ResearchToolBench |
|--------|----------|-------------------|
| Domains | Retail, Airline, Telecom | Academic, News, Technical |
| Dual-control | Telecom only | Technical domain |
| Focus | Customer service | Research tasks |
| Metrics | pass^k, binary | Multi-dimensional + pass^k |

**Comparison with OpenEnv:**
| Aspect | OpenEnv | ResearchToolBench |
|--------|---------|-------------------|
| Purpose | RL training environments | Evaluation benchmark |
| APIs | step/reset/state | Same + evaluation |
| Domains | Coding, Atari, Games | Research domains |
| Output | Training signal | Evaluation scores |

**Novel Contributions:**
1. Combines τ²-bench dual-control with OpenEnv APIs
2. Research-specific domains with academic rigor requirements
3. Policy compliance evaluation for domain rules
4. Database state tracking for verifiable outcomes

---

### 3. How are results different from related benchmarks?

| Dimension | τ²-bench | OpenEnv | ResearchToolBench |
|-----------|----------|---------|-------------------|
| Scoring | Binary pass/fail | Reward signal | 5-dimension weighted |
| Reliability | pass^k | N/A | pass^k + consistency |
| User Model | LLM simulator | N/A | LLM + tool-enabled |
| State Track | DB comparison | Observation | Both |
| RL Ready | Limited | Yes | Yes (OpenEnv APIs) |

---

### 4. What quality issues in agent evaluation does your benchmark address?

**From τ²-bench limitations:**
- ✅ Extends beyond customer service to research domains
- ✅ Adds multi-dimensional scoring beyond binary pass/fail
- ✅ Provides granular policy compliance tracking

**From OpenEnv limitations:**
- ✅ Adds evaluation layer on top of environment APIs
- ✅ Provides ground truth for research quality assessment
- ✅ Includes human-like user simulation

**General issues addressed:**
- ✅ **Reproducibility**: Deterministic task definitions with expected outcomes
- ✅ **Subjectivity**: Objective metrics based on tool use, sources, facts
- ✅ **Single-dimension**: 5 weighted evaluation dimensions
- ✅ **No RL support**: Full OpenEnv API compatibility
- ✅ **Artificial interactions**: τ²-bench style dual-control for realistic troubleshooting

---

### 5. Environment Setup Description

**State Space:**
```python
Observation:
  - task_id: str
  - domain: str ("academic", "news", "technical")
  - user_message: str
  - conversation_history: List[Dict]
  - agent_tools: List[Dict]      # Tools for agent
  - user_tools: List[Dict]       # τ²-bench: Tools for user
  - policies: List[str]          # Domain rules
  - database_state: Dict         # Current DB state
  - user_state: Dict             # User's system state
  - step_count: int
  - max_steps: int
  - done: bool
```

**Action Space:**
```python
Action:
  - type: "tool_call" | "message"
  - tool_name: str (if tool_call)
  - parameters: Dict (if tool_call)
  - message: str (if message)
```

**Environment Dynamics:**
1. Agent receives observation with user query
2. Agent can call tools or send messages
3. User simulator responds (with tool use in dual-control)
4. Database state updates based on actions
5. Episode ends on submission or max steps

**Task Completion:**
- Agent calls submission tool (summarize_findings, submit_report, submit_solution)
- Or max steps reached (truncated)
- Or user indicates satisfaction

---

### 6. Evaluation Methodology

**Metrics with Formulas:**

1. **Tool Use Score (20%)**
   ```
   tool_score = (required_tools_used / total_required_tools) × 100
   ```

2. **Source Score (20%)**
   ```
   source_score = (required_sources_cited / total_required_sources) × 100
   ```

3. **Fact Score (25%)**
   ```
   fact_score = (expected_facts_found / total_expected_facts) × 100
   ```

4. **Policy Score (15%)**
   ```
   policy_score = (1 - violations / total_policies) × 100
   ```

5. **DB State Score (20%)** - τ²-bench style
   ```
   db_score = (matching_state_items / expected_state_items) × 100
   ```

**Total Score:**
```
total = tool×0.20 + source×0.20 + fact×0.25 + policy×0.15 + db_state×0.20
passed = total >= 70
```

**pass^k Metric (τ²-bench):**
```
pass@k = (n_passed / n_trials)^k
```

---

### 7. Data Preparation

**Tasks (4 total):**

| Task ID | Domain | Dual-Control | Difficulty |
|---------|--------|--------------|------------|
| academic_transformers | academic | No | Medium |
| academic_reasoning | academic | No | Hard |
| news_ai_safety | news | No | Medium |
| technical_pytorch_troubleshoot | technical | **Yes** | Hard |

**Databases:**

1. **Academic**: 5 papers (Transformers, BERT, GPT-3, CoT, Constitutional AI)
2. **News**: 3 articles (AI breakthrough, safety concerns, opinion)
3. **Technical**: 2 docs (PyTorch 2.0, Transformers 4.35)

---

### 8. Validation Methodology

**Ground Truth Validation:**
- Each task has predefined expected outcomes
- Required tools, sources, facts, policies specified
- Expected database state defined (τ²-bench style)

**Baseline Testing:**
- Purple agent achieves 60-75% on tasks
- Dual-control task shows user tool tracking works
- pass@1 and pass@2 computed correctly

**Edge Cases Tested:**
- Agent doesn't call required tools → Low tool score
- Missing citations → Low source score
- Policy violations detected → Low policy score
- Database state mismatch → Low DB score

**OpenEnv API Validation:**
- `reset()` returns valid Observation
- `step()` returns valid StepResult
- `state()` returns current EnvironmentState
- `close()` cleans up resources

---

### 9. Reproducibility Confirmation

✅ **Yes, assessments are reproducible**

- Deterministic task definitions
- Fixed database contents
- Consistent evaluation metrics
- Docker containerization ensures environment parity

---

## Quick Checklist

- [x] GitHub repository with source code
- [x] README with setup instructions
- [x] Baseline purple agent (A2A compatible)
- [x] Dockerfile for green agent
- [x] τ²-bench features (dual-control, pass^k, DB state)
- [x] OpenEnv features (step/reset/state/close APIs)
- [x] Multi-dimensional evaluation
- [ ] Demo video (3 min max)
- [ ] AgentBeats registration

---

## Key Differentiators

1. **First benchmark combining τ²-bench + OpenEnv** concepts
2. **Dual-control research domain** where users can run commands
3. **Research-focused evaluation** with academic rigor requirements
4. **5-dimensional scoring** beyond binary pass/fail
5. **RL-ready** with full Gymnasium-style APIs
