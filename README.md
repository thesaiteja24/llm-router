# LLM Router

An intelligent query routing system that dynamically routes user queries to specialized LLM experts based on task classification. Implements context-aware routing with adaptive parameter tuning.

## 🎯 Key Features

- **Dynamic Task Routing**: Classifies queries into 4 categories (code, reasoning, summarization, general)
- **Expert Selection**: Routes to specialized models (GPT-4o, GPT-4o-mini)
- **Adaptive Parameters**: Dynamically adjusts temperature, max_tokens based on task
- **Cost Optimization**: Uses cheaper models for simpler tasks
- **Evaluation Framework**: Benchmarks with publication-quality chart generation

---

## 📁 Project Structure

```
llm-router/
├── app.py                    # Streamlit chat interface
├── routing_engine.py         # Core routing + expert execution
├── config.py                 # API key configuration
├── dashboard.py              # Metrics visualization dashboard
├── generate_paper_charts.py  # Publication chart generator
│
├── router/                   # Routing Module
│   ├── router_llm.py        # LLM-based classifier
│   ├── prompt.py            # System prompt for router
│   └── schema.py            # Pydantic schemas
│
├── experts/                  # Expert Models
│   ├── base.py              # Base expert class
│   ├── gpt_code.py          # GPT-4o for code generation
│   ├── gpt_reasoning.py     # GPT-4o for reasoning
│   ├── gpt_summary.py       # GPT-4o-mini for summarization
│   └── fallback.py          # GPT-4o-mini fallback
│
├── evaluation/               # Evaluation Framework
│   ├── build_dataset.py     # Dataset builder (HuggingFace/synthetic)
│   ├── run_evaluation.py    # Evaluation runner
│   ├── metrics.py           # Metrics & cost calculation
│   └── results.json         # Evaluation results
│
└── paper_charts/            # Generated publication charts
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file:

```env
OPENAI_API_KEY=sk-your-openai-key
GOOGLE_API_KEY=your-google-api-key
```

### 3. Run Chat Interface

```bash
streamlit run app.py
```

---

## 📋 Commands Reference

| Command                                                 | Description                            |
| ------------------------------------------------------- | -------------------------------------- |
| `streamlit run app.py`                                  | Launch interactive chat interface      |
| `streamlit run dashboard.py`                            | Launch metrics visualization dashboard |
| `python3 evaluation/build_dataset.py`                   | Build evaluation dataset (500 samples) |
| `python3 evaluation/run_evaluation.py`                  | Run full evaluation                    |
| `python3 evaluation/run_evaluation.py --max-samples 20` | Quick evaluation (20 samples)          |
| `python3 generate_paper_charts.py`                      | Generate all publication charts        |

---

## 📊 Evaluation Workflow

```bash
# Step 1: Build dataset (uses HuggingFace if available, else synthetic)
python evaluation/build_dataset.py

# Step 2: Run evaluation (adjust samples as needed)
python evaluation/run_evaluation.py --max-samples 50

# Step 3: Generate publication-ready charts
python generate_paper_charts.py
```

**Output**: `paper_charts/` folder with 10 PNG charts (300 DPI)

---

## 🔧 How It Works

1. **User Query** → Router LLM classifies task type
2. **Router Decision** → Selects expert + assigns parameters
3. **Expert Execution** → Specialized model generates response
4. **Metrics Tracking** → Latency, cost, accuracy logged

### Task Categories & Experts

| Task Type        | Expert          | Model       |
| ---------------- | --------------- | ----------- |
| Code Generation  | `gpt_code`      | GPT-4o      |
| Reasoning        | `gpt_reasoning` | GPT-4o      |
| Summarization    | `gpt_summary`   | GPT-4o-mini |
| General/Fallback | `fallback`      | GPT-4o-mini |

---

## 📈 Generated Charts

| #   | Chart                   | Purpose                           |
| --- | ----------------------- | --------------------------------- |
| 01  | Accuracy by Category    | Routing accuracy per task type    |
| 02  | Cost Comparison         | Router vs single-model baselines  |
| 03  | Latency Distribution    | Response times by expert          |
| 04  | Cost per Query          | Cost efficiency by task           |
| 05  | Router Overhead         | Classification vs generation time |
| 06  | Confusion Matrix        | Predicted vs expected tasks       |
| 07  | Expert Usage            | Distribution of expert selection  |
| 08  | Confidence Distribution | Router confidence scores          |
| 09  | Temperature by Task     | Dynamic temperature assignment    |
| 10  | Max Tokens by Task      | Dynamic token limit assignment    |

---

## 📚 Datasets Used (HuggingFace)

| Task          | Dataset         | Description                 |
| ------------- | --------------- | --------------------------- |
| Code          | HumanEval, MBPP | Python programming problems |
| Reasoning     | GSM8K           | Grade school math problems  |
| Summarization | XSum            | BBC article summarization   |
| General       | TriviaQA        | Trivia questions            |

To use real datasets: `pip install datasets`

---

## 📄 License

MIT License
