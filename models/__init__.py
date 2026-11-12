"""
Model client layer — abstracts LLM provider calls with fallback and budget tracking.

Architecture:
  ModelClient (unified interface)
    ├── GroqClient (real Groq API, OpenAI-compatible)
    ├── FallbackRouter (decides which model/service to use)
    └── TokenBudgetTracker (monitors quota usage proactively)
"""
from models.model_client import ModelClient
from models.fallback_router import FallbackRouter
from models.token_budget_tracker import TokenBudgetTracker
