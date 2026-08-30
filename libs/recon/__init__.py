"""Agentic reconnaissance workflow, ported from HexAgent (github.com/urcuqui/HexAgent).

A LangGraph state machine (intake -> plan -> execute -> evaluate -> replan/human
-> report) that drives mock recon/HTTP tools through domain-specialist agents,
gating sensitive actions behind human approval. Mock-only: no real network
activity, no LLM calls — the planner/executor/evaluator use the same
deterministic heuristics HexAgent falls back to offline.
"""
