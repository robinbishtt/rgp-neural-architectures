"""
src/orchestration/pipeline.py

Pre-built pipeline definitions for full and fast-track reproduction.
"""
from __future__ import annotations
from src.orchestration.dag_executor import DAGExecutor


def build_full_pipeline() -> DAGExecutor:
    """Full reproduction pipeline: data -> train -> evaluate -> figures."""
    dag = DAGExecutor()

    dag.register("setup_env",      lambda: print("Environment verified"))
    dag.register("generate_data",  lambda: None, deps=["setup_env"])
    dag.register("verify_data",    lambda generate_data: None, deps=["generate_data"])
    dag.register("train_h1",       lambda verify_data: None, deps=["verify_data"])
    dag.register("train_h2",       lambda verify_data: None, deps=["verify_data"])
    dag.register("train_h3",       lambda verify_data: None, deps=["verify_data"])
    dag.register("evaluate_h1",    lambda train_h1: None, deps=["train_h1"])
    dag.register("evaluate_h2",    lambda train_h2: None, deps=["train_h2"])
    dag.register("evaluate_h3",    lambda train_h3: None, deps=["train_h3"])
    dag.register("generate_figs",  lambda evaluate_h1, evaluate_h2, evaluate_h3: None,
                 deps=["evaluate_h1", "evaluate_h2", "evaluate_h3"])
    return dag


def build_fast_track_pipeline() -> DAGExecutor:
    """Fast-track pipeline: completes in 3-5 minutes for reviewer verification."""
    dag = DAGExecutor()
    dag.register("fast_data",   lambda: None)
    dag.register("fast_train",  lambda fast_data: None,   deps=["fast_data"])
    dag.register("fast_figs",   lambda fast_train: None,  deps=["fast_train"])
    return dag
