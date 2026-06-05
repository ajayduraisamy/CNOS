"""Benchmark — evaluates the Dynamic Layer Router across 300 queries.

Generates 100 queries of each complexity type (simple, medium, complex),
runs them through the :class:`LayerSelector` under multiple routing
policies, and produces a comparative report.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from complexity_detector import ComplexityDetector
from layer_selector import LayerSelector
from metrics import CumulativeMetrics
from routing_policy import StaticPolicy, AdaptivePolicy, ExperimentalPolicy, create_policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query bank — curated examples for each complexity tier
# ---------------------------------------------------------------------------

SIMPLE_QUERIES: List[str] = [
    "What is 2+2?",
    "What is the capital of France?",
    "What day is it today?",
    "How are you?",
    "What is your name?",
    "Is the sky blue?",
    "How many legs does a dog have?",
    "What colour is an apple?",
    "Who wrote Romeo and Juliet?",
    "What is 10 times 5?",
    "Where is the Eiffel Tower?",
    "What is the opposite of hot?",
    "How many days in a week?",
    "What is the speed of light?",
    "Who is the president of the United States?",
    "What year was the internet invented?",
    "How many continents are there?",
    "What is the largest ocean?",
    "What language is spoken in Brazil?",
    "What is the boiling point of water?",
    "Is the earth round?",
    "What is a triangle?",
    "How many sides does a square have?",
    "What is the colour of the sky on a clear day?",
    "Who invented the light bulb?",
    "What is the smallest planet?",
    "What is the chemical symbol for water?",
    "How many bones are in the human body?",
    "What is the tallest mountain?",
    "What is the square root of 16?",
    "How to get the length of a list in Python?",
    "What does print() do in Python?",
    "How to write a comment in Python?",
    "What is a variable?",
    "How to install pip?",
    "What is a function in Python?",
    "How to open a file in Python?",
    "What is a list in Python?",
    "How to check Python version?",
    "What is a dictionary in Python?",
    "How to loop in Python?",
    "What does len() return?",
    "How to import a module?",
    "What is a string in Python?",
    "How to convert int to string?",
    "What is a boolean in Python?",
    "How to use if-else in Python?",
    "What is a tuple in Python?",
    "How to define a class in Python?",
    "What is an integer in Python?",
]

MEDIUM_QUERIES: List[str] = [
    "Write a Python function to find all prime numbers up to n.",
    "Write Python code for binary search.",
    "Implement a stack data structure in Python.",
    "Write a function that reverses a linked list.",
    "Write a Python function to merge two sorted arrays.",
    "Explain how the bubble sort algorithm works.",
    "Write a recursive solution to the Fibonacci sequence.",
    "Implement a simple HTTP server in Python.",
    "Write a SQL query to find duplicate records in a table.",
    "Explain the difference between REST and GraphQL.",
    "How do you deploy a Flask application to production?",
    "Write a regular expression to validate email addresses.",
    "Explain how virtual memory works.",
    "What is a deadlock and how do you prevent it?",
    "Write a Python script to scrape a website.",
    "Explain the difference between TCP and UDP.",
    "How does a relational database index work?",
    "Write a Python class for a simple bank account with deposit and withdraw.",
    "Explain the concept of time complexity.",
    "Write a function that checks if a string is a palindrome.",
    "How do you handle exceptions in Python?",
    "Write a SQL JOIN query for customers and orders.",
    "Explain what an API rate limiter is and how to implement one.",
    "Write a Python decorator that measures execution time.",
    "How does the MapReduce programming model work?",
    "Write a Python generator that yields Fibonacci numbers.",
    "Explain the principles of object-oriented programming.",
    "Write a unit test for a calculator function in pytest.",
    "How do you manage database migrations in a web application?",
    "Write a Python context manager for timing code blocks.",
    "Explain what a microservice is and its advantages.",
    "Write a function to detect cycles in a directed graph.",
    "How do you implement caching in a web API?",
    "Write a Python function to solve the knapsack problem.",
    "Explain the CAP theorem in distributed systems.",
    "Write a Python script to read and write CSV files.",
    "How does a load balancer distribute traffic?",
    "Write a function to find the longest common subsequence.",
    "Explain the difference between authorization and authentication.",
    "Write a Python function using lambda and filter.",
    "How do you secure a REST API?",
    "Write a Python function to compute Levenshtein distance.",
    "Explain the event-driven architecture pattern.",
    "Write a SQL query with GROUP BY and HAVING clauses.",
    "How do you implement pagination in an API?",
    "Write a Python function for matrix multiplication.",
    "Explain the concept of idempotency in APIs.",
    "Write a pytest fixture for a test database.",
    "How does a CDN improve website performance?",
    "Write a function to perform topological sort on a DAG.",
    "Explain the difference between a thread and a process.",
    "Write a Python class for a thread-safe queue.",
    "How do you implement rate limiting in a web application?",
    "Write a function to compress a string using run-length encoding.",
    "Explain the ACID properties of database transactions.",
    "Write a Python function for breadth-first search on a graph.",
    "How does a message queue like RabbitMQ work?",
    "Write a function to find the median of two sorted arrays.",
    "Explain how garbage collection works in Python.",
    "Write a Python function using the Strategy design pattern.",
]

COMPLEX_QUERIES: List[str] = [
    "Design a distributed microservice architecture for an e-commerce platform.",
    "Design a fault-tolerant distributed database system with consensus.",
    "Architect a real-time streaming data pipeline for millions of events per second.",
    "Design a multi-tenant SaaS platform with horizontal scaling.",
    "Design a distributed caching layer with consistent hashing.",
    "Architect a global content delivery network with edge computing.",
    "Design a blockchain-based supply chain tracking system.",
    "Architect a high-frequency trading platform with nanosecond latency.",
    "Design a distributed file system inspired by Google File System.",
    "Architect an end-to-end ML pipeline for real-time fraud detection.",
    "Design a container orchestration platform like Kubernetes.",
    "Architect a multi-region active-active database deployment.",
    "Design a distributed transaction coordinator for microservices.",
    "Architect a real-time collaborative editing system like Google Docs.",
    "Design a distributed metrics and monitoring system at scale.",
    "Architect an auto-scaling recommendation engine for 100M users.",
    "Design a cross-datacenter replication strategy with conflict resolution.",
    "Architect a serverless computing platform with cold-start mitigation.",
    "Design a distributed search engine with inverted indexes and sharding.",
    "Architect a real-time anomaly detection system for network security.",
    "Design a global user authentication and authorization infrastructure.",
    "Architect a distributed task queue with priority scheduling and retries.",
    "Design a hybrid cloud multi-region disaster recovery architecture.",
    "Architect an end-to-end encrypted messaging platform with forward secrecy.",
    "Design a distributed ledger system for interbank settlements.",
    "Architect a real-time video transcoding pipeline at scale.",
    "Design a distributed configuration management system for microservices.",
    "Architect an autoscaling web scraping infrastructure with proxy rotation.",
    "Design a distributed SQL database with MVCC and snapshot isolation.",
    "Architect a global DNS load balancing system with health checks.",
    "Design a streaming SQL engine for real-time analytics.",
    "Architect a distributed tracing system with sampling and aggregation.",
    "Design a multi-cloud cost optimisation and resource scheduling framework.",
    "Architect a real-time personalisation engine with feature store and online learning.",
    "Design a distributed lock service based on Paxos or Raft.",
    "Architect an API gateway with rate limiting, auth, and circuit breaking.",
    "Design a distributed metrics aggregator with downsampling and retention policies.",
    "Architect a chaos engineering platform for distributed systems testing.",
    "Design a global session management system with sticky-less routing.",
    "Architect a real-time fraud detection ensemble with model versioning.",
    "Design a distributed neural network training infrastructure with gradient compression.",
    "Architect a multi-tenant vector database for semantic search at scale.",
    "Design a federated learning platform with differential privacy.",
    "Architect a distributed time-series database with columnar storage.",
    "Design a real-time AB testing platform with statistical significance computation.",
    "Architect a global data pipeline with schema registry and evolution.",
    "Design a distributed rate limiter with sliding window counters.",
    "Architect a multi-cloud service mesh with zero-trust security.",
    "Design a distributed transaction outbox pattern for event-driven systems.",
    "Architect a real-time compliance monitoring system for financial transactions.",
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregate benchmark output for a single policy."""

    policy_name: str
    metrics: CumulativeMetrics
    elapsed_seconds: float


class Benchmark:
    """Runs the router over a standardised query set and reports metrics.

    Args:
        policies: List of policy names to evaluate.
        num_simple: Number of simple queries to use.
        num_medium: Number of medium queries to use.
        num_complex: Number of complex queries to use.
        num_layers: Total transformer layers.
    """

    def __init__(
        self,
        policies: Optional[List[str]] = None,
        num_simple: int = 100,
        num_medium: int = 100,
        num_complex: int = 100,
        num_layers: int = 80,
    ) -> None:
        self.policies = policies or ["static", "adaptive", "experimental/density"]
        self.num_layers = num_layers
        self._queries: Dict[str, List[str]] = {
            "simple": SIMPLE_QUERIES * (num_simple // len(SIMPLE_QUERIES) + 1),
            "medium": MEDIUM_QUERIES * (num_medium // len(MEDIUM_QUERIES) + 1),
            "complex": COMPLEX_QUERIES * (num_complex // len(COMPLEX_QUERIES) + 1),
        }
        self._counts = {"simple": num_simple, "medium": num_medium, "complex": num_complex}

        logger.info(
            "Benchmark initialised  —  %d policies, %d queries total",
            len(self.policies),
            sum(self._counts.values()),
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Evaluate every policy and return results."""
        results: List[BenchmarkResult] = []
        for policy_name in self.policies:
            result = self._run_single(policy_name)
            results.append(result)

        self._print_comparison(results)
        return results

    def _run_single(self, policy_name: str) -> BenchmarkResult:
        """Run all queries through a single policy."""
        selector = LayerSelector(
            detector=ComplexityDetector(self.num_layers),
            policy=create_policy(policy_name, self.num_layers),
            num_layers=self.num_layers,
        )
        metrics = CumulativeMetrics()
        metrics.policy_name = policy_name

        start = time.perf_counter()

        for qtype in ("simple", "medium", "complex"):
            queries = self._queries[qtype][: self._counts[qtype]]
            for query in queries:
                result = selector.select(query)
                metrics.update(result)

        elapsed = time.perf_counter() - start
        logger.info("Policy %s finished in %.2f s", policy_name, elapsed)

        return BenchmarkResult(policy_name=policy_name, metrics=metrics, elapsed_seconds=elapsed)

    def _print_comparison(self, results: List[BenchmarkResult]) -> None:
        """Print a side-by-side policy comparison."""
        print("\n" + "=" * 70)
        print("  Policy Comparison Summary")
        print("=" * 70)
        header = f"  {'Policy':<24} {'Layers/Query':<14} {'Reduction':<12} {'Time (s)':<10} {'Mem Saved':<10}"
        sep = f"  {'-'*24} {'-'*14} {'-'*12} {'-'*10} {'-'*10}"
        print(header)
        print(sep)
        for r in results:
            s = r.metrics.summary()
            print(
                f"  {r.policy_name:<24} "
                f"{s['avg_layers_selected']:<14} "
                f"{s['compute_reduction_pct']:<11}% "
                f"{r.elapsed_seconds:<10.2f} "
                f"{s['memory_savings_estimate_mb']:<10.0f}"
            )
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stdout,
        format="[%(levelname)s] %(message)s",
    )

    # Parse simple flags (or use defaults)
    import argparse
    parser = argparse.ArgumentParser(description="CNOS Dynamic Layer Router — Benchmark")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Policies to benchmark (e.g. static adaptive experimental/density)")
    parser.add_argument("--num-simple", type=int, default=100)
    parser.add_argument("--num-medium", type=int, default=100)
    parser.add_argument("--num-complex", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=80)
    args = parser.parse_args()

    bench = Benchmark(
        policies=args.policies,
        num_simple=args.num_simple,
        num_medium=args.num_medium,
        num_complex=args.num_complex,
        num_layers=args.num_layers,
    )
    print(f"\n  CNOS Dynamic Layer Router — Benchmark")
    print(f"  {args.num_simple + args.num_medium + args.num_complex} queries "
          f"({args.num_simple}S / {args.num_medium}M / {args.num_complex}C) "
          f"| {args.num_layers} layers")
    print(f"  Policies: {', '.join(bench.policies)}")

    results = bench.run_all()

    # Print full report for the first policy
    if results:
        print(f"\nDetailed report for policy: {results[0].policy_name}")
        results[0].metrics.print_report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
