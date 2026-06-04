"""Central configuration — all tunable params and constants."""

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


SOURCEBOOK_NAMES: dict[str, str] = {
    "BCOBS": "Banking: Conduct of Business Sourcebook",
    "CASS": "Client Assets Sourcebook",
    "CMCOB": "Claims Management: Conduct of Business Sourcebook",
    "COBS": "Conduct of Business Sourcebook",
    "ESG": "Environmental, Social and Governance Sourcebook",
    "FPCOB": "Funeral Plan: Conduct of Business Sourcebook",
    "ICOBS": "Insurance: Conduct of Business Sourcebook",
    "MAR": "Market Conduct Sourcebook",
    "MCOB": "Mortgages and Home Finance: Conduct of Business Sourcebook",
    "PDCOB": "Pensions Dashboards: Conduct of Business Sourcebook",
}

VALID_SOURCEBOOKS = tuple(SOURCEBOOK_NAMES.keys())
SOURCEBOOK_PATTERN = "|".join(VALID_SOURCEBOOKS)

RULE_TYPES = ("R", "G", "E", "D", "EU", "UK")

# Matches rule IDs like "COBS 2.1.1", "BCOBS 1.1.4A", "ESG 1A.1.1"
RULE_ID_RE = re.compile(
    rf"({SOURCEBOOK_PATTERN})"
    r"\s+(\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]*)"
)

# Matches cross-references in text with optional type suffix and sub-para ref
# Allows references inside *italic*, (parens), or after whitespace
XREF_RE = re.compile(
    rf"(?:^|[\s(*])({SOURCEBOOK_PATTERN})"
    r"\s+([\d]+[A-Z]?\.[\d]+[A-Z]?\.[\d]+[A-Z]*)"
    r"([RGDEUK]{0,2})"
    r"(?:\([^)]*\))*"
)


@dataclass
class Settings:
    # Paths
    parsed_json_dir: str = "llama_parse_output"

    # Chunking
    min_child_tokens: int = 50
    max_parent_tokens: int = 1500

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_dim: int = 1024

    # Weaviate
    weaviate_url: str = field(default_factory=lambda: os.getenv("WEAVIATE_URL", ""))
    weaviate_api_key: str = field(default_factory=lambda: os.getenv("WEAVIATE_API_KEY", ""))
    weaviate_collection: str = "FCARule"

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", ""))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))

    # LLM (OpenRouter)
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    generation_model: str = "openai/gpt-oss-120b"
    fallback_model: str = "google/gemma-4-26b-a4b-it:free"

    # Retrieval
    hybrid_alpha: float = 0.5
    initial_retrieval_k: int = 50
    final_top_k: int = 5
    graph_hops: int = 2
    graph_expansion_limit: int = 30
    max_agent_steps: int = 5


settings = Settings()
