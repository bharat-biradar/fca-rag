"""Lazy singletons for expensive clients — load once, reuse everywhere."""

from __future__ import annotations

from src.config import Settings, settings

_weaviate_client = None
_neo4j_driver = None
_embedding_model = None


def get_weaviate_client(cfg: Settings = settings):
    global _weaviate_client
    if _weaviate_client is None:
        import weaviate

        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cfg.weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(cfg.weaviate_api_key),
        )
    return _weaviate_client


def get_neo4j_driver(cfg: Settings = settings):
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase

        _neo4j_driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
        )
    return _neo4j_driver


def get_embedding_model(cfg: Settings = settings):
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(cfg.embedding_model)
    return _embedding_model


def close_all():
    global _weaviate_client, _neo4j_driver
    if _weaviate_client is not None:
        _weaviate_client.close()
        _weaviate_client = None
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None
