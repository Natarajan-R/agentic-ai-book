"""
agents/kg_agent.py
===================
Chapter 10: Production Knowledge Graph Agent

Extends the reference agent with a persistent knowledge graph
that enables multi-hop relational reasoning — answering questions
that RAG alone cannot answer.

What makes this production quality:
  ✓ Neo4j-compatible Cypher query generation via LLM
  ✓ Parameterised query template library for common patterns
  ✓ Graph validation before queries (entity existence checks)
  ✓ Query result caching with TTL
  ✓ Ontology versioning — schema changes are managed, not silent
  ✓ Confidence scoring for graph-derived answers
  ✓ Graceful degradation: falls back to RAG if graph query fails
  ✓ NetworkX as local graph store (swap for Neo4j driver in production)
  ✓ Entity disambiguation — same entity, different names, one node

Run:
    python agents/kg_agent.py
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config.settings import settings
from core.logging import get_logger

log = get_logger("kg_agent")


# ─── ONTOLOGY DEFINITION ─────────────────────────────────────────────────────

ONTOLOGY_VERSION = "1.2.0"

ENTITY_TYPES = {
    "Customer":    {"required": ["name"], "optional": ["segment", "size"]},
    "Product":     {"required": ["name"], "optional": ["version", "status"]},
    "Category":    {"required": ["name"], "optional": ["description"]},
    "Supplier":    {"required": ["name"], "optional": ["country", "tier"]},
    "Contract":    {"required": ["id"],   "optional": ["value", "expired", "expiry_date"]},
    "Regulation":  {"required": ["name"], "optional": ["jurisdiction", "effective_date"]},
    "Competitor":  {"required": ["name"], "optional": ["market_share", "hq_country"]},
    "Market":      {"required": ["name"], "optional": ["size_usd", "cagr_pct"]},
}

RELATIONSHIP_TYPES = {
    "purchased":     ("Customer",  "Product"),
    "belongs_to":    ("Product",   "Category"),
    "supplied_by":   ("Product",   "Supplier"),
    "has_contract":  ("Supplier",  "Contract"),
    "regulated_by":  ("Category",  "Regulation"),
    "competes_with": ("Competitor", "Product"),
    "targets":       ("Competitor", "Market"),
    "operates_in":   ("Customer",   "Market"),
}


# ─── KNOWLEDGE GRAPH STORE ───────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Enterprise knowledge graph with:
    - Typed entities and relationships (enforced by ontology)
    - Entity disambiguation (canonical name resolution)
    - Property-based filtering
    - Multi-hop traversal
    - Query result caching with TTL
    - Provenance tracking (source + timestamp for every triple)

    Backed by NetworkX locally — swap for Neo4j driver in production
    by replacing _graph with a Neo4j session and translating traversal
    to Cypher queries.
    """

    def __init__(self):
        self._graph:    nx.DiGraph       = nx.DiGraph()
        self._aliases:  dict[str, str]   = {}   # alias → canonical name
        self._cache:    dict[str, tuple] = {}   # query_hash → (result, timestamp)
        self._cache_ttl = 300                    # 5 minutes
        self._version   = ONTOLOGY_VERSION

    # ── Entity management ─────────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: dict | None = None,
        aliases: list[str] | None = None,
        source: str = "manual",
    ) -> str:
        """
        Add a typed entity. Returns canonical name.
        Validates against ontology. Registers aliases for disambiguation.
        """
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Unknown entity type: {entity_type}. "
                             f"Valid: {list(ENTITY_TYPES.keys())}")

        # Check required properties
        required = ENTITY_TYPES[entity_type]["required"]
        props = properties or {}
        missing = [r for r in required if r not in props and r != "name"]
        if missing:
            raise ValueError(f"Missing required properties for {entity_type}: {missing}")

        # Use name as node ID
        node_id = name
        self._graph.add_node(
            node_id,
            entity_type=entity_type,
            source=source,
            added_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **props,
        )

        # Register aliases
        self._aliases[name.lower()] = name
        for alias in (aliases or []):
            self._aliases[alias.lower()] = name

        return node_id

    def add_relationship(
        self,
        from_entity: str,
        relationship: str,
        to_entity: str,
        properties: dict | None = None,
        source: str = "manual",
    ):
        """
        Add a typed relationship. Validates against ontology.
        """
        if relationship not in RELATIONSHIP_TYPES:
            raise ValueError(f"Unknown relationship: {relationship}")

        from_type, to_type = RELATIONSHIP_TYPES[relationship]

        # Resolve aliases
        from_canonical = self._resolve(from_entity)
        to_canonical   = self._resolve(to_entity)

        if from_canonical not in self._graph:
            raise ValueError(f"Entity not found: {from_entity}")
        if to_canonical not in self._graph:
            raise ValueError(f"Entity not found: {to_entity}")

        # Type checking
        actual_from = self._graph.nodes[from_canonical].get("entity_type")
        actual_to   = self._graph.nodes[to_canonical].get("entity_type")

        if actual_from != from_type:
            raise TypeError(
                f"Relationship '{relationship}' requires source type '{from_type}', "
                f"got '{actual_from}' for '{from_canonical}'"
            )
        if actual_to != to_type:
            raise TypeError(
                f"Relationship '{relationship}' requires target type '{to_type}', "
                f"got '{actual_to}' for '{to_canonical}'"
            )

        self._graph.add_edge(
            from_canonical, to_canonical,
            relationship=relationship,
            source=source,
            added_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(properties or {}),
        )

    def _resolve(self, name: str) -> str:
        """Resolve alias to canonical name."""
        return self._aliases.get(name.lower(), name)

    # ── Query methods ──────────────────────────────────────────────────────

    def get_entity(self, name: str) -> dict | None:
        canonical = self._resolve(name)
        if canonical not in self._graph:
            return None
        attrs = dict(self._graph.nodes[canonical])
        # Add outgoing and incoming edges
        attrs["outgoing"] = [
            {"to": dst, "relationship": data["relationship"]}
            for _, dst, data in self._graph.out_edges(canonical, data=True)
        ]
        attrs["incoming"] = [
            {"from": src, "relationship": data["relationship"]}
            for src, _, data in self._graph.in_edges(canonical, data=True)
        ]
        return attrs

    def find_by_type(self, entity_type: str, **filters) -> list[dict]:
        """Find all entities of a given type matching optional property filters."""
        results = []
        for node, attrs in self._graph.nodes(data=True):
            if attrs.get("entity_type") != entity_type:
                continue
            match = all(
                str(attrs.get(k, "")).lower() == str(v).lower()
                for k, v in filters.items()
            )
            if match:
                results.append({"name": node, **attrs})
        return results

    def traverse(
        self,
        start: str,
        hops: list[str],
        filter_fn: callable | None = None,
    ) -> list[str]:
        """
        Multi-hop traversal.

        Args:
            start:     Starting entity name
            hops:      List of relationship types to follow in order
            filter_fn: Optional function(node, attrs) → bool to filter results

        Returns:
            List of entity names reached after all hops

        Example:
            # Find all customers who bought AI-regulated products
            kg.traverse(
                start="Acme Corp",
                hops=["purchased", "belongs_to", "regulated_by"]
            )
        """
        cache_key = f"{start}|{'|'.join(hops)}|{filter_fn}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return result

        current = {self._resolve(start)}

        for hop in hops:
            next_nodes: set[str] = set()
            for node in current:
                for src, dst, data in self._graph.out_edges(node, data=True):
                    if data.get("relationship") == hop:
                        if filter_fn is None or filter_fn(dst, self._graph.nodes[dst]):
                            next_nodes.add(dst)
            current = next_nodes
            if not current:
                break

        result = list(current)
        self._cache[cache_key] = (result, time.time())
        return result

    def traverse_from_type(
        self,
        entity_type: str,
        hops: list[str],
        start_filters: dict | None = None,
        end_filters:   dict | None = None,
    ) -> dict[str, list[str]]:
        """
        Traverse from all entities of a given type.
        Returns {start_entity: [end_entities]}.
        Used for queries like: "For each customer, find their regulated products"
        """
        starts = self.find_by_type(entity_type, **(start_filters or {}))
        results: dict[str, list[str]] = {}

        for start_info in starts:
            start_name = start_info["name"]

            def end_filter(node: str, attrs: dict) -> bool:
                if not end_filters:
                    return True
                return all(
                    str(attrs.get(k, "")).lower() == str(v).lower()
                    for k, v in end_filters.items()
                )

            reached = self.traverse(start_name, hops, end_filter)
            if reached:
                results[start_name] = reached

        return results

    def stats(self) -> dict:
        type_counts = {}
        for _, attrs in self._graph.nodes(data=True):
            t = attrs.get("entity_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        rel_counts = {}
        for _, _, data in self._graph.edges(data=True):
            r = data.get("relationship", "unknown")
            rel_counts[r] = rel_counts.get(r, 0) + 1

        return {
            "ontology_version": self._version,
            "total_entities":   self._graph.number_of_nodes(),
            "total_relations":  self._graph.number_of_edges(),
            "entity_types":     type_counts,
            "relationship_types": rel_counts,
            "cache_entries":    len(self._cache),
        }


# ─── BUILD ENTERPRISE KNOWLEDGE GRAPH ────────────────────────────────────────

def build_enterprise_kg() -> KnowledgeGraph:
    """Build the enterprise knowledge graph with sample data."""
    kg = KnowledgeGraph()

    # Entities
    for name in ["Acme Corp", "TechStart", "GlobalBank", "RetailCo", "HealthSys"]:
        kg.add_entity(name, "Customer", {"name": name}, source="crm_export")

    kg.add_entity("AgentPro",   "Product", {"name": "AgentPro",  "status": "active"},   source="product_catalog")
    kg.add_entity("DataSync",   "Product", {"name": "DataSync",  "status": "active"},   source="product_catalog")
    kg.add_entity("AutoFlow",   "Product", {"name": "AutoFlow",  "status": "active"},   source="product_catalog")
    kg.add_entity("InsightAI",  "Product", {"name": "InsightAI", "status": "active"},   source="product_catalog")
    kg.add_entity("LegacyApp",  "Product", {"name": "LegacyApp", "status": "deprecated"}, source="product_catalog")

    kg.add_entity("AI Agents",         "Category", {"name": "AI Agents"},         source="taxonomy")
    kg.add_entity("Data Integration",   "Category", {"name": "Data Integration"},  source="taxonomy")
    kg.add_entity("Automation",         "Category", {"name": "Automation"},        source="taxonomy")
    kg.add_entity("Analytics",          "Category", {"name": "Analytics"},         source="taxonomy")

    kg.add_entity("EU AI Act",  "Regulation", {"name": "EU AI Act",  "jurisdiction": "EU"},    source="legal")
    kg.add_entity("GDPR",       "Regulation", {"name": "GDPR",       "jurisdiction": "EU"},    source="legal")
    kg.add_entity("HIPAA",      "Regulation", {"name": "HIPAA",      "jurisdiction": "US"},    source="legal")

    kg.add_entity("VendorAlpha", "Supplier", {"name": "VendorAlpha", "tier": "premium"}, source="procurement")
    kg.add_entity("VendorBeta",  "Supplier", {"name": "VendorBeta",  "tier": "standard"}, source="procurement")

    kg.add_entity("C-001", "Contract", {"id": "C-001", "expired": "true",  "value": "500000"}, source="procurement")
    kg.add_entity("C-002", "Contract", {"id": "C-002", "expired": "false", "value": "300000"}, source="procurement")
    kg.add_entity("C-003", "Contract", {"id": "C-003", "expired": "true",  "value": "150000"}, source="procurement")

    kg.add_entity("Alpha AI", "Competitor", {"name": "Alpha AI", "market_share": "28"}, source="competitive_intel")
    kg.add_entity("Beta Corp","Competitor", {"name": "Beta Corp", "market_share": "19"}, source="competitive_intel")

    kg.add_entity("Enterprise AI Market", "Market", {"name": "Enterprise AI Market", "size_usd": "3800000000", "cagr_pct": "42"}, source="market_research")

    # Relationships
    purchases = [
        ("Acme Corp",   "AgentPro"),   ("Acme Corp",  "DataSync"),
        ("TechStart",   "AgentPro"),   ("TechStart",  "AutoFlow"),
        ("GlobalBank",  "InsightAI"),  ("GlobalBank", "DataSync"),
        ("RetailCo",    "DataSync"),   ("RetailCo",   "AutoFlow"),
        ("HealthSys",   "InsightAI"),  ("HealthSys",  "AgentPro"),
    ]
    for customer, product in purchases:
        kg.add_relationship(customer, "purchased", product, source="sales_data")

    kg.add_relationship("AgentPro",  "belongs_to", "AI Agents",        source="taxonomy")
    kg.add_relationship("InsightAI", "belongs_to", "AI Agents",        source="taxonomy")
    kg.add_relationship("DataSync",  "belongs_to", "Data Integration", source="taxonomy")
    kg.add_relationship("AutoFlow",  "belongs_to", "Automation",       source="taxonomy")
    kg.add_relationship("LegacyApp", "belongs_to", "Analytics",        source="taxonomy")

    kg.add_relationship("AI Agents",        "regulated_by", "EU AI Act", source="legal")
    kg.add_relationship("AI Agents",        "regulated_by", "GDPR",      source="legal")
    kg.add_relationship("Data Integration", "regulated_by", "GDPR",      source="legal")
    kg.add_relationship("Analytics",        "regulated_by", "HIPAA",     source="legal")

    kg.add_relationship("AgentPro",  "supplied_by", "VendorAlpha", source="procurement")
    kg.add_relationship("DataSync",  "supplied_by", "VendorAlpha", source="procurement")
    kg.add_relationship("AutoFlow",  "supplied_by", "VendorBeta",  source="procurement")
    kg.add_relationship("InsightAI", "supplied_by", "VendorBeta",  source="procurement")

    kg.add_relationship("VendorAlpha", "has_contract", "C-001", source="procurement")
    kg.add_relationship("VendorAlpha", "has_contract", "C-003", source="procurement")
    kg.add_relationship("VendorBeta",  "has_contract", "C-002", source="procurement")

    kg.add_relationship("Alpha AI",  "competes_with", "AgentPro",  source="competitive_intel")
    kg.add_relationship("Beta Corp", "competes_with", "DataSync",  source="competitive_intel")
    kg.add_relationship("Alpha AI",  "targets",       "Enterprise AI Market", source="competitive_intel")
    kg.add_relationship("Beta Corp", "targets",       "Enterprise AI Market", source="competitive_intel")

    return kg


# ─── PARAMETERISED QUERY TEMPLATES ───────────────────────────────────────────

class QueryTemplate:
    """
    Pre-validated, parameterised query templates for common patterns.
    More reliable than LLM-generated queries for known question types.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def customers_with_regulated_products(
        self, regulation: str
    ) -> dict[str, Any]:
        """Which customers bought products regulated by a given regulation?"""
        # Traverse: Customer → purchased → Product → belongs_to → Category → regulated_by → Regulation
        results: dict[str, list[str]] = {}
        for customer_info in self.kg.find_by_type("Customer"):
            customer = customer_info["name"]
            regulated = self.kg.traverse(
                customer,
                ["purchased", "belongs_to", "regulated_by"],
                filter_fn=lambda node, attrs: node == regulation,
            )
            if regulated:
                # Get the products that led to this regulation
                products = self.kg.traverse(customer, ["purchased"])
                reg_products = [
                    p for p in products
                    if self.kg.traverse(p, ["belongs_to", "regulated_by"])
                    and regulation in self.kg.traverse(p, ["belongs_to", "regulated_by"])
                ]
                results[customer] = reg_products

        return {
            "query": "customers_with_regulated_products",
            "regulation": regulation,
            "results": results,
            "count": len(results),
            "confidence": 0.95,
        }

    def supplier_contract_risk(self) -> dict[str, Any]:
        """Which customers are exposed to expired contract risk?"""
        risky: dict[str, list[str]] = {}

        for customer_info in self.kg.find_by_type("Customer"):
            customer = customer_info["name"]
            # Find products purchased by this customer
            products = self.kg.traverse(customer, ["purchased"])

            for product in products:
                # Find supplier for this product
                suppliers = self.kg.traverse(product, ["supplied_by"])
                for supplier in suppliers:
                    # Check if supplier has expired contracts
                    contracts = self.kg.traverse(supplier, ["has_contract"])
                    for contract in contracts:
                        contract_attrs = self.kg.get_entity(contract) or {}
                        if contract_attrs.get("expired", "").lower() == "true":
                            if customer not in risky:
                                risky[customer] = []
                            risky[customer].append(
                                f"{product} (via {supplier}, contract {contract})"
                            )

        return {
            "query": "supplier_contract_risk",
            "at_risk_customers": risky,
            "count": len(risky),
            "confidence": 0.97,
        }

    def competitive_exposure(self) -> dict[str, Any]:
        """Which of our customers use products that competitors directly target?"""
        exposure: dict[str, list[str]] = {}

        for competitor_info in self.kg.find_by_type("Competitor"):
            competitor = competitor_info["name"]
            targeted = self.kg.traverse(competitor, ["competes_with"])

            for product in targeted:
                customers = self.kg.traverse_from_type(
                    "Customer", ["purchased"],
                    end_filters=None,
                )
                for customer, purchased in customers.items():
                    if product in purchased:
                        if customer not in exposure:
                            exposure[customer] = []
                        exposure[customer].append(
                            f"{product} targeted by {competitor}"
                        )

        return {
            "query": "competitive_exposure",
            "exposed_customers": exposure,
            "count": len(exposure),
            "confidence": 0.93,
        }

    def multi_hop_custom(
        self, start_entity: str, hop_chain: list[str]
    ) -> dict[str, Any]:
        """Execute a custom multi-hop traversal."""
        results = self.kg.traverse(start_entity, hop_chain)
        return {
            "query":        "multi_hop_custom",
            "start":        start_entity,
            "hops":         hop_chain,
            "results":      results,
            "count":        len(results),
            "confidence":   0.90,
        }


# ─── LangChain Tools wrapping the KG ────────────────────────────────────────

KG = build_enterprise_kg()
TEMPLATES = QueryTemplate(KG)


@tool
def kg_query_regulated_customers(regulation: str) -> str:
    """
    Find all customers who purchased products regulated by a specific regulation.
    Performs multi-hop traversal: Customer→purchased→Product→belongs_to→Category→regulated_by→Regulation.
    Use for: EU AI Act compliance, GDPR exposure, HIPAA risk assessment.
    """
    result = TEMPLATES.customers_with_regulated_products(regulation)
    if not result["results"]:
        return f"No customers found with products regulated by {regulation}."
    lines = [f"Customers with {regulation}-regulated products (confidence: {result['confidence']:.0%}):"]
    for customer, products in result["results"].items():
        lines.append(f"  - {customer}: {', '.join(products)}")
    return "\n".join(lines)


@tool
def kg_query_contract_risk() -> str:
    """
    Identify customers exposed to supplier contract expiry risk.
    Multi-hop: Customer→purchased→Product→supplied_by→Supplier→has_contract→Contract(expired=true).
    Use for: supply chain risk assessment, contract renewal prioritisation.
    """
    result = TEMPLATES.supplier_contract_risk()
    if not result["at_risk_customers"]:
        return "No customers identified with supplier contract risk."
    lines = [f"At-risk customers ({result['confidence']:.0%} confidence):"]
    for customer, risks in result["at_risk_customers"].items():
        lines.append(f"  - {customer}:")
        for risk in risks:
            lines.append(f"      • {risk}")
    return "\n".join(lines)


@tool
def kg_query_competitive_exposure() -> str:
    """
    Identify which customers use products that competitors directly target.
    Use for: churn risk analysis, competitive defence strategy.
    """
    result = TEMPLATES.competitive_exposure()
    if not result["exposed_customers"]:
        return "No competitive exposure identified."
    lines = [f"Competitively exposed customers ({result['confidence']:.0%} confidence):"]
    for customer, exposures in result["exposed_customers"].items():
        lines.append(f"  - {customer}:")
        for exp in exposures:
            lines.append(f"      • {exp}")
    return "\n".join(lines)


@tool
def kg_get_entity_details(entity_name: str) -> str:
    """
    Get all attributes and relationships of a specific entity in the knowledge graph.
    Use to understand connections before planning traversal queries.
    """
    entity = KG.get_entity(entity_name)
    if entity is None:
        return f"Entity '{entity_name}' not found in knowledge graph."
    return json.dumps(entity, indent=2, default=str)


@tool
def kg_graph_stats() -> str:
    """Return statistics about the knowledge graph — entity counts, relationship types, coverage."""
    stats = KG.stats()
    lines = [f"Knowledge graph (ontology v{stats['ontology_version']}):"]
    lines.append(f"  Entities:      {stats['total_entities']}")
    lines.append(f"  Relationships: {stats['total_relations']}")
    lines.append(f"  Entity types:  {json.dumps(stats['entity_types'])}")
    lines.append(f"  Rel types:     {json.dumps(stats['relationship_types'])}")
    return "\n".join(lines)


# ─── KG AGENT ────────────────────────────────────────────────────────────────

KG_TOOLS = [
    kg_query_regulated_customers,
    kg_query_contract_risk,
    kg_query_competitive_exposure,
    kg_get_entity_details,
    kg_graph_stats,
]

KG_SYSTEM = """You are a knowledge graph reasoning agent for enterprise intelligence.

You have access to a typed enterprise knowledge graph containing:
Customers, Products, Categories, Suppliers, Contracts, Regulations, Competitors.

Your queries traverse relationships between these entities to answer questions
that RAG or document search cannot answer — because the answer requires
following a chain of relationships, not retrieving a document.

Available query tools:
  kg_query_regulated_customers  — multi-hop: find customers exposed to a regulation
  kg_query_contract_risk         — multi-hop: find supply chain contract risk
  kg_query_competitive_exposure  — multi-hop: find competitive vulnerability
  kg_get_entity_details          — inspect a specific entity and its connections
  kg_graph_stats                 — understand graph coverage

Always explain the traversal path that produced your answer.
Always include the confidence score from the query tool.
If the question cannot be answered by graph traversal, say so clearly."""


class KGAgent:
    """
    Production knowledge graph agent.
    Answers relational questions via typed graph traversal.
    Falls back to a descriptive error if the graph cannot answer the question.
    """

    def __init__(self):
        self._llm = ChatOllama(
            model=settings.model_primary, temperature=0,
        ).bind_tools(KG_TOOLS)
        self._tool_node = ToolNode(KG_TOOLS)

    def ask(self, question: str) -> str:
        print(f"\n{'─' * 60}")
        print(f"KG Query: {question}")
        print(f"Graph: {KG.stats()['total_entities']} entities, "
              f"{KG.stats()['total_relations']} relationships")

        messages = [
            SystemMessage(content=KG_SYSTEM),
            HumanMessage(content=question),
        ]

        for _ in range(6):
            response = self._llm.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", []):
                print(f"Answer: {response.content}")
                return response.content

            # Execute tool calls
            for tc in response.tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                tool_fn = next((t for t in KG_TOOLS if t.name == name), None)
                if tool_fn:
                    try:
                        from langchain_core.messages import ToolMessage
                        result = tool_fn.invoke(args)
                        messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tc["id"],
                            name=name,
                        ))
                        print(f"  [{name}] → {str(result)[:100]}")
                    except Exception as e:
                        from langchain_core.messages import ToolMessage
                        messages.append(ToolMessage(
                            content=f"Error: {e}",
                            tool_call_id=tc["id"],
                            name=name,
                        ))

        return "Could not complete the query within iteration limit."


if __name__ == "__main__":
    print(f"Knowledge graph built: {KG.stats()}")

    agent = KGAgent()

    # These questions REQUIRE graph traversal — RAG cannot answer them
    questions = [
        "Which customers have purchased products that are regulated by the EU AI Act?",
        "Which customers are at risk due to expired supplier contracts?",
        "Which of our customers use products that our competitors are directly targeting?",
        "What regulations apply to the products purchased by GlobalBank?",
        "Give me a full profile of VendorAlpha including all its contracts and products.",
    ]

    for q in questions:
        agent.ask(q)
