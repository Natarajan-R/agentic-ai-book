"""
Chapter 10: Knowledge Graph + LangGraph
=========================================
Multi-hop relational reasoning using NetworkX as the graph store.
Agent queries the knowledge graph to answer relational questions
that RAG alone cannot answer.

Run:  python ch10_knowledge_graph.py
Need: ollama pull qwen2.5:14b
      pip install langgraph langchain-ollama networkx
"""

import json, re
import networkx as nx
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

MODEL    = "qwen2.5:7b"
MAX_HOPS = 6                      # hard cap on reason<->tools loops
llm      = ChatOllama(model=MODEL, temperature=0)


# ─── BUILD KNOWLEDGE GRAPH ───────────────────────────────────────────────────

def build_knowledge_graph() -> nx.DiGraph:
    """
    Ontology: Customer, Product, Category, Supplier, Contract, Regulation
    Relationships: purchased, belongs_to, supplied_by, has_contract, regulated_by
    """
    G = nx.DiGraph()

    # Entities
    customers  = ["Acme Corp", "TechStart", "GlobalBank", "RetailCo"]
    products   = ["AgentPro",  "DataSync",  "AutoFlow",   "InsightAI"]
    categories = ["AI Agents", "Data Integration", "Automation"]
    suppliers  = ["VendorAlpha", "VendorBeta"]
    contracts  = [
        {"id": "C001", "supplier": "VendorAlpha", "expired": True},
        {"id": "C002", "supplier": "VendorBeta",  "expired": False},
    ]
    regulations = ["EU AI Act", "GDPR"]

    # Add nodes with type attribute
    for c in customers:   G.add_node(c, type="Customer")
    for p in products:    G.add_node(p, type="Product")
    for cat in categories:G.add_node(cat, type="Category")
    for s in suppliers:   G.add_node(s, type="Supplier")
    for r in regulations: G.add_node(r, type="Regulation")
    for c in contracts:
        G.add_node(c["id"], type="Contract",
                   expired=c["expired"], supplier=c["supplier"])

    # Relationships: Customer → purchased → Product
    purchases = [
        ("Acme Corp",   "AgentPro"),
        ("Acme Corp",   "DataSync"),
        ("TechStart",   "AgentPro"),
        ("TechStart",   "AutoFlow"),
        ("GlobalBank",  "InsightAI"),
        ("RetailCo",    "DataSync"),
        ("RetailCo",    "AutoFlow"),
    ]
    for cust, prod in purchases:
        G.add_edge(cust, prod, rel="purchased")

    # Product → belongs_to → Category
    G.add_edge("AgentPro",  "AI Agents",          rel="belongs_to")
    G.add_edge("InsightAI", "AI Agents",          rel="belongs_to")
    G.add_edge("DataSync",  "Data Integration",    rel="belongs_to")
    G.add_edge("AutoFlow",  "Automation",          rel="belongs_to")

    # Category → regulated_by → Regulation
    G.add_edge("AI Agents", "EU AI Act", rel="regulated_by")
    G.add_edge("AI Agents", "GDPR",      rel="regulated_by")

    # Product → supplied_by → Supplier
    G.add_edge("AgentPro",  "VendorAlpha", rel="supplied_by")
    G.add_edge("DataSync",  "VendorAlpha", rel="supplied_by")
    G.add_edge("AutoFlow",  "VendorBeta",  rel="supplied_by")
    G.add_edge("InsightAI", "VendorBeta",  rel="supplied_by")

    # Supplier → has_contract → Contract
    G.add_edge("VendorAlpha", "C001", rel="has_contract")
    G.add_edge("VendorBeta",  "C002", rel="has_contract")

    return G


KG = build_knowledge_graph()


# ─── GRAPH QUERY TOOLS ───────────────────────────────────────────────────────

@tool
def kg_find_neighbors(entity: str, relationship: str) -> str:
    """Find all entities connected to 'entity' via 'relationship' in the knowledge graph."""
    results = []
    for src, dst, data in KG.edges(data=True):
        if data.get("rel") == relationship:
            if src == entity:
                results.append(dst)
            if dst == entity and not KG.is_directed():
                results.append(src)
    return json.dumps(results) if results else f"No {relationship} connections found for {entity}."

@tool
def kg_multihop_query(start_entity: str, hop_pattern: str) -> str:
    """
    Traverse multiple relationship hops from a start entity.
    hop_pattern: relationship names separated by '->'. Prefix a relationship
    with '<' to traverse it BACKWARDS (against the arrow direction).
    Forward example: 'belongs_to -> regulated_by'
    Backward example: '<regulated_by -> <belongs_to -> <purchased'
    """
    hops = [h.strip() for h in hop_pattern.split("->")]
    current_nodes = {start_entity}
    path_log = [f"Start: {start_entity}"]

    for hop in hops:
        reverse = hop.startswith("<")
        rel = hop.lstrip("<").strip()
        next_nodes = set()
        for src, dst, data in KG.edges(data=True):
            if data.get("rel") != rel:
                continue
            if not reverse and src in current_nodes:
                next_nodes.add(dst)
            if reverse and dst in current_nodes:
                next_nodes.add(src)
        if not next_nodes:
            return f"No results after hop '{hop}' from {list(current_nodes)}"
        current_nodes = next_nodes
        path_log.append(f"After '{hop}': {list(current_nodes)}")

    return "\n".join(path_log) + f"\n\nFinal nodes: {list(current_nodes)}"

@tool
def kg_filter_by_attribute(node_type: str, attribute: str, value: str) -> str:
    """Find all nodes of a given type where an attribute equals a value."""
    results = []
    for node, attrs in KG.nodes(data=True):
        if attrs.get("type") == node_type:
            attr_val = str(attrs.get(attribute, "")).lower()
            if attr_val == value.lower():
                results.append(node)
    return json.dumps(results) if results else f"No {node_type} nodes with {attribute}={value}."

@tool
def kg_get_node_info(entity: str) -> str:
    """Get all attributes and relationships of a specific entity."""
    if entity not in KG:
        return f"Entity '{entity}' not found in knowledge graph."
    attrs = dict(KG.nodes[entity])
    out_edges = [(dst, data["rel"]) for _, dst, data in KG.out_edges(entity, data=True)]
    in_edges  = [(src, data["rel"]) for src, _, data in KG.in_edges(entity, data=True)]
    return json.dumps({
        "entity": entity,
        "attributes": attrs,
        "outgoing": out_edges,
        "incoming": in_edges,
    }, indent=2)


# ─── STATE + GRAPH ────────────────────────────────────────────────────────────

class KGState(TypedDict):
    question:  str
    messages:  list
    answer:    str | None
    iteration: int

KG_TOOLS = [kg_find_neighbors, kg_multihop_query, kg_filter_by_attribute, kg_get_node_info]
llm_kg   = ChatOllama(model=MODEL, temperature=0).bind_tools(KG_TOOLS)

from langgraph.prebuilt import ToolNode
kg_tool_node = ToolNode(KG_TOOLS)

def node_kg_reason(state: KGState) -> KGState:
    response = llm_kg.invoke(state["messages"])
    return {**state, "messages": state["messages"] + [response],
            "iteration": state.get("iteration", 0) + 1}

def node_kg_finish(state: KGState) -> KGState:
    # Use the last message that actually has text content as the answer.
    answer = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "content", "") and not getattr(msg, "tool_calls", None):
            answer = msg.content
            break
    return {**state, "answer": answer or "(no answer produced)"}

def route_kg(state: KGState) -> str:
    last = state["messages"][-1]
    # Stop looping if we've hit the cap (prevents runaway traversal on a small
    # model that keeps calling tools without concluding).
    if state.get("iteration", 0) >= MAX_HOPS:
        return "finish"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "finish"

def build_kg_graph() -> Any:
    g = StateGraph(KGState)
    g.add_node("reason", node_kg_reason)
    g.add_node("tools",  kg_tool_node)
    g.add_node("finish", node_kg_finish)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", route_kg, {"tools": "tools", "finish": "finish"})
    g.add_edge("tools", "reason")
    g.add_edge("finish", END)
    return g.compile()

def ask_kg(question: str):
    print(f"\n{'─' * 55}\nQuestion: {question}")
    graph = build_kg_graph()
    system = SystemMessage(content=(
        "You are a knowledge graph query agent. Use the tools to traverse the "
        "graph, then answer in plain language.\n\n"
        "GRAPH SCHEMA — entity types (examples):\n"
        "- Customer: Acme Corp, TechStart, GlobalBank, RetailCo\n"
        "- Product: AgentPro, DataSync, AutoFlow, InsightAI\n"
        "- Category: AI Agents, Data Integration, Automation\n"
        "- Supplier: VendorAlpha, VendorBeta\n"
        "- Regulation: EU AI Act, GDPR\n"
        "- Contract: C001 (expired), C002 (active)\n\n"
        "RELATIONSHIPS (arrow = direction):\n"
        "- Customer --purchased--> Product\n"
        "- Product --belongs_to--> Category\n"
        "- Category --regulated_by--> Regulation\n"
        "- Product --supplied_by--> Supplier\n"
        "- Supplier --has_contract--> Contract\n\n"
        "Use kg_multihop_query(start_entity, hop_pattern). Prefix a relationship "
        "with '<' to go BACKWARDS. Worked examples:\n"
        "- 'What regulations apply to AgentPro?' -> "
        "kg_multihop_query('AgentPro', 'belongs_to -> regulated_by')\n"
        "- 'Which customers purchased products regulated by the EU AI Act?' -> "
        "kg_multihop_query('EU AI Act', '<regulated_by -> <belongs_to -> <purchased')\n"
        "- 'Which products does VendorAlpha supply?' -> "
        "kg_multihop_query('VendorAlpha', '<supplied_by')\n"
        "After the tool returns the final nodes, state them as the answer."
    ))
    result = graph.invoke({
        "question":  question,
        "messages":  [system, HumanMessage(content=question)],
        "answer":    None,
        "iteration": 0,
    })
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    print("Knowledge graph nodes:", list(KG.nodes(data=False))[:8], "...")

    # These questions require multi-hop traversal — RAG cannot answer them
    ask_kg(
        "Which customers purchased products that are regulated by the EU AI Act?"
    )
    # VendorAlpha's contract C001 has expired. The risk path is:
    # VendorAlpha <-supplied_by- products <-purchased- customers.
    ask_kg(
        "VendorAlpha's contract has expired. Which customers bought products "
        "that are supplied by VendorAlpha, and are therefore exposed to this risk?"
    )
    ask_kg("What regulations apply to the AgentPro product?")