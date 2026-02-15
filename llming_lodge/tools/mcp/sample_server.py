#!/usr/bin/env python3
"""
Sample MCP Server for testing llming-lodge's MCP integration.

Provides fake database tools for:
- Searching products
- Getting product details
- Listing categories
- Customer lookup

Usage:
    python -m llming_lodge.tools.mcp.sample_server
"""

import asyncio
import json
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


# Fake database
PRODUCTS_DB = {
    "P001": {"id": "P001", "name": "Wireless Headphones", "category": "Electronics", "price": 79.99, "stock": 150},
    "P002": {"id": "P002", "name": "USB-C Cable", "category": "Electronics", "price": 12.99, "stock": 500},
    "P003": {"id": "P003", "name": "Laptop Stand", "category": "Office", "price": 45.00, "stock": 75},
    "P004": {"id": "P004", "name": "Mechanical Keyboard", "category": "Electronics", "price": 129.99, "stock": 30},
    "P005": {"id": "P005", "name": "Desk Lamp", "category": "Office", "price": 35.00, "stock": 200},
    "P006": {"id": "P006", "name": "Coffee Mug", "category": "Kitchen", "price": 15.00, "stock": 1000},
    "P007": {"id": "P007", "name": "Water Bottle", "category": "Kitchen", "price": 25.00, "stock": 300},
    "P008": {"id": "P008", "name": "Notebook Set", "category": "Office", "price": 18.00, "stock": 450},
}

CUSTOMERS_DB = {
    "C001": {"id": "C001", "name": "John Smith", "email": "john@example.com", "tier": "Gold", "total_orders": 15},
    "C002": {"id": "C002", "name": "Jane Doe", "email": "jane@example.com", "tier": "Silver", "total_orders": 8},
    "C003": {"id": "C003", "name": "Bob Wilson", "email": "bob@example.com", "tier": "Bronze", "total_orders": 3},
}

CATEGORIES = ["Electronics", "Office", "Kitchen"]


# Create MCP server
server = Server(
    "sample-db-server",
    instructions="""Sample Database MCP Server for testing.

Provides fake database tools for product and customer lookup.
Use these tools to test MCP integration in llming-lodge.
"""
)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_products",
            description="Search products by name or category. Returns matching products with basic info.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (matches product name or category)",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category: Electronics, Office, Kitchen",
                        "enum": CATEGORIES + [""],
                        "default": "",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_product_details",
            description="Get detailed information about a specific product by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID (e.g., 'P001')",
                    },
                },
                "required": ["product_id"],
            },
        ),
        Tool(
            name="list_categories",
            description="List all available product categories with product counts.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_customer",
            description="Look up customer information by ID or email. Provide either customer_id or email.",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID (e.g., 'C001')",
                        "default": "",
                    },
                    "email": {
                        "type": "string",
                        "description": "Customer email address",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="check_inventory",
            description="Check inventory levels for products. Can filter by low stock or category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "low_stock_only": {
                        "type": "boolean",
                        "description": "Only show products with stock < 100",
                        "default": False,
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category: Electronics, Office, Kitchen",
                        "enum": CATEGORIES + [""],
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "search_products":
        query = arguments.get("query", "").lower()
        category = arguments.get("category", "")
        max_results = arguments.get("max_results", 5)

        results = []
        for product in PRODUCTS_DB.values():
            # Match query against name or category
            if query in product["name"].lower() or query in product["category"].lower():
                if not category or product["category"] == category:
                    results.append(product)

        results = results[:max_results]

        if not results:
            return [TextContent(type="text", text=f"No products found matching '{query}'")]

        lines = [f"Found {len(results)} product(s):\n"]
        for p in results:
            lines.append(f"- {p['id']}: {p['name']} (${p['price']:.2f}) - {p['category']}")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "get_product_details":
        product_id = arguments.get("product_id", "")

        if product_id not in PRODUCTS_DB:
            return [TextContent(type="text", text=f"Product '{product_id}' not found")]

        p = PRODUCTS_DB[product_id]
        details = f"""Product Details:
- ID: {p['id']}
- Name: {p['name']}
- Category: {p['category']}
- Price: ${p['price']:.2f}
- Stock: {p['stock']} units
- Status: {'In Stock' if p['stock'] > 0 else 'Out of Stock'}"""

        return [TextContent(type="text", text=details)]

    elif name == "list_categories":
        lines = ["Product Categories:\n"]
        for cat in CATEGORIES:
            count = sum(1 for p in PRODUCTS_DB.values() if p["category"] == cat)
            lines.append(f"- {cat}: {count} products")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "get_customer":
        customer_id = arguments.get("customer_id", "")
        email = arguments.get("email", "")

        customer = None
        if customer_id and customer_id in CUSTOMERS_DB:
            customer = CUSTOMERS_DB[customer_id]
        elif email:
            customer = next((c for c in CUSTOMERS_DB.values() if c["email"] == email), None)

        if not customer:
            return [TextContent(type="text", text="Customer not found")]

        details = f"""Customer Details:
- ID: {customer['id']}
- Name: {customer['name']}
- Email: {customer['email']}
- Tier: {customer['tier']}
- Total Orders: {customer['total_orders']}"""

        return [TextContent(type="text", text=details)]

    elif name == "check_inventory":
        low_stock_only = arguments.get("low_stock_only", False)
        category = arguments.get("category", "")

        products = list(PRODUCTS_DB.values())

        if category:
            products = [p for p in products if p["category"] == category]

        if low_stock_only:
            products = [p for p in products if p["stock"] < 100]

        if not products:
            return [TextContent(type="text", text="No products match the criteria")]

        lines = ["Inventory Report:\n"]
        for p in sorted(products, key=lambda x: x["stock"]):
            status = "⚠️ LOW" if p["stock"] < 100 else "✓ OK"
            lines.append(f"- {p['id']} {p['name']}: {p['stock']} units {status}")

        return [TextContent(type="text", text="\n".join(lines))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    import sys
    print("Sample DB MCP Server starting...", file=sys.stderr)
    print(f"Products: {len(PRODUCTS_DB)}, Customers: {len(CUSTOMERS_DB)}", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
