"""
Jira MCP Server — cho phép Cursor/Claude Desktop query Jira tickets
mà không cần nhúng Jira logic vào Streamlit app.

Usage:
    python jira_mcp_server.py

Yêu cầu:
    pip install mcp requests

Cấu hình:
    Set JIRA_URL và JIRA_API_TOKEN trong .env hoặc environment variables.
"""

import json
import os
import requests
import urllib3
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Load .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

JIRA_URL   = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN", "").strip()

app = Server("jira-mcp")


@app.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="search_jira",
            description="Tìm kiếm Jira tickets theo JQL query. "
                        "Trả về danh sách tickets với key, summary, description, status, market.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jql": {
                        "type": "string",
                        "description": "JQL query string. Ví dụ: 'project = VF6 AND status != Closed'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Số kết quả tối đa (default: 50)",
                        "default": 50
                    }
                },
                "required": ["jql"]
            }
        ),
        types.Tool(
            name="get_ticket",
            description="Lấy chi tiết một Jira ticket theo key (ví dụ: VF6-12345). "
                        "Trả về toàn bộ fields của ticket.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Jira issue key (ví dụ: VF6-12345)"
                    }
                },
                "required": ["key"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if not JIRA_URL or not JIRA_TOKEN:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error": "JIRA_URL và JIRA_API_TOKEN chưa được cấu hình trong .env"
            }, ensure_ascii=False)
        )]

    headers = {
        "Authorization": f"Bearer {JIRA_TOKEN}",
        "Accept": "application/json"
    }

    if name == "search_jira":
        params = {
            "jql": arguments["jql"],
            "maxResults": arguments.get("max_results", 50),
            "fields": "key,summary,description,status,customfield_14506,customfield_12101,customfield_market"
        }
        try:
            resp = requests.get(
                f"{JIRA_URL}/rest/api/2/search",
                params=params,
                headers=headers,
                verify=False,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            # Rút gọn response — chỉ giữ fields cần thiết
            simplified = []
            for issue in data.get("issues", []):
                f = issue.get("fields", {})
                market_raw = (
                    f.get("customfield_14506") or f.get("customfield_12101") or
                    f.get("customfield_market") or ""
                )
                if isinstance(market_raw, dict):
                    market_raw = market_raw.get("value", "")
                elif isinstance(market_raw, list):
                    market_raw = ", ".join([
                        m.get("value", m) if isinstance(m, dict) else str(m)
                        for m in market_raw
                    ])

                simplified.append({
                    "key": issue.get("key"),
                    "summary": f.get("summary"),
                    "description": (f.get("description") or "")[:500],
                    "status": f.get("status", {}).get("name"),
                    "market": str(market_raw or ""),
                })

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "total": data.get("total", 0),
                    "issues": simplified
                }, ensure_ascii=False, indent=2)
            )]
        except requests.exceptions.RequestException as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, ensure_ascii=False)
            )]

    elif name == "get_ticket":
        try:
            resp = requests.get(
                f"{JIRA_URL}/rest/api/2/issue/{arguments['key']}",
                headers=headers,
                verify=False,
                timeout=30
            )
            resp.raise_for_status()
            return [types.TextContent(
                type="text",
                text=json.dumps(resp.json(), ensure_ascii=False, indent=2)
            )]
        except requests.exceptions.RequestException as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, ensure_ascii=False)
            )]

    return [types.TextContent(
        type="text",
        text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
    )]


if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(app))