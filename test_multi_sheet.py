#!/usr/bin/env python3

import os
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

# Test script to verify multi-sheet processing
input_file = os.path.abspath("NBCE_The Voice S28_Fall_25_RFP Template (Samsung Ads) 5.13.xlsx")

server_params = StdioServerParameters(
    command="python3",
    args=["-m", "excel_mcp", "stdio"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

print("Testing multi-sheet Excel file processing...")
print(f"Input file: {input_file}")

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")
    
    # Test getting workbook metadata
    try:
        for tool in mcp_tools:
            if tool.name == "get_workbook_metadata":
                print("\nTesting get_workbook_metadata...")
                result = tool.invoke({"file_path": input_file})
                print(f"Metadata result: {result}")
                break
    except Exception as e:
        print(f"Error getting metadata: {e}")

print("Test completed.")