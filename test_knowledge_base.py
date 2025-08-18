#!/usr/bin/env python3
"""
Simple test script to verify the knowledge base loading functionality
"""

from excel_query_agents import ExcelKnowledgeBase
import pandas as pd

def test_knowledge_base():
    """Test the knowledge base loading functionality"""
    print("🧪 Testing Excel Knowledge Base")
    print("=" * 40)
    
    # Initialize knowledge base
    kb = ExcelKnowledgeBase()
    
    print(f"\n📊 Loaded {len(kb.knowledge_base)} datasets")
    
    # Show details of each dataset
    for key, data in kb.knowledge_base.items():
        print(f"\n🔹 Dataset: {key}")
        print(f"   Project: {data['project_name']}")
        print(f"   Sheet: {data['sheet_name']}")
        print(f"   Shape: {data['row_count']} rows × {data['column_count']} columns")
        print(f"   Columns: {data['columns'][:5]}{'...' if len(data['columns']) > 5 else ''}")
        
        if data['summary']['sample_data']:
            print(f"   Sample data (first row): {data['summary']['sample_data'][0]}")
    
    # Test search functionality
    print(f"\n🔍 Testing search functionality...")
    
    test_queries = [
        "media plan",
        "advertising",
        "Samsung",
        "data",
        "columns"
    ]
    
    for query in test_queries:
        results = kb.search_data(query)
        print(f"\nQuery: '{query}' -> Found {len(results)} relevant datasets")
        for result in results[:2]:  # Show top 2 results
            print(f"  - {result['data']['project_name']} - {result['data']['sheet_name']} (score: {result['relevance_score']})")
    
    print(f"\n✅ Knowledge base test completed!")

if __name__ == "__main__":
    test_knowledge_base()