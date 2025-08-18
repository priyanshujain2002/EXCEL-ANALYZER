#!/usr/bin/env python3
"""
Demo script for Excel Query Agents
Shows how to use the query system with sample queries
"""

from excel_query_agents import ExcelQuerySystem
import os

def run_demo():
    """Run a demonstration of the Excel Query Agents system"""
    print("🚀 Excel Query Agents Demo")
    print("=" * 50)
    
    # Initialize the query system
    print("\n📊 Initializing query system...")
    query_system = ExcelQuerySystem()
    
    if not query_system.knowledge_base.knowledge_base:
        print("❌ No processed Excel files found.")
        print("💡 Please run excel_processor.py first to process some Excel files.")
        return
    
    # Show available data
    print("\n📋 Available datasets:")
    print(query_system.list_available_data())
    
    # Sample queries to demonstrate capabilities
    sample_queries = [
        "What data is available in the media plan?",
        "Show me information about high impact advertising",
        "What are the column names in the datasets?",
        "How many rows of data do we have?",
        "What kind of advertising data is available?",
        "Tell me about the Samsung Ads data structure"
    ]
    
    print("\n🔍 Running sample queries...")
    print("=" * 50)
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)
        
        try:
            result = query_system.query_data(query)
            print(f"📊 Result:\n{result}")
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
        
        print("\n" + "="*60)
    
    print("\n✅ Demo completed!")
    print("\n💡 To run interactive mode, use: python excel_query_agents.py")

if __name__ == "__main__":
    run_demo()