# Excel Query Agents System

This system creates intelligent AI agents that can answer queries based on processed Excel data from `excel_processor.py`. The agents use the cleaned Excel files as a knowledge source to provide insights and answer questions about your data.

## Overview

The Excel Query Agents system consists of three specialized AI agents:

1. **Query Interpreter Agent** - Understands natural language questions and identifies relevant data sources
2. **Data Analyst Agent** - Analyzes Excel data and provides insights based on user queries  
3. **Report Generator Agent** - Creates comprehensive reports and summaries from analysis results

## Files

- `excel_query_agents.py` - Main query system with interactive mode
- `demo_query_agents.py` - Demonstration script with sample queries
- `README_query_agents.md` - This documentation file

## Prerequisites

1. **Processed Excel Files**: Run `excel_processor.py` first to process your Excel files
2. **Dependencies**: Same as `excel_processor.py` (CrewAI, pandas, etc.)
3. **AWS Credentials**: Configured for Bedrock Claude access

## Usage

### Interactive Mode

Run the main script for an interactive query session:

```bash
python excel_query_agents.py
```

Available commands:
- `list` - Show all available datasets
- `details [project] [sheet]` - Show details for specific data
- `query <your question>` - Ask a question about the data
- `quit` - Exit the system

### Demo Mode

Run the demo to see sample queries:

```bash
python demo_query_agents.py
```

### Programmatic Usage

```python
from excel_query_agents import ExcelQuerySystem

# Initialize the system
query_system = ExcelQuerySystem()

# Ask questions about your data
result = query_system.query_data("What data is available in the media plan?")
print(result)

# List available datasets
datasets = query_system.list_available_data()
print(datasets)

# Get detailed information about specific data
details = query_system.get_data_details(project_name="Samsung", sheet_name="Media")
print(details)
```

## Sample Queries

The system can handle various types of questions:

### Data Structure Questions
- "What columns are available in the datasets?"
- "How many rows of data do we have?"
- "What is the structure of the media plan data?"

### Content Questions  
- "What data is available in the media plan?"
- "Show me information about high impact advertising"
- "What kind of advertising data is available?"

### Analysis Questions
- "What are the key metrics in the Samsung Ads data?"
- "Compare the different advertising formats"
- "What insights can you provide about the media planning data?"

## How It Works

1. **Knowledge Base Loading**: The system automatically loads all processed Excel files from the `processed_files/` directory
2. **Query Processing**: When you ask a question, the system:
   - Interprets your natural language query
   - Searches for relevant datasets
   - Analyzes the data to find answers
   - Generates a comprehensive response
3. **Multi-Agent Collaboration**: The three agents work together sequentially to provide accurate, well-structured answers

## Data Sources

The system uses the output from `excel_processor.py`:
- Location: `processed_files/` directory
- Format: Cleaned Excel files with naming pattern `cleaned_[ProjectName]_[SheetName].xlsx`
- Structure: Each file contains tabular data extracted from original Excel sheets

## Features

### Intelligent Data Discovery
- Automatically discovers and indexes all processed Excel files
- Creates searchable metadata for quick data retrieval
- Provides data summaries and statistics

### Natural Language Processing
- Understands questions in plain English
- Identifies relevant datasets based on query content
- Handles various question types and formats

### Comprehensive Analysis
- Performs data analysis based on query requirements
- Provides statistical insights when relevant
- Offers actionable recommendations

### Structured Reporting
- Generates clear, well-formatted responses
- Includes supporting data and evidence
- Mentions data sources and limitations

## Error Handling

The system includes robust error handling:
- Graceful handling of corrupted Excel files
- Clear error messages for missing data
- Fallback options when specific data isn't found

## Extending the System

### Adding New Agent Types
You can extend the system by adding specialized agents:

```python
# Example: Financial Analysis Agent
financial_agent = Agent(
    role='Financial Analyst',
    goal='Analyze financial data and provide budget insights',
    backstory='Expert in financial analysis and budget planning...',
    llm=self.llm
)
```

### Custom Query Types
Add custom query processing for specific domains:

```python
def process_financial_query(self, query: str):
    # Custom logic for financial queries
    pass
```

### Integration with Other Tools
The system can be integrated with:
- Business intelligence tools
- Reporting dashboards  
- Data visualization platforms
- API endpoints for web applications

## Troubleshooting

### Common Issues

1. **No data found**: Ensure `excel_processor.py` has been run and created files in `processed_files/`
2. **AWS errors**: Check your AWS credentials and Bedrock access
3. **Memory issues**: Large datasets may require system optimization
4. **Query not understood**: Try rephrasing questions more specifically

### Performance Tips

- Use specific project/sheet filters for large datasets
- Ask focused questions rather than very broad queries
- Break complex questions into smaller parts

## Future Enhancements

Potential improvements:
- Web interface for easier interaction
- Integration with visualization libraries
- Support for real-time data updates
- Advanced analytics and machine learning insights
- Export capabilities for reports and analysis