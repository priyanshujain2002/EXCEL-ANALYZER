#!/usr/bin/env python3
"""
Simple wrapper script to process multi-sheet Excel files.
This script provides an easy way to specify different input files.
"""

import sys
import os
from multi_sheet_excel_processor import process_multi_sheet_excel

def main():
    """Main function with user-friendly interface."""
    
    print("📊 Multi-Sheet Excel Processor")
    print("=" * 35)
    
    # Get input file
    if len(sys.argv) > 1:
        # File provided as command line argument
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_sheets"
        
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        
    else:
        # Interactive mode
        print("\n🔍 Enter the path to your Excel file:")
        print("Examples:")
        print("  - my_file.xlsx")
        print("  - /path/to/my_file.xlsx")
        print("  - C:\\Users\\username\\Documents\\file.xlsx")
        
        input_file = input("\n❓ Excel file path: ").strip()
        
        if not input_file:
            print("❌ No file specified!")
            return
        
        # Remove quotes if user added them
        input_file = input_file.strip('"').strip("'")
        
        # Ask for output directory
        output_dir = input("❓ Output directory (press Enter for 'processed_sheets'): ").strip()
        if not output_dir:
            output_dir = "processed_sheets"
    
    # Validate file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found!")
        print("Please check the file path and try again.")
        return
    
    # Check if it's an Excel file
    if not input_file.lower().endswith(('.xlsx', '.xls')):
        print(f"❌ Error: '{input_file}' doesn't appear to be an Excel file!")
        print("Please provide a .xlsx or .xls file.")
        return
    
    # Show what will happen
    print(f"\n🎯 Ready to process:")
    print(f"  📄 Input: {input_file}")
    print(f"  📁 Output: {output_dir}/")
    print(f"  🔄 Each sheet will be processed individually")
    
    # Confirm before processing
    confirm = input(f"\n❓ Continue? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 Processing cancelled.")
        return
    
    # Process the file
    try:
        process_multi_sheet_excel(input_file, output_dir)
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()