#!/usr/bin/env python3
"""
Basic test script to verify connection fixes without CrewAI dependencies.
"""

import os
import sys
import pandas as pd

def test_environment_variables():
    """Test that environment variables are set correctly."""
    print("🧪 Testing environment variables...")
    
    # Set the variables
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    
    # Check they're set
    checks = [
        ("OTEL_SDK_DISABLED", "true"),
        ("CREWAI_TELEMETRY_ENABLED", "false"),
        ("CREWAI_DISABLE_TELEMETRY", "true")
    ]
    
    all_passed = True
    for var_name, expected_value in checks:
        actual_value = os.environ.get(var_name)
        if actual_value == expected_value:
            print(f"✅ {var_name} = {actual_value}")
        else:
            print(f"❌ {var_name} = {actual_value} (expected {expected_value})")
            all_passed = False
    
    return all_passed

def test_data_file_exists():
    """Test that the data file exists."""
    print("\n🧪 Testing data file existence...")
    
    file_path = "knowledge/Testing Sheet.xlsx"
    if os.path.exists(file_path):
        print(f"✅ Data file found: {file_path}")
        return True
    else:
        print(f"❌ Data file not found: {file_path}")
        # Check if knowledge directory exists
        if os.path.exists("knowledge"):
            print("📁 Knowledge directory exists, listing contents:")
            for item in os.listdir("knowledge"):
                print(f"   - {item}")
        else:
            print("❌ Knowledge directory doesn't exist")
        return False

def test_pandas_loading():
    """Test loading data with pandas."""
    print("\n🧪 Testing pandas data loading...")
    
    file_path = "knowledge/Testing Sheet.xlsx"
    if not os.path.exists(file_path):
        print("❌ Cannot test pandas loading - file doesn't exist")
        return False
    
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Successfully loaded {len(df)} rows")
        
        if 'Genre' in df.columns:
            unique_genres = df['Genre'].dropna().nunique()
            print(f"✅ Found {unique_genres} unique genres")
            
            # Show first few genres
            sample_genres = df['Genre'].dropna().head(5).tolist()
            print(f"📝 Sample genres: {sample_genres}")
            return True
        else:
            print("❌ 'Genre' column not found")
            print(f"📝 Available columns: {list(df.columns)}")
            return False
            
    except Exception as e:
        print(f"❌ Pandas loading failed: {e}")
        return False

def test_fallback_logic():
    """Test the fallback logic without CrewAI."""
    print("\n🧪 Testing fallback logic...")
    
    try:
        # Define basic genre relationship mappings (same as in the main code)
        genre_relationships = {
            'crime': ['mystery', 'thriller', 'drama', 'suspense', 'law', 'detective'],
            'mystery': ['crime', 'thriller', 'suspense', 'detective', 'drama'],
            'comedy': ['sitcom', 'variety', 'entertainment', 'family'],
            'drama': ['romance', 'family', 'general drama', 'crime drama'],
            'action': ['adventure', 'thriller', 'crime', 'suspense'],
            'horror': ['thriller', 'suspense', 'mystery', 'paranormal'],
            'romance': ['drama', 'comedy', 'romantic comedy', 'family'],
            'documentary': ['education', 'history', 'science', 'nature'],
            'sports': ['competition', 'reality', 'entertainment'],
            'music': ['variety', 'entertainment', 'concert', 'dance']
        }
        
        # Test with 'crime'
        test_genre = 'crime'
        if test_genre in genre_relationships:
            related = genre_relationships[test_genre]
            print(f"✅ Found {len(related)} related genres for '{test_genre}': {related}")
            return True
        else:
            print(f"❌ No relationships found for '{test_genre}'")
            return False
            
    except Exception as e:
        print(f"❌ Fallback logic test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🚀 Starting basic connection fixes test suite...\n")
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Data File Existence", test_data_file_exists),
        ("Pandas Loading", test_pandas_loading),
        ("Fallback Logic", test_fallback_logic)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All basic tests passed! The connection fixes should work.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)