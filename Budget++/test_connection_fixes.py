#!/usr/bin/env python3
"""
Test script to verify connection fixes for the genre recommender.
"""

import os
import sys
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    try:
        # Disable telemetry before importing
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        
        from genre_recommender import load_and_preview_data, validate_genre_exists
        print("✅ Imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n🧪 Testing data loading...")
    try:
        from genre_recommender import load_and_preview_data
        df = load_and_preview_data()
        if df is not None:
            print("✅ Data loading successful")
            return True
        else:
            print("❌ Data loading returned None")
            return False
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_genre_validation():
    """Test genre validation functionality."""
    print("\n🧪 Testing genre validation...")
    try:
        from genre_recommender import load_and_preview_data, validate_genre_exists
        df = load_and_preview_data()
        if df is not None:
            # Test with a known genre
            exists, similar = validate_genre_exists("crime", df)
            if exists:
                print("✅ Genre validation successful")
                return True
            else:
                print(f"⚠️  'crime' not found, but found similar: {similar}")
                return len(similar) > 0
        return False
    except Exception as e:
        print(f"❌ Genre validation failed: {e}")
        return False

def test_fallback_response():
    """Test fallback response generation."""
    print("\n🧪 Testing fallback response...")
    try:
        from genre_recommender import generate_fallback_response
        response = generate_fallback_response("crime")
        if response and len(response) > 50:  # Basic check for meaningful response
            print("✅ Fallback response generation successful")
            print(f"📝 Sample response (first 100 chars): {response[:100]}...")
            return True
        else:
            print("❌ Fallback response too short or empty")
            return False
    except Exception as e:
        print(f"❌ Fallback response failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting connection fixes test suite...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Genre Validation", test_genre_validation),
        ("Fallback Response", test_fallback_response)
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
        print("🎉 All tests passed! The connection fixes should work.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)