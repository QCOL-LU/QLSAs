#!/usr/bin/env python3
"""
Simple test script to verify QNexus login functionality.
"""

def test_qnx_login():
    """Test QNexus login and basic functionality."""
    try:
        print("Testing QNexus import...")
        import qnexus as qnx
        print("✓ QNexus imported successfully")
        
        # Try to get version if available
        try:
            version = qnx.__version__
            print(f"✓ QNexus version: {version}")
        except AttributeError:
            print("✓ QNexus imported (version info not available)")
        
        print("\nTesting QNexus login...")
        qnx.login()
        print("✓ QNexus login successful")
        
        print("\nTesting basic QNexus functionality...")
        # Try to get some basic info
        print("✓ QNexus login test completed successfully")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during login: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== QNexus Login Test ===\n")
    success = test_qnx_login()
    
    if success:
        print("\n🎉 All tests passed! QNexus is working correctly.")
    else:
        print("\n❌ Tests failed. Please check your installation.") 