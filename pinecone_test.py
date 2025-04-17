"""
Test file to check proper Pinecone 6.0.2 import and usage
"""

# First, check what's installed
import pkg_resources
import sys

print("Python version:", sys.version)
print("Installed packages:")
for pkg in pkg_resources.working_set:
    if "pinecone" in pkg.key:
        print(f"  {pkg.key}=={pkg.version}")

# Try different import patterns
print("\nTrying different import patterns:")

try:
    print("\nPattern 1: 'import pinecone'")
    import pinecone

    print("  Success! Available attributes:")
    print("  ", dir(pinecone))
except Exception as e:
    print("  Error:", e)

try:
    print("\nPattern 2: 'from pinecone import Pinecone'")
    from pinecone import Pinecone

    print("  Success! Creating client with Pinecone()")
    pc = Pinecone(api_key="test_key")
    print("  Client methods:", dir(pc))
except Exception as e:
    print("  Error:", e)

try:
    print("\nPattern 3: Import from submodules")
    # This pattern is sometimes used in newer versions
    from pinecone.core import client

    print("  Success! Available in pinecone.core.client:", dir(client))
except Exception as e:
    print("  Error:", e)

print("\nTest completed")
