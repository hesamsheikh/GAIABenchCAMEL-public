"""Test script to verify toolkit functionality."""

from camel.toolkits import SearchToolkit

def test_search_wiki():
    """Test the search_wiki function."""
    print("=" * 60)
    print("Testing search_wiki")
    print("=" * 60)

    toolkit = SearchToolkit()

    # Test 1: Simple entity search
    print("\n1. Searching for 'Moon'...")
    result = toolkit.search_wiki(entity="Moon")
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result) if result else 0}")
    print(f"Result preview: {result[:500] if result else 'None'}...")

    # Test 2: Search for specific data
    print("\n2. Searching for 'Orbit of the Moon'...")
    result = toolkit.search_wiki(entity="Orbit of the Moon")
    print(f"Result length: {len(result) if result else 0}")
    print(f"Result preview: {result[:500] if result else 'None'}...")

    # Test 3: Search for a person
    print("\n3. Searching for 'Eliud Kipchoge'...")
    result = toolkit.search_wiki(entity="Eliud Kipchoge")
    print(f"Result length: {len(result) if result else 0}")
    print(f"Result preview: {result[:500] if result else 'None'}...")

    # Test 4: Check if we get world record info
    print("\n4. Checking if result contains '2:01' (world record time)...")
    if result and "2:01" in result:
        print("SUCCESS: Found world record time mention")
    else:
        print("FAIL: World record time not found in search_wiki result")

    # Test 5: Check if we get perigee info from Moon search
    print("\n5. Searching for 'Lunar distance'...")
    result = toolkit.search_wiki(entity="Lunar distance")
    print(f"Result length: {len(result) if result else 0}")
    if result and ("perigee" in result.lower() or "356" in result or "362" in result):
        print("SUCCESS: Found perigee-related info")
    else:
        print("FAIL: No perigee info in result")
    print(f"Result: {result[:800] if result else 'None'}...")


def test_search_duckduckgo():
    """Test the search_duckduckgo function."""
    print("\n" + "=" * 60)
    print("Testing search_duckduckgo")
    print("=" * 60)

    toolkit = SearchToolkit()

    print("\n1. Searching for 'Moon minimum perigee distance km'...")
    result = toolkit.search_duckduckgo(
        query="Moon minimum perigee distance km",
        source="text",
        number_of_result_pages=3
    )
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")


def check_tool_descriptions():
    """Check how tool descriptions appear to the agent."""
    print("\n" + "=" * 60)
    print("Checking tool descriptions")
    print("=" * 60)

    toolkit = SearchToolkit()

    # Get the search_wiki function tool
    import inspect

    print("\nsearch_wiki docstring:")
    print(toolkit.search_wiki.__doc__)

    print("\nsearch_duckduckgo docstring:")
    print(toolkit.search_duckduckgo.__doc__)


if __name__ == "__main__":
    test_search_wiki()
    test_search_duckduckgo()
    check_tool_descriptions()
