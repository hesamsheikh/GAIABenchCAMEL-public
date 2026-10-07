"""
Test script to diagnose the document retriever functionality.

This tests the GAIARetriever to identify issues with document processing and retrieval.
"""

import os
import sys
from pathlib import Path

# Set up environment
from dotenv import load_dotenv
load_dotenv()

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from camel.embeddings import OpenAIEmbedding
from camel.types import StorageType
from src.gaia import GAIARetriever


def test_retriever_with_docx():
    """Test the retriever with the specific docx file that failed."""

    print("=" * 60)
    print("Testing GAIARetriever with DOCX file")
    print("=" * 60)

    # The file that failed
    task_id = "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb"
    file_path = Path("dataset/2023/validation") / f"{task_id}.docx"

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    print(f"File exists: {file_path} ({file_path.stat().st_size} bytes)")

    # Create a fresh test directory
    test_storage_dir = "execution/retriever_test/"
    Path(test_storage_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n1. Creating retriever...")
    try:
        retriever = GAIARetriever(
            vector_storage_local_path=test_storage_dir,
            storage_type=StorageType.QDRANT,
            embedding_model=OpenAIEmbedding(),
        )
        print("   Retriever created successfully")
    except Exception as e:
        print(f"   ERROR creating retriever: {e}")
        return False

    print(f"\n2. Resetting retriever for task {task_id}...")
    try:
        reset_result = retriever.reset(task_id=task_id)
        print(f"   Reset result: {reset_result}")
        print(f"   Storage path: {retriever.vector_storage_local_path}")
    except Exception as e:
        print(f"   ERROR resetting retriever: {e}")
        return False

    print(f"\n3. Testing document parsing directly with UnstructuredIO...")
    try:
        from camel.loaders import UnstructuredIO
        uio = UnstructuredIO()
        elements = uio.parse_file_or_url(str(file_path))
        print(f"   Parsed {len(elements)} elements from document")
        if elements:
            print(f"   First element type: {type(elements[0])}")
            print(f"   First element preview: {str(elements[0])[:200]}...")
    except Exception as e:
        print(f"   ERROR parsing document: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n4. Testing retrieval...")
    query = "An office held a Secret Santa gift exchange where each of its twelve employees was assigned one other employee in the group to present with a gift."

    try:
        result = retriever.retrieve(
            query=query,
            contents=[str(file_path)]
        )
        print(f"   Result keys: {result.keys()}")
        retrieved_context = result.get("Retrieved Context", [])
        print(f"   Retrieved {len(retrieved_context)} chunks")
        print(f"   Type of chunks: {type(retrieved_context)}")
        if retrieved_context:
            print(f"   Type of first chunk: {type(retrieved_context[0])}")

        if retrieved_context:
            print(f"\n   First chunk preview:")
            chunk = retrieved_context[0]
            if isinstance(chunk, dict):
                print(f"   {chunk.get('text', '')[:300]}...")
            else:
                print(f"   {str(chunk)[:300]}...")
            return True
        else:
            print(f"\n   WARNING: No content retrieved!")
            print(f"   Full result: {result}")
            return False

    except Exception as e:
        print(f"   ERROR during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retriever_with_txt():
    """Test with a simple text file to verify basic functionality."""

    print("\n" + "=" * 60)
    print("Testing GAIARetriever with a simple TXT file")
    print("=" * 60)

    # Create a test text file
    test_file = Path("execution/test_document.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    test_content = """
    Secret Santa Gift Exchange Results

    Employee Profiles:
    1. Alice - Likes: hiking, reading, coffee
    2. Bob - Likes: gaming, movies, pizza
    3. Carol - Likes: yoga, tea, gardening
    4. Dan - Likes: sports, beer, BBQ
    5. Eve - Likes: travel, photography, wine

    Gift Exchange:
    - Alice gave Bob a video game (gaming interest)
    - Bob gave Carol a yoga mat (yoga interest)
    - Carol gave Dan a sports jersey (sports interest)
    - Dan gave Eve a camera accessory (photography interest)
    - Eve gave Alice a hiking backpack (hiking interest)

    Fred did not give a gift because he was sick.
    """

    test_file.write_text(test_content)
    print(f"Created test file: {test_file}")

    test_storage_dir = "execution/retriever_test_txt/"
    Path(test_storage_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n1. Creating retriever...")
    try:
        retriever = GAIARetriever(
            vector_storage_local_path=test_storage_dir,
            storage_type=StorageType.QDRANT,
            embedding_model=OpenAIEmbedding(),
        )
        print("   Retriever created successfully")
    except Exception as e:
        print(f"   ERROR creating retriever: {e}")
        return False

    print(f"\n2. Resetting retriever...")
    try:
        reset_result = retriever.reset(task_id="test-txt-task")
        print(f"   Reset result: {reset_result}")
    except Exception as e:
        print(f"   ERROR resetting retriever: {e}")
        return False

    print(f"\n3. Testing retrieval...")
    query = "Who did not give a gift?"

    try:
        result = retriever.retrieve(
            query=query,
            contents=[str(test_file)]
        )
        retrieved_context = result.get("Retrieved Context", [])
        print(f"   Retrieved {len(retrieved_context)} chunks")

        if retrieved_context:
            for i, chunk in enumerate(retrieved_context):
                print(f"\n   Chunk {i+1}:")
                if isinstance(chunk, dict):
                    print(f"   Similarity: {chunk.get('similarity score', 'N/A')}")
                    print(f"   Text: {chunk.get('text', '')[:200]}...")
                else:
                    print(f"   Text: {str(chunk)[:200]}...")
            return True
        else:
            print(f"\n   WARNING: No content retrieved!")
            return False

    except Exception as e:
        print(f"   ERROR during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


def test_auto_retriever_directly():
    """Test AutoRetriever directly to isolate the issue."""

    print("\n" + "=" * 60)
    print("Testing AutoRetriever directly")
    print("=" * 60)

    from camel.retrievers import AutoRetriever

    task_id = "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb"
    file_path = Path("dataset/2023/validation") / f"{task_id}.docx"

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    test_storage_dir = f"execution/retriever_direct_test/{task_id}"
    Path(test_storage_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n1. Creating AutoRetriever...")
    try:
        retriever = AutoRetriever(
            vector_storage_local_path=test_storage_dir,
            storage_type=StorageType.QDRANT,
            embedding_model=OpenAIEmbedding(),
        )
        print("   AutoRetriever created successfully")
    except Exception as e:
        print(f"   ERROR creating retriever: {e}")
        return False

    print(f"\n2. Running vector retriever...")
    query = "Secret Santa gift exchange"

    try:
        result = retriever.run_vector_retriever(
            query=query,
            contents=[str(file_path)],
            similarity_threshold=0.3,
        )
        retrieved_context = result.get("Retrieved Context", [])
        print(f"   Retrieved {len(retrieved_context)} chunks")

        if retrieved_context:
            print(f"\n   SUCCESS! Content retrieved:")
            for i, chunk in enumerate(retrieved_context[:3]):
                print(f"\n   Chunk {i+1}:")
                if isinstance(chunk, dict):
                    print(f"   Similarity: {chunk.get('similarity score', 'N/A')}")
                    text = chunk.get('text', '')[:200]
                    print(f"   Text: {text}...")
                else:
                    print(f"   Text: {str(chunk)[:200]}...")
            return True
        else:
            print(f"\n   WARNING: No content retrieved!")
            print(f"   Full result: {result}")
            return False

    except Exception as e:
        print(f"   ERROR during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_docx_content():
    """Check if we can read the docx file content at all."""

    print("\n" + "=" * 60)
    print("Checking DOCX file content")
    print("=" * 60)

    task_id = "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb"
    file_path = Path("dataset/2023/validation") / f"{task_id}.docx"

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    print(f"\n1. Trying python-docx...")
    try:
        from docx import Document
        doc = Document(file_path)
        text_content = []
        for para in doc.paragraphs:
            text_content.append(para.text)
        full_text = "\n".join(text_content)
        print(f"   Extracted {len(full_text)} characters")
        print(f"   Preview: {full_text[:500]}...")
        return True
    except ImportError:
        print("   python-docx not installed")
    except Exception as e:
        print(f"   ERROR: {e}")

    print(f"\n2. Trying unstructured...")
    try:
        from unstructured.partition.docx import partition_docx
        elements = partition_docx(filename=str(file_path))
        print(f"   Extracted {len(elements)} elements")
        for i, elem in enumerate(elements[:5]):
            print(f"   Element {i+1}: {str(elem)[:100]}...")
        return True
    except ImportError:
        print("   unstructured not installed")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

    return False


def test_gaia_benchmark_integration():
    """Test the exact code path used in GAIABenchmark._prepare_task()."""

    print("\n" + "=" * 60)
    print("Testing GAIABenchmark integration")
    print("=" * 60)

    task_id = "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb"
    file_path = Path("dataset/2023/validation") / f"{task_id}.docx"

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    print(f"File exists: {file_path}")

    test_storage_dir = "execution/retriever_integration_test/"
    Path(test_storage_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n1. Creating retriever...")
    try:
        retriever = GAIARetriever(
            vector_storage_local_path=test_storage_dir,
            storage_type=StorageType.QDRANT,
            embedding_model=OpenAIEmbedding(),
        )
        print("   Retriever created successfully")
    except Exception as e:
        print(f"   ERROR creating retriever: {e}")
        return False

    print(f"\n2. Resetting retriever for task...")
    try:
        reset_result = retriever.reset(task_id=task_id)
        print(f"   Reset result: {reset_result}")
    except Exception as e:
        print(f"   ERROR resetting retriever: {e}")
        return False

    print(f"\n3. Testing retrieval with GAIABenchmark code pattern...")
    query = "An office held a Secret Santa gift exchange"

    try:
        retrieved_info = retriever.retrieve(
            query=query, contents=[str(file_path)]
        )

        # This is the EXACT code pattern from GAIABenchmark._prepare_task()
        # After the fix, it handles both dict and string formats
        retrieved_context = retrieved_info.get("Retrieved Context", [])
        retrieved_content = []
        for item in retrieved_context:
            if isinstance(item, dict):
                retrieved_content.append(item.get("text", ""))
            else:
                # Item is already a string
                retrieved_content.append(str(item))

        print(f"   Retrieved {len(retrieved_content)} content items")

        if retrieved_content:
            final_content = "\n".join(retrieved_content)
            print(f"\n   SUCCESS! Content length: {len(final_content)} characters")
            print(f"   Preview: {final_content[:300]}...")
            return True
        else:
            print(f"\n   WARNING: No content retrieved!")
            return False

    except Exception as e:
        print(f"   ERROR during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Document Retriever Test Suite")
    print("=" * 60)
    print()

    # Check if we have OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Embeddings will fail.")
        print("Please set OPENAI_API_KEY in .env file or environment.")
        sys.exit(1)

    results = {}

    # Test 1: Check docx content
    print("\n" + "=" * 60)
    print("TEST 1: Check DOCX content readability")
    print("=" * 60)
    results["docx_content"] = check_docx_content()

    # Test 2: Simple txt file
    print("\n" + "=" * 60)
    print("TEST 2: Simple TXT file retrieval")
    print("=" * 60)
    results["txt_retrieval"] = test_retriever_with_txt()

    # Test 3: Direct AutoRetriever test
    print("\n" + "=" * 60)
    print("TEST 3: Direct AutoRetriever test with DOCX")
    print("=" * 60)
    results["auto_retriever"] = test_auto_retriever_directly()

    # Test 4: GAIARetriever with docx
    print("\n" + "=" * 60)
    print("TEST 4: GAIARetriever with DOCX")
    print("=" * 60)
    results["gaia_retriever"] = test_retriever_with_docx()

    # Test 5: GAIABenchmark integration test
    print("\n" + "=" * 60)
    print("TEST 5: GAIABenchmark integration")
    print("=" * 60)
    results["gaia_benchmark_integration"] = test_gaia_benchmark_integration()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
