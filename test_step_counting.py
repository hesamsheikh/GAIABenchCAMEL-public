"""Test that message_count correctly reflects actual messages in workforce."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_message_counting():
    from main_workforce import create_gaia_workforce, WorkforceWrapper
    from camel.messages import BaseMessage
    
    print("Creating workforce...")
    workforce = await create_gaia_workforce()
    wrapper = WorkforceWrapper(workforce)
    
    test_message = BaseMessage.make_user_message(
        role_name="User",
        content="What is 2 + 2? Just give me the number."
    )
    
    print("Running test task...")
    result = await wrapper.astep(test_message)
    
    detailed_history = result.info.get("detailed_history", [])
    
    # Count by role
    from collections import Counter
    role_counts = Counter(msg.get('role', 'unknown') for msg in detailed_history)
    print(f"\n=== MESSAGE COUNTS BY ROLE ===")
    for role, count in role_counts.items():
        print(f"  {role}: {count}")
    print(f"  TOTAL: {len(detailed_history)}")
    
    # Count by worker
    worker_counts = Counter(msg.get('worker_description', 'unknown')[:50] for msg in detailed_history)
    print(f"\n=== MESSAGE COUNTS BY WORKER ===")
    for worker, count in worker_counts.items():
        print(f"  {worker}: {count}")
        
    # The actual "steps" would be user+assistant pairs, not system messages
    non_system = [m for m in detailed_history if m.get('role') != 'system']
    print(f"\n=== NON-SYSTEM MESSAGES: {len(non_system)} ===")
    for msg in non_system:
        role = msg.get('role')
        worker = msg.get('worker_description', '')[:30]
        content = str(msg.get('content', ''))[:60].replace('\n', ' ')
        print(f"  [{role}] {worker}: {content}...")

if __name__ == "__main__":
    asyncio.run(test_message_counting())
