#!/usr/bin/env python3
"""
Analyze and compare GAIA benchmark results between baseline and with-memory setups.

Usage:
    python analyze_results.py
    python analyze_results.py --baseline task_results/baseline --memory task_results_with_memory
    python analyze_results.py --include-timeouts  # Include timed-out tasks in comparison
"""

import json
import os
import argparse


def is_timeout(task_data):
    """Check if a task result is a timeout."""
    error = task_data.get('error', '')
    if error and 'timed out' in str(error).lower():
        return True
    return False


def analyze_folder(folder_path):
    """Load all task results from a folder."""
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                task_id = data.get('task_id')
                results[task_id] = data
    return results


def filter_timeouts(baseline, with_memory):
    """Remove tasks that timed out in either baseline or with_memory."""
    excluded = set()
    for task_id in set(baseline.keys()) | set(with_memory.keys()):
        b_timeout = task_id in baseline and is_timeout(baseline[task_id])
        m_timeout = task_id in with_memory and is_timeout(with_memory[task_id])
        if b_timeout or m_timeout:
            excluded.add(task_id)

    filtered_baseline = {k: v for k, v in baseline.items() if k not in excluded}
    filtered_memory = {k: v for k, v in with_memory.items() if k not in excluded}

    return filtered_baseline, filtered_memory, excluded


def compute_stats(results):
    """Compute accuracy statistics from results."""
    total = len(results)
    correct = sum(1 for r in results.values() if r.get('score') == 1)
    accuracy = correct / total * 100 if total > 0 else 0

    # Compute message and tool call stats
    total_messages = 0
    total_tool_calls = 0
    for r in results.values():
        total_messages += r.get('message_count', 0)
        total_tool_calls += len(r.get('tool_calls', []))

    avg_messages = total_messages / total if total > 0 else 0
    avg_tool_calls = total_tool_calls / total if total > 0 else 0

    # By level
    by_level = {}
    for r in results.values():
        level = r.get('level', 'unknown')
        if level not in by_level:
            by_level[level] = {'total': 0, 'correct': 0}
        by_level[level]['total'] += 1
        if r.get('score') == 1:
            by_level[level]['correct'] += 1

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'by_level': by_level,
        'total_messages': total_messages,
        'total_tool_calls': total_tool_calls,
        'avg_messages': avg_messages,
        'avg_tool_calls': avg_tool_calls,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze GAIA benchmark results')
    parser.add_argument('--baseline', default='task_results/baseline',
                        help='Path to baseline results folder')
    parser.add_argument('--memory', default='task_results_with_memory',
                        help='Path to with-memory results folder')
    parser.add_argument('--include-timeouts', action='store_true',
                        help='Include timed-out tasks in comparison (default: exclude them)')
    args = parser.parse_args()

    # Load results
    baseline_raw = analyze_folder(args.baseline)
    with_memory_raw = analyze_folder(args.memory)

    # Filter out timeouts by default
    if args.include_timeouts:
        baseline = baseline_raw
        with_memory = with_memory_raw
        excluded_timeouts = set()
    else:
        baseline, with_memory, excluded_timeouts = filter_timeouts(baseline_raw, with_memory_raw)

    if excluded_timeouts:
        print(f'Note: Excluding {len(excluded_timeouts)} timed-out tasks from comparison.')
        print(f'      Use --include-timeouts to include them.')
        print()

    b_stats = compute_stats(baseline)
    m_stats = compute_stats(with_memory)

    # Overall comparison
    print('=' * 65)
    print('           GAIA BENCHMARK ACCURACY COMPARISON')
    print('=' * 65)
    print()
    print(f"{'Setup':<20} {'Tasks':>8} {'Correct':>10} {'Accuracy':>12}")
    print('-' * 50)
    print(f"{'Baseline':<20} {b_stats['total']:>8} {b_stats['correct']:>10} {b_stats['accuracy']:>11.1f}%")
    print(f"{'With Memory':<20} {m_stats['total']:>8} {m_stats['correct']:>10} {m_stats['accuracy']:>11.1f}%")
    print()
    diff = m_stats['accuracy'] - b_stats['accuracy']
    print(f"Difference: {diff:+.1f} percentage points")
    print()

    # Message and tool call comparison
    print('=' * 65)
    print('           MESSAGES & TOOL CALLS')
    print('=' * 65)
    print()
    print(f"{'Setup':<20} {'Total Msgs':>12} {'Avg Msgs':>10} {'Total Tools':>13} {'Avg Tools':>11}")
    print('-' * 66)
    print(f"{'Baseline':<20} {b_stats['total_messages']:>12} {b_stats['avg_messages']:>10.1f} {b_stats['total_tool_calls']:>13} {b_stats['avg_tool_calls']:>11.1f}")
    print(f"{'With Memory':<20} {m_stats['total_messages']:>12} {m_stats['avg_messages']:>10.1f} {m_stats['total_tool_calls']:>13} {m_stats['avg_tool_calls']:>11.1f}")
    print()
    msg_diff = m_stats['avg_messages'] - b_stats['avg_messages']
    tool_diff = m_stats['avg_tool_calls'] - b_stats['avg_tool_calls']
    print(f"Avg messages diff:   {msg_diff:+.1f}")
    print(f"Avg tool calls diff: {tool_diff:+.1f}")
    print()

    # By level comparison
    print('=' * 65)
    print('           BY DIFFICULTY LEVEL')
    print('=' * 65)
    print()
    print(f"{'Level':<10} {'Baseline':>20} {'With Memory':>20} {'Diff':>10}")
    print('-' * 60)

    all_levels = sorted(set(b_stats['by_level'].keys()) | set(m_stats['by_level'].keys()))
    for level in all_levels:
        b = b_stats['by_level'].get(level, {'total': 0, 'correct': 0})
        m = m_stats['by_level'].get(level, {'total': 0, 'correct': 0})

        b_acc = b['correct'] / b['total'] * 100 if b['total'] > 0 else 0
        m_acc = m['correct'] / m['total'] * 100 if m['total'] > 0 else 0

        b_str = f"{b['correct']}/{b['total']} ({b_acc:.1f}%)"
        m_str = f"{m['correct']}/{m['total']} ({m_acc:.1f}%)"
        level_diff = m_acc - b_acc

        print(f"Level {level:<4} {b_str:>20} {m_str:>20} {level_diff:>+9.1f}%")
    print()

    # Task overlap analysis
    baseline_ids = set(baseline.keys())
    memory_ids = set(with_memory.keys())
    common_ids = baseline_ids & memory_ids

    print('=' * 65)
    print('           TASK OVERLAP ANALYSIS')
    print('=' * 65)
    print(f'Baseline tasks: {len(baseline_ids)}')
    print(f'With Memory tasks: {len(memory_ids)}')
    print(f'Common tasks: {len(common_ids)}')
    print(f'Only in Baseline: {len(baseline_ids - memory_ids)}')
    print(f'Only in With Memory: {len(memory_ids - baseline_ids)}')
    print()

    # Head-to-head comparison
    if common_ids:
        print('=' * 65)
        print('           HEAD-TO-HEAD COMPARISON (Same Tasks)')
        print('=' * 65)

        both_correct = 0
        baseline_only = 0
        memory_only = 0
        both_wrong = 0

        for task_id in common_ids:
            b_correct = baseline[task_id].get('score', 0) == 1
            m_correct = with_memory[task_id].get('score', 0) == 1

            if b_correct and m_correct:
                both_correct += 1
            elif b_correct and not m_correct:
                baseline_only += 1
            elif not b_correct and m_correct:
                memory_only += 1
            else:
                both_wrong += 1

        print(f'Both correct:                {both_correct:>5} tasks')
        print(f'Both wrong:                  {both_wrong:>5} tasks')
        print(f'Baseline correct only:       {baseline_only:>5} tasks (regression with memory)')
        print(f'With Memory correct only:    {memory_only:>5} tasks (improvement with memory)')
        print()

        # Head-to-head by level
        print('=' * 65)
        print('           HEAD-TO-HEAD BY LEVEL')
        print('=' * 65)

        by_level = {}
        for task_id in common_ids:
            level = baseline[task_id].get('level')
            if level not in by_level:
                by_level[level] = {'both_correct': 0, 'both_wrong': 0, 'b_only': 0, 'm_only': 0, 'total': 0}

            by_level[level]['total'] += 1
            b_score = baseline[task_id].get('score', 0) == 1
            m_score = with_memory[task_id].get('score', 0) == 1

            if b_score and m_score:
                by_level[level]['both_correct'] += 1
            elif b_score:
                by_level[level]['b_only'] += 1
            elif m_score:
                by_level[level]['m_only'] += 1
            else:
                by_level[level]['both_wrong'] += 1

        print(f"{'Level':<8} {'Total':>6} {'Both✓':>8} {'Both✗':>8} {'Base only':>10} {'Mem only':>10}")
        print('-' * 60)
        for level in sorted(by_level.keys()):
            d = by_level[level]
            print(f"Level {level:<2} {d['total']:>6} {d['both_correct']:>8} {d['both_wrong']:>8} {d['b_only']:>10} {d['m_only']:>10}")

        print()
        print('=' * 65)
        print('           SUMMARY')
        print('=' * 65)
        net_change = memory_only - baseline_only
        direction = "improvement" if net_change > 0 else "regression" if net_change < 0 else "no change"
        print(f'Net change with memory: {net_change:+d} tasks ({memory_only} gained, {baseline_only} lost) - {direction}')
        print()


if __name__ == '__main__':
    main()
