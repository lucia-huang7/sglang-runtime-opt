import torch

def loop_prefix_copy(table, rows, sources):
    for row, source in zip(rows, sources):
        if len(source):
            table[row, :len(source)] = source

def copy_prefixes(table, rows, sources, min_segments=16, backend="cpp"):
    return loop_prefix_copy(table, rows, sources)
