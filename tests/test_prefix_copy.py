import unittest
import torch
from runtime_opt.prefix_copy import copy_prefixes, loop_prefix_copy

class PrefixCopyTests(unittest.TestCase):
    def test_integer_shapes(self):
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_device(device)
            for batch in [0, 1, 8, 32, 128, 512]:
                for width in [1, 8, 31, 128, 513, 4097]:
                    with self.subTest(device=device, batch=batch, width=width):
                        table = torch.full((batch + 4, width + 17), -99, dtype=torch.int32, device=f'cuda:{device}')
                        rows = list(range(2, batch + 2))
                        sources = [torch.arange((i * 37) % (width + 1), device=table.device,
                                   dtype=torch.int64 if i % 2 else torch.int32) - i * 100003 for i in range(batch)]
                        expected = table.clone(); loop_prefix_copy(expected, rows, sources)
                        copy_prefixes(table, rows, sources, min_segments=0)
                        torch.testing.assert_close(table, expected, rtol=0, atol=0)

    def test_cast_overflow(self):
        table = torch.zeros((1, 6), dtype=torch.int32, device='cuda')
        source = torch.tensor([-2**40, -2**31-1, 2**31, 2**32+1, 2**40, 2**63-1], device='cuda', dtype=torch.int64)
        copy_prefixes(table, [0], [source], min_segments=0)
        torch.testing.assert_close(table[0], source.to(torch.int32), rtol=0, atol=0)
