import unittest
import torch
from runtime_opt.prefix_copy import copy_prefixes, loop_prefix_copy, prepare_prefix_copy, prepare_prefix_copy_cpp


class PrefixCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from runtime_opt.descriptor_extension import initialize
        initialize()

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

    def test_fallbacks(self):
        table = torch.full((4, 16), -9, dtype=torch.int32, device='cuda')
        cases = [([0], [torch.arange(16, device='cuda')[::2]]),
                 ([0], [torch.arange(8, device='cuda', dtype=torch.float32)]),
                 ([0, 0], [torch.ones(8, device='cuda', dtype=torch.int64), torch.zeros(4, device='cuda', dtype=torch.int64)]),
                 ([1], [table[0, :8]])]
        for rows, sources in cases:
            self.assertIsNone(prepare_prefix_copy(table, rows, sources))
            self.assertIsNone(prepare_prefix_copy_cpp(table, rows, sources))
            expected = table.clone(); loop_prefix_copy(expected, rows, sources)
            copy_prefixes(table, rows, sources, min_segments=0)
            torch.testing.assert_close(table, expected, rtol=0, atol=0)

    def test_stream_lifetime(self):
        table = torch.full((32, 1024), -9, dtype=torch.int32, device='cuda')
        stream = torch.cuda.Stream()
        for iteration in range(20):
            with torch.cuda.stream(stream):
                sources = [torch.full((513,), i + iteration, dtype=torch.int64, device='cuda') for i in range(32)]
                copy_prefixes(table, list(range(32)), sources, min_segments=0)
                del sources
                trash = torch.empty((32 * 513,), dtype=torch.int64, device='cuda')
                trash.fill_(-500)
            stream.synchronize()
            expected = torch.arange(32, device='cuda', dtype=torch.int32) + iteration
            torch.testing.assert_close(table[:, :513], expected[:, None].expand(-1, 513), rtol=0, atol=0)
            self.assertTrue(torch.all(table[:, 513:] == -9).item())

    def test_producer_consumer_streams(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            sources = [torch.full((257,), i, dtype=torch.int64, device='cuda') for i in range(32)]
        consumer.wait_stream(producer)
        with torch.cuda.stream(consumer):
            table = torch.full((32, 300), -99, dtype=torch.int32, device='cuda')
            copy_prefixes(table, list(range(32)), sources)
        del sources
        with torch.cuda.stream(producer):
            trash = torch.full((32 * 257,), -500, dtype=torch.int64, device='cuda')
        consumer.synchronize()
        expected = torch.arange(32, device='cuda', dtype=torch.int32)
        torch.testing.assert_close(table[:, :257], expected[:, None].expand(-1, 257), rtol=0, atol=0)

    def test_overlapping_destination_fallback(self):
        storage = torch.zeros(16, dtype=torch.int32, device='cuda')
        table = storage[None, :].expand(32, -1)
        sources = [torch.full((16,), i, dtype=torch.int64, device='cuda') for i in range(32)]
        self.assertIsNone(prepare_prefix_copy(table, list(range(32)), sources))
        self.assertIsNone(prepare_prefix_copy_cpp(table, list(range(32)), sources))
        copy_prefixes(table, list(range(32)), sources)
        self.assertTrue(torch.all(table == 31).item())

    def test_complete_api_capture_fallback(self):
        table = torch.zeros((32, 128), dtype=torch.int32, device='cuda')
        sources = [torch.zeros(128, dtype=torch.int64, device='cuda') for _ in range(32)]
        copy_prefixes(table, list(range(32)), sources)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): copy_prefixes(table, list(range(32)), sources)
        for i, source in enumerate(sources): source.fill_(i + 1)
        graph.replay()
        expected = torch.arange(1, 33, dtype=torch.int32, device='cuda')
        torch.testing.assert_close(table, expected[:, None].expand(-1, 128), rtol=0, atol=0)

    def test_graph_changed_inputs(self):
        table = torch.empty((8, 256), dtype=torch.int32, device='cuda')
        sources = [torch.zeros(256, device='cuda', dtype=torch.int64) for _ in range(8)]
        plan = prepare_prefix_copy_cpp(table, list(range(8)), sources)
        plan.launch(table); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): plan.launch(table)
        for iteration in range(10):
            for i, source in enumerate(sources): source.fill_(i + iteration)
            graph.replay()
            expected = torch.arange(8, device='cuda', dtype=torch.int32) + iteration
            torch.testing.assert_close(table, expected[:, None].expand(-1, 256), rtol=0, atol=0)


if __name__ == '__main__': unittest.main(verbosity=2)
