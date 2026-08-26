from __future__ import annotations

import threading
import time
import unittest

from tools.generate_trajectories import (
    ConcurrencyLimitedChatClient,
    ThreadLocalClientPool,
    bounded_completed_futures,
    retry_call,
)


class _ClosableClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class StageAConcurrencyTest(unittest.TestCase):
    def test_thread_local_clients_are_not_shared_between_workers(self):
        created: list[_ClosableClient] = []
        lock = threading.Lock()

        def factory() -> _ClosableClient:
            value = _ClosableClient()
            with lock:
                created.append(value)
            return value

        clients = ThreadLocalClientPool(factory)
        barrier = threading.Barrier(2)
        seen: list[_ClosableClient] = []

        def get_client() -> None:
            value = clients.get()
            seen.append(value)
            barrier.wait(timeout=2)

        first = threading.Thread(target=get_client)
        second = threading.Thread(target=get_client)
        first.start()
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertEqual(len(seen), 2)
        self.assertIsNot(seen[0], seen[1])
        clients.close()
        self.assertTrue(all(client.closed for client in created))

    def test_bounded_completed_futures_reject_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "max_workers"):
            list(bounded_completed_futures(lambda value: value, [1], max_workers=0, max_pending=1))
        with self.assertRaisesRegex(ValueError, "max_pending"):
            list(bounded_completed_futures(lambda value: value, [1], max_workers=1, max_pending=0))

    def test_bounded_completed_futures_returns_every_item(self):
        completed = {
            (value, future.result())
            for value, future in bounded_completed_futures(
                lambda value: value * 10,
                range(7),
                max_workers=2,
                max_pending=3,
            )
        }
        self.assertEqual(completed, {(value, value * 10) for value in range(7)})

    def test_retry_call_retries_transient_failure(self):
        attempts = 0

        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient")
            return "ok"

        self.assertEqual(retry_call(flaky, retries=2, backoff_seconds=0), "ok")
        self.assertEqual(attempts, 3)

    def test_chat_limit_applies_only_to_inflight_completion_calls(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        class Client:
            def complete(self, messages, tools):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return {"role": "assistant", "content": "ok"}

        semaphore = threading.BoundedSemaphore(2)
        clients = [ConcurrencyLimitedChatClient(Client(), semaphore) for _ in range(4)]
        futures = []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as executor:
            for client in clients:
                futures.append(executor.submit(client.complete, [], []))
            for future in futures:
                future.result()
        self.assertEqual(peak, 2)


if __name__ == "__main__":
    unittest.main()
