#!/usr/bin/env python3
"""Health-aware, least-inflight endpoint selection shared by legacy drivers.

Round-robin scheduling sends work to unavailable hosts and wastes retries,
which can make infrastructure failures resemble model failures. A historical
preemption required twenty minutes to restore model weights on a new host.

Recheck /health every 60 seconds and use the least-busy healthy endpoint.
Offline hosts are skipped; recovered hosts rejoin without restarting evaluation.
"""
from __future__ import annotations

import asyncio

class EndpointPool:
    def __init__(self, endpoints: list[str]):
        self.eps = [e.strip().rstrip("/") for e in endpoints if e.strip()]
        if not self.eps:
            raise ValueError("Endpoint list is empty")
        self.inflight = {e: 0 for e in self.eps}
        
        self.healthy = {e: True for e in self.eps}

        self._sessions: dict[int, object] = {}
        self._cv = asyncio.Condition()

    def attach(self, ctx) -> None:
        """Register an inflight session so it can move if its endpoint goes offline."""
        self._sessions[id(ctx)] = ctx

    def detach(self, ctx) -> None:
        self._sessions.pop(id(ctx), None)

    def _reassign(self, dead: str) -> int:
        alive = [e for e in self.eps if self.healthy[e]]
        if not alive:
            return 0
        n = 0
        for ctx in list(self._sessions.values()):
            if getattr(ctx, "adapter_url", None) == dead:
                
                tgt = min(alive, key=lambda x: self.inflight[x])

                object.__setattr__(ctx, "adapter_url", tgt)
                self.inflight[dead] = max(0, self.inflight[dead] - 1)
                self.inflight[tgt] += 1
                n += 1
        return n

    async def health_loop(self, period: int = 60) -> None:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as c:
            while True:
              try:
                for e in self.eps:
                    try:
                        ok = (await c.get(f"{e}/health")).status_code == 200
                    except Exception:
                        ok = False
                    if ok != self.healthy[e]:
                        print(f"[pool] {e} -> {'HEALTHY' if ok else 'DOWN'}", flush=True)
                    was = self.healthy[e]
                    self.healthy[e] = ok
                    if was and not ok:

                        try:
                            moved = self._reassign(e)
                            print(f"[pool] {e} went offline; reassigned {moved} inflight sessions", flush=True)
                        except Exception as ex:
                            print(f"[pool] Reassignment failed ({type(ex).__name__}: {ex}); health checks continue", flush=True)
                async with self._cv:
                    self._cv.notify_all()
              except Exception as ex:          
                  print(f"[pool] Health loop error ({type(ex).__name__}: {ex}); continuing", flush=True)
              await asyncio.sleep(period)

    async def acquire(self) -> str:
        async with self._cv:
            while not any(self.healthy.values()):
                print("[pool] No healthy endpoints; waiting for the next check...", flush=True)
                await self._cv.wait()
            e = min((x for x in self.eps if self.healthy[x]), key=lambda x: self.inflight[x])
            self.inflight[e] += 1
            return e

    async def release(self, e: str) -> None:
        async with self._cv:
            self.inflight[e] = max(0, self.inflight[e] - 1)
            self._cv.notify_all()

    def snapshot(self) -> str:
        return " ".join(f"{e.split('//')[-1]}:{'✓' if self.healthy[e] else '✗'}{self.inflight[e]}"
                        for e in self.eps)
