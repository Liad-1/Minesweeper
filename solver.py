"""The AUTO SOLVER.

Looks only at what a player can see (hidden / flagged cells and the numbers on
revealed cells) and returns the next moves:

1. opening move in the centre (safe with easy start),
2. count rule (all mines flagged / every hidden cell is a mine),
3. single-number rule,
4. subset rule between two numbers,
5. exact mine probabilities over all arrangements of the frontier
   (0% -> reveal, 100% -> flag),
6. otherwise a guess: the cell with the lowest mine probability.
"""
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

NODE_LIMIT = 200_000  # backtracking budget per frontier component
EPS = 1e-9


@dataclass
class Move:
    type: str  # "reveal" or "flag"
    x: int
    y: int
    reason: str
    guess: bool = False
    p: float = 0.0


@dataclass
class Result:
    moves: List[Move]
    probs: Optional[Dict[tuple, float]] = None  # (x, y) -> mine probability
    stage: str = "rule"


@dataclass
class _Constraint:
    x: int
    y: int
    value: int
    cells: List[tuple]
    flags: int
    remaining: int
    cell_set: set = field(default_factory=set)


def pct(p: float) -> str:
    if 0 < p < 0.005:
        return "<1"
    if 0.995 < p < 1:
        return ">99"
    return str(round(p * 100))


def plural(n: int, word: str) -> str:
    return "%i %s%s" % (n, word, "" if n == 1 else "s")


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def convolve(a: List[float], b: List[float]) -> List[float]:
    r = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                r[i + j] += ai * bj
    return r


def label(x: int, y: int) -> str:
    return "(%i, %i)" % (x + 1, y + 1)


class Solver:

    def __init__(self, grid, easy_start: bool) -> None:
        self.grid = grid
        self.easy_start = easy_start
        self.h, self.w = grid.contents.shape

    def neighbours(self, x: int, y: int):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if (dx or dy) and 0 <= x + dx < self.w and 0 <= y + dy < self.h:
                    yield x + dx, y + dy

    def next_moves(self) -> Result:
        grid = self.grid
        cells = grid.contents
        if grid.troll_mode:
            return Result([], stage="troll")

        # 1. Opening move.
        if not grid.clicked_cell:
            x, y = self.w // 2, self.h // 2
            if self.easy_start:
                return Result([Move("reveal", x, y, "opening move; easy start keeps the first click safe")])
            p = grid.mines / grid.tiles
            return Result([Move("reveal", x, y, "opening guess, %s%% risk" % pct(p), True, p)], stage="guess")

        flags = 0
        unknown = []
        for y in range(self.h):
            for x in range(self.w):
                if cells[y][x].flagged:
                    flags += 1
                elif cells[y][x].hidden:
                    unknown.append((x, y))
        unknown_set = set(unknown)
        mines_left = grid.mines - flags
        if not unknown:
            return Result([], stage="done")

        # 2. Count rule.
        if mines_left <= 0:
            return Result([Move("reveal", x, y, "every mine is already flagged") for x, y in unknown])
        if len(unknown) == mines_left:
            reason = ("the last hidden cell must be the last mine" if len(unknown) == 1
                      else "the %i hidden cells left must all be mines" % len(unknown))
            return Result([Move("flag", x, y, reason) for x, y in unknown])

        # Constraints: each revealed number with hidden neighbours.
        cons: List[_Constraint] = []
        for y in range(self.h):
            for x in range(self.w):
                cell = cells[y][x]
                if cell.hidden or not 1 <= cell.value <= 8:
                    continue
                hidden, f = [], 0
                for nx, ny in self.neighbours(x, y):
                    if cells[ny][nx].flagged:
                        f += 1
                    elif (nx, ny) in unknown_set:
                        hidden.append((nx, ny))
                if hidden:
                    cons.append(_Constraint(x, y, cell.value, hidden, f, cell.value - f, set(hidden)))

        out: Dict[tuple, Move] = {}

        def add(kind: str, pos: tuple, reason: str):
            if pos not in out:
                out[pos] = Move(kind, pos[0], pos[1], reason)

        # 3. Single-number rule.
        for c in cons:
            at = label(c.x, c.y)
            if c.remaining == 0:
                for pos in c.cells:
                    add("reveal", pos, "the %i at %s already has its %s" % (c.value, at, "mine" if c.value == 1 else "mines"))
            elif c.remaining == len(c.cells):
                if c.flags == 0:
                    reason = "the %i at %s has exactly %s" % (c.value, at, plural(c.value, "hidden neighbour"))
                else:
                    reason = "the %i at %s needs %i more and has %s" % (c.value, at, c.remaining, plural(c.remaining, "hidden neighbour"))
                for pos in c.cells:
                    add("flag", pos, reason)
        if out:
            return Result(list(out.values()))

        # 4. Subset rule: a's hidden cells are all next to b too.
        for a in cons:
            for b in cons:
                if a is b or len(a.cells) >= len(b.cells):
                    continue
                if abs(a.x - b.x) > 2 or abs(a.y - b.y) > 2 or not a.cell_set <= b.cell_set:
                    continue
                diff = [pos for pos in b.cells if pos not in a.cell_set]
                rd = b.remaining - a.remaining
                pair = "the %i at %s" % (b.value, label(b.x, b.y))
                other = "the %i at %s" % (a.value, label(a.x, a.y))
                if rd == 0:
                    for pos in diff:
                        add("reveal", pos, "%s gets all its remaining mines from the cells next to %s" % (pair, other))
                elif rd == len(diff):
                    for pos in diff:
                        add("flag", pos, "%s needs %s than %s can supply" % (pair, plural(rd, "more mine"), other))
        if out:
            return Result(list(out.values()))

        # 5. Exact probabilities.
        probs = self.probabilities(cons, unknown, mines_left)
        for pos, p in probs.items():
            if p <= EPS:
                add("reveal", pos, "0% mine in every possible arrangement")
            elif p >= 1 - EPS:
                add("flag", pos, "a mine in every possible arrangement")
        if out:
            return Result(list(out.values()), probs, "probability")

        # 6. Guess the safest cell (ties: corners, then edges).
        def edge_score(pos):
            x, y = pos
            return (x in (0, self.w - 1)) + (y in (0, self.h - 1))

        best, best_p = None, math.inf
        for pos, p in probs.items():
            if p < best_p - EPS or (abs(p - best_p) <= EPS and edge_score(pos) > edge_score(best)):
                best, best_p = pos, p
        return Result([Move("reveal", best[0], best[1], "lowest risk, %s%%" % pct(best_p), True, best_p)], probs, "guess")

    def probabilities(self, cons: List[_Constraint], unknown: List[tuple], mines_left: int) -> Dict[tuple, float]:
        # Group frontier cells into components linked by shared numbers.
        parent: Dict[tuple, tuple] = {}

        def find(pos):
            while parent[pos] != pos:
                parent[pos] = parent[parent[pos]]
                pos = parent[pos]
            return pos

        for c in cons:
            for pos in c.cells:
                parent.setdefault(pos, pos)
        for c in cons:
            first = find(c.cells[0])
            for pos in c.cells[1:]:
                parent[find(pos)] = first
        comps: Dict[tuple, dict] = {}
        for pos in parent:
            comps.setdefault(find(pos), {"cells": [], "cons": []})["cells"].append(pos)
        for c in cons:
            comps[find(c.cells[0])]["cons"].append(c)

        interior = [pos for pos in unknown if pos not in parent]
        n_int = len(interior)
        probs: Dict[tuple, float] = {}

        def fallback():
            for c in cons:
                for pos in c.cells:
                    probs[pos] = max(probs.get(pos, 0.0), c.remaining / len(c.cells))
            for pos in interior:
                probs[pos] = min(1.0, mines_left / len(unknown))
            return probs

        dists = []
        for comp in comps.values():
            d = self.enumerate(comp, mines_left)
            if d is None:
                return fallback()
            dists.append(d)

        total = [1.0]
        for d in dists:
            total = convolve(total, d["counts"])

        def log_w(k):
            return log_choose(n_int, mines_left - k)

        valid = [log_w(k) for k, t in enumerate(total) if t > 0 and log_w(k) > -math.inf]
        if not valid:
            return fallback()
        max_lw = max(valid)

        def weight(k):
            lw = log_w(k)
            return 0.0 if lw == -math.inf else math.exp(lw - max_lw)

        z = interior_mines = 0.0
        for k, t in enumerate(total):
            wk = t * weight(k)
            z += wk
            interior_mines += wk * (mines_left - k)
        if not z > 0:
            return fallback()

        for i, d in enumerate(dists):
            rest = [1.0]
            for j, other in enumerate(dists):
                if j != i:
                    rest = convolve(rest, other["counts"])
            sums = [0.0] * len(d["cells"])
            for k, cm in enumerate(d["cell_mines"]):
                if cm is None:
                    continue
                wk = sum(r * weight(k + k2) for k2, r in enumerate(rest) if r)
                if wk:
                    for c, m in enumerate(cm):
                        sums[c] += m * wk
            for c, pos in enumerate(d["cells"]):
                probs[pos] = sums[c] / z
        p_int = interior_mines / z / n_int if n_int else 0.0
        for pos in interior:
            probs[pos] = p_int
        return probs

    def enumerate(self, comp: dict, mines_left: int) -> Optional[dict]:
        """Counts the mine arrangements of one component that satisfy its numbers,
        by number of mines k, and how often each cell is a mine (None = too big)."""
        cells = comp["cells"]
        n = len(cells)
        local = {pos: i for i, pos in enumerate(cells)}
        cons = [{"r": c.remaining, "mines": 0, "left": len(c.cells)} for c in comp["cons"]]
        var_cons = [[] for _ in range(n)]
        for c, src in zip(cons, comp["cons"]):
            for pos in src.cells:
                var_cons[local[pos]].append(c)

        counts = [0] * (n + 1)
        cell_mines: List[Optional[List[int]]] = [None] * (n + 1)
        assign = [0] * n
        state = {"nodes": 0, "mines": 0, "aborted": False}

        def rec(i):
            state["nodes"] += 1
            if state["aborted"] or state["nodes"] > NODE_LIMIT:
                state["aborted"] = True
                return
            if i == n:
                k = state["mines"]
                counts[k] += 1
                cm = cell_mines[k]
                if cm is None:
                    cm = cell_mines[k] = [0] * n
                for v in range(n):
                    if assign[v]:
                        cm[v] += 1
                return
            for val in (0, 1):
                if val and state["mines"] >= mines_left:
                    continue
                ok = True
                for c in var_cons[i]:
                    c["left"] -= 1
                    c["mines"] += val
                    if c["mines"] > c["r"] or c["mines"] + c["left"] < c["r"]:
                        ok = False
                if ok:
                    assign[i] = val
                    state["mines"] += val
                    rec(i + 1)
                    state["mines"] -= val
                    assign[i] = 0
                for c in var_cons[i]:
                    c["left"] += 1
                    c["mines"] -= val

        rec(0)
        if state["aborted"]:
            return None
        return {"cells": cells, "counts": counts, "cell_mines": cell_mines}
