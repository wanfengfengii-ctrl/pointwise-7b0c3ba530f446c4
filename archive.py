"""archive — persistent LP model/scenario archive with sensitivity recomputation.

A cost analyst stores linear-programming models (variables, minimization
objective, bounded constraints) and named scenarios.  A scenario lists
constraint *edges* (constraint id + lower/upper side) together with rates;
``sensitivity`` returns the right derivative of the optimal value at t = 0
when every listed edge bound is shifted by ``t * rate`` (all listed edges
independently and simultaneously).  Scenarios survive model replacement
("version change") and archive reopen, so stored scenarios can be recomputed
against the current model afterwards.

All numbers accepted and returned are exact rationals (``fractions.Fraction``).
Only the Python standard library is used; persistence is provided by SQLite.
"""

import json
import sqlite3
import sys
import tempfile
from fractions import Fraction

__all__ = ["Archive", "Unbounded", "Infeasible"]


class Unbounded(Exception):
    """The LP objective has no lower bound (minimum is -infinity)."""


class Infeasible(Exception):
    """The LP is infeasible at t = 0 or everywhere in a right neighborhood."""


# ---------------------------------------------------------------------------
# input validation helpers
# ---------------------------------------------------------------------------

def _check_id(value, what):
    """IDs and scenario names must be non-empty strings."""
    if not isinstance(value, str) or value == "":
        raise ValueError("%s must be a non-empty string, got %r" % (what, value))
    return value


def _num(value, what):
    """Accept a non-bool int, a rational string, or a Fraction -> Fraction."""
    if isinstance(value, bool):
        raise ValueError("%s must be an int, rational string or Fraction, got bool" % what)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            raise ValueError("%s is not a rational number string: %r" % (what, value))
    raise ValueError(
        "%s must be a non-bool int, rational string or Fraction, got %r" % (what, value))


def _seq(value, what):
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list or tuple, got %r" % (what, value))
    return list(value)


def _validate_model(model_id, m):
    """Validate (variables, objective, constraints); return normalized model.

    Raises ValueError on any illegal input; nothing is written anywhere.
    """
    _check_id(model_id, "model id")
    parts = _seq(m, "model")
    if len(parts) != 3:
        raise ValueError("model must be a (variables, objective, constraints) triple")
    raw_vars, raw_obj, raw_cons = parts

    variables = []
    seen = set()
    for item in _seq(raw_vars, "variables"):
        pair = _seq(item, "variable")
        if len(pair) != 2:
            raise ValueError("variable must be an (id, free/nonnegative) pair")
        vid, kind = pair
        _check_id(vid, "variable id")
        if vid in seen:
            raise ValueError("duplicate variable id: %r" % vid)
        seen.add(vid)
        if kind not in ("free", "nonnegative"):
            raise ValueError("variable kind must be 'free' or 'nonnegative', got %r" % (kind,))
        variables.append({"id": vid, "kind": kind})
    if not 1 <= len(variables) <= 3:
        raise ValueError("a model needs 1 to 3 variables, got %d" % len(variables))

    nvars = len(variables)
    objective = [_num(c, "objective coefficient") for c in _seq(raw_obj, "objective")]
    if len(objective) != nvars:
        raise ValueError("objective must have exactly %d coefficients" % nvars)

    constraints = []
    seen = set()
    for item in _seq(raw_cons, "constraints"):
        quad = _seq(item, "constraint")
        if len(quad) != 4:
            raise ValueError("constraint must be an (id, coefficients, lower, upper) tuple")
        cid, coeffs, lower, upper = quad
        _check_id(cid, "constraint id")
        if cid in seen:
            raise ValueError("duplicate constraint id: %r" % cid)
        seen.add(cid)
        coeffs = [_num(c, "constraint coefficient") for c in _seq(coeffs, "constraint coefficients")]
        if len(coeffs) != nvars:
            raise ValueError("constraint %r must have exactly %d coefficients" % (cid, nvars))
        lower = None if lower is None else _num(lower, "lower bound")
        upper = None if upper is None else _num(upper, "upper bound")
        if lower is None and upper is None:
            raise ValueError("constraint %r needs at least one bound" % cid)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("constraint %r has lower bound above upper bound" % cid)
        constraints.append({"id": cid, "coeffs": coeffs, "lower": lower, "upper": upper})
    if len(constraints) > 6:
        raise ValueError("a model allows at most 6 constraints, got %d" % len(constraints))

    return {"variables": variables, "objective": objective, "constraints": constraints}


def _validate_scenario(name, d):
    """Validate scenario name and edge list; return [(cid, side, Fraction)]."""
    _check_id(name, "scenario name")
    edges = []
    seen = set()
    for item in _seq(d, "scenario edges"):
        triple = _seq(item, "scenario edge")
        if len(triple) != 3:
            raise ValueError("scenario edge must be a (constraint id, lower/upper, rate) triple")
        cid, side, rate = triple
        _check_id(cid, "constraint id")
        if side not in ("lower", "upper"):
            raise ValueError("scenario edge side must be 'lower' or 'upper', got %r" % (side,))
        rate = _num(rate, "scenario edge rate")
        key = (cid, side)
        if key in seen:
            raise ValueError("duplicate scenario edge: %r %s" % (cid, side))
        seen.add(key)
        edges.append((cid, side, rate))
    return edges


# ---------------------------------------------------------------------------
# JSON (de)serialization — Fractions stored as rational strings
# ---------------------------------------------------------------------------

def _model_to_json(model):
    body = {
        "variables": [[v["id"], v["kind"]] for v in model["variables"]],
        "objective": [str(c) for c in model["objective"]],
        "constraints": [
            [c["id"], [str(a) for a in c["coeffs"]],
             None if c["lower"] is None else str(c["lower"]),
             None if c["upper"] is None else str(c["upper"])]
            for c in model["constraints"]
        ],
    }
    return json.dumps(body, ensure_ascii=False)


def _model_from_json(text):
    body = json.loads(text)
    return {
        "variables": [{"id": vid, "kind": kind} for vid, kind in body["variables"]],
        "objective": [Fraction(c) for c in body["objective"]],
        "constraints": [
            {"id": cid,
             "coeffs": [Fraction(a) for a in coeffs],
             "lower": None if lower is None else Fraction(lower),
             "upper": None if upper is None else Fraction(upper)}
            for cid, coeffs, lower, upper in body["constraints"]
        ],
    }


def _scenario_to_json(edges):
    return json.dumps([[cid, side, str(rate)] for cid, side, rate in edges],
                      ensure_ascii=False)


def _scenario_from_json(text):
    return [(cid, side, Fraction(rate)) for cid, side, rate in json.loads(text)]


# ---------------------------------------------------------------------------
# exact LP solver: two-phase simplex over Fraction with Bland's rule
# ---------------------------------------------------------------------------

def _simplex_eq(c, A, b):
    """Solve  min c.x  s.t. A x = b, x >= 0  exactly.

    Returns (status, value, x): status is 'optimal' (value is a Fraction and
    x a list of Fractions), 'infeasible', or 'unbounded'.
    """
    n = len(c)
    A = [list(row) for row in A]
    b = list(b)
    m = len(A)
    for i in range(m):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]

    # Phase 1 tableau: n original columns + m artificial columns, rhs last.
    ncols = n + m
    T = []
    for i in range(m):
        row = A[i] + [Fraction(0)] * m + [b[i]]
        row[n + i] = Fraction(1)
        T.append(row)
    T.append([Fraction(0)] * (ncols + 1))  # row m: reduced costs
    basis = [n + i for i in range(m)]

    def pivot(pi, pj):
        inv = Fraction(1) / T[pi][pj]
        T[pi] = [v * inv for v in T[pi]]
        prow = T[pi]
        for i in range(m + 1):
            if i != pi:
                f = T[i][pj]
                if f:
                    T[i] = [a - f * bb for a, bb in zip(T[i], prow)]

    def set_objective(costs):
        # reduced cost r_j = c_j - sum_i c_{basis_i} * T[i][j]; rhs entry = -z
        for j in range(ncols + 1):
            s = Fraction(0)
            for i in range(m):
                bi = basis[i]
                if costs[bi]:
                    s += costs[bi] * T[i][j]
            T[m][j] = (costs[j] if j < ncols else Fraction(0)) - s

    def optimize():
        # Bland's rule: smallest-index entering/leaving variable, no cycling.
        while True:
            pj = -1
            row0 = T[m]
            for j in range(ncols):
                if row0[j] < 0:
                    pj = j
                    break
            if pj < 0:
                return "optimal"
            pi = -1
            best_ratio = None
            best_basis = None
            for i in range(m):
                a = T[i][pj]
                if a > 0:
                    ratio = T[i][ncols] / a
                    if (best_ratio is None or ratio < best_ratio
                            or (ratio == best_ratio and basis[i] < best_basis)):
                        best_ratio = ratio
                        best_basis = basis[i]
                        pi = i
            if pi < 0:
                return "unbounded"
            pivot(pi, pj)
            basis[pi] = pj

    # Phase 1: minimize the sum of artificial variables.
    set_objective([Fraction(0)] * n + [Fraction(1)] * m)
    optimize()
    if T[m][ncols] < 0:  # optimal artificial sum = -T[m][ncols] > 0
        return "infeasible", None, None

    # Drive remaining artificial basics out; drop redundant rows.
    for i in range(m):
        if basis[i] >= n:
            pj = -1
            for j in range(n):
                if T[i][j] != 0:
                    pj = j
                    break
            if pj >= 0:
                pivot(i, pj)
                basis[i] = pj
    keep = [i for i in range(m) if basis[i] < n]

    # Phase 2 tableau: original columns only.
    T = [[T[i][j] for j in range(n)] + [T[i][ncols]] for i in keep]
    T.append([Fraction(0)] * (n + 1))
    basis = [basis[i] for i in keep]
    m = len(keep)
    ncols = n

    set_objective(list(c))
    if optimize() == "unbounded":
        return "unbounded", None, None
    x = [Fraction(0)] * n
    for i in range(m):
        x[basis[i]] = T[i][ncols]
    return "optimal", -T[m][ncols], x


def _simplex(c, constraints):
    """Solve  min c.x, x >= 0  s.t. mixed constraints, exactly.

    constraints: list of (coeffs, sense, rhs) with sense in ('<=', '>=', '=');
    all entries Fraction.  Returns the same triple as _simplex_eq.
    """
    n = len(c)
    A = []
    b = []
    ncols = n
    for coeffs, sense, rhs in constraints:
        coeffs = list(coeffs)
        if len(coeffs) != n:
            raise ValueError("constraint coefficient count mismatch")
        if sense == "=":
            A.append(coeffs + [Fraction(0)] * (ncols - n))
            b.append(rhs)
        elif sense in ("<=", ">="):
            for row in A:
                row.append(Fraction(0))
            ncols += 1
            slack = Fraction(1) if sense == "<=" else Fraction(-1)
            A.append(coeffs + [Fraction(0)] * (ncols - n - 1) + [slack])
            b.append(rhs)
        else:
            raise ValueError("unknown constraint sense: %r" % (sense,))
    c_ext = list(c) + [Fraction(0)] * (ncols - n)
    return _simplex_eq(c_ext, A, b)


# ---------------------------------------------------------------------------
# LP formulations built from a stored model
# ---------------------------------------------------------------------------

def _expanded_columns(model):
    """Map variables to non-negative LP columns: (var index, sign)."""
    cols = []
    for j, v in enumerate(model["variables"]):
        if v["kind"] == "nonnegative":
            cols.append((j, 1))
        else:  # free: x = x+ - x-
            cols.append((j, 1))
            cols.append((j, -1))
    return cols


def _primal_lp(model):
    """min c.x over expanded columns; returns (c, constraints, cols)."""
    cols = _expanded_columns(model)
    c = [model["objective"][j] * s for j, s in cols]
    cons = []
    for con in model["constraints"]:
        row = [con["coeffs"][j] * s for j, s in cols]
        if con["lower"] is not None:
            cons.append((row, ">=", con["lower"]))
        if con["upper"] is not None:
            cons.append((row, "<=", con["upper"]))
    return c, cons, cols


def _bound_rows(model, rates):
    """Constraints as >= rows over the original variables: (coeffs, b, d)."""
    rows = []
    for con in model["constraints"]:
        if con["lower"] is not None:
            rows.append((list(con["coeffs"]), con["lower"],
                         rates.get((con["id"], "lower"), Fraction(0))))
        if con["upper"] is not None:
            rows.append(([-a for a in con["coeffs"]], -con["upper"],
                         -rates.get((con["id"], "upper"), Fraction(0))))
    return rows


def _derivative_lp(model, v0, rates):
    """max d.y s.t. y dual feasible and b.y = v0  (as a minimization LP).

    y_k >= 0 is the dual variable of the k-th >= row.  The optimal value of
    this LP is the right derivative of the primal optimum at t = 0; the LP is
    unbounded exactly when the perturbed primal is infeasible for all small
    t > 0 (a Farkas ray with r.b = 0 and r.d > 0 exists).
    """
    rows = _bound_rows(model, rates)
    m = len(rows)
    c = [-rows[k][2] for k in range(m)]  # minimize -d.y
    cons = []
    for j, v in enumerate(model["variables"]):
        col = [rows[k][0][j] for k in range(m)]
        if v["kind"] == "nonnegative":
            cons.append((col, "<=", model["objective"][j]))
        else:  # free variable -> equality in the dual
            cons.append((col, "=", model["objective"][j]))
    cons.append(([rows[k][1] for k in range(m)], "=", v0))  # dual optimality
    return c, cons


# ---------------------------------------------------------------------------
# the archive
# ---------------------------------------------------------------------------

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS models ("
    "  id TEXT PRIMARY KEY, body TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS scenarios ("
    "  model_id TEXT NOT NULL, name TEXT NOT NULL, body TEXT NOT NULL,"
    "  PRIMARY KEY (model_id, name))",
)


class Archive:
    """SQLite-backed archive of LP models and named perturbation scenarios."""

    def __init__(self, path):
        self._conn = sqlite3.connect(path)
        with self._conn:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- internal loaders --------------------------------------------------
    def _load_model(self, model_id, cur=None):
        cur = cur or self._conn.cursor()
        row = cur.execute("SELECT body FROM models WHERE id = ?", (model_id,)).fetchone()
        if row is None:
            raise KeyError("no model stored under id %r" % (model_id,))
        return _model_from_json(row[0])

    def _load_scenario(self, model_id, name, cur=None):
        cur = cur or self._conn.cursor()
        row = cur.execute(
            "SELECT body FROM scenarios WHERE model_id = ? AND name = ?",
            (model_id, name)).fetchone()
        if row is None:
            raise KeyError("no scenario %r stored for model %r" % (name, model_id))
        return _scenario_from_json(row[0])

    # -- writes ------------------------------------------------------------
    def put_model(self, model_id, m):
        """Store m = (variables, objective, constraints) under model_id.

        Replaces any model previously stored under the same id; stored
        scenarios are kept.  Illegal input raises ValueError and leaves the
        previously stored model untouched.
        """
        model = _validate_model(model_id, m)
        body = _model_to_json(model)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO models (id, body) VALUES (?, ?)",
                (model_id, body))

    def save_scenario(self, model_id, name, d):
        """Store scenario *name* for *model_id* as a list of edges.

        Each edge is (constraint id, 'lower'/'upper', rate); the list may be
        empty, and saving under an existing name overwrites it.  Every edge
        must exist in the current model and may not be repeated.  Missing
        model records raise KeyError; illegal input or missing edges raise
        ValueError and leave any previously stored scenario untouched.
        """
        _check_id(model_id, "model id")
        edges = _validate_scenario(name, d)
        model = self._load_model(model_id)  # KeyError if absent
        by_id = {c["id"]: c for c in model["constraints"]}
        for cid, side, _rate in edges:
            con = by_id.get(cid)
            if con is None or con[side] is None:
                raise ValueError("model %r has no %s edge on constraint %r"
                                 % (model_id, side, cid))
        body = _scenario_to_json(edges)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO scenarios (model_id, name, body)"
                " VALUES (?, ?, ?)", (model_id, name, body))

    # -- computations ------------------------------------------------------
    def solve(self, model_id):
        """Return (minimum, {variable id: value}) as Fractions."""
        _check_id(model_id, "model id")
        model = self._load_model(model_id)
        c, cons, cols = _primal_lp(model)
        status, value, x = _simplex(c, cons)
        if status == "infeasible":
            raise Infeasible("model %r is infeasible" % (model_id,))
        if status == "unbounded":
            raise Unbounded("model %r has no lower bound" % (model_id,))
        values = {}
        for (j, s), xv in zip(cols, x):
            vid = model["variables"][j]["id"]
            values[vid] = values.get(vid, Fraction(0)) + s * xv
        return value, values

    def sensitivity(self, model_id, name):
        """Right derivative of the optimal value at t = 0 for a scenario.

        The current model and the stored scenario are read in a single read
        transaction and bound by constraint id and edge; only the listed
        edges are shifted, each by t * rate, independently and at once.
        """
        _check_id(model_id, "model id")
        _check_id(name, "scenario name")
        # Read the current model and the scenario in one read transaction.
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            model = self._load_model(model_id, cur)
            edges = self._load_scenario(model_id, name, cur)
        finally:
            self._conn.commit()

        rates = {}
        by_id = {c["id"]: c for c in model["constraints"]}
        for cid, side, rate in edges:
            con = by_id.get(cid)
            if con is None or con[side] is None:
                raise ValueError("model %r has no %s edge on constraint %r"
                                 % (model_id, side, cid))
            if (cid, side) in rates:
                raise ValueError("duplicate scenario edge: %r %s" % (cid, side))
            rates[(cid, side)] = rate

        # Zero point first: infeasible -> Infeasible, no lower bound -> Unbounded.
        c, cons, _cols = _primal_lp(model)
        status, v0, _x = _simplex(c, cons)
        if status == "infeasible":
            raise Infeasible("model %r is infeasible at t = 0" % (model_id,))
        if status == "unbounded":
            raise Unbounded("model %r has no lower bound at t = 0" % (model_id,))

        # Right derivative = max d.y over optimal dual solutions; unbounded
        # here means the perturbed model is infeasible for every small t > 0.
        dc, dcons = _derivative_lp(model, v0, rates)
        status, value, _y = _simplex(dc, dcons)
        if status == "unbounded":
            raise Infeasible(
                "model %r is infeasible in every right neighborhood of t = 0"
                % (model_id,))
        if status != "optimal":  # strong duality guarantees an optimal dual
            raise Infeasible("dual of model %r is unexpectedly infeasible" % (model_id,))
        return -value


# ---------------------------------------------------------------------------
# CLI: version-change verification and demonstration
# ---------------------------------------------------------------------------

def _spec_models():
    """The worked example: free x, minimize x, edges a/b bounding x below."""
    v1 = ([("x", "free")], [1],
          [("a", [1], 0, None), ("b", [1], 0, None)])
    # version change: constraint b now bounds x by 1 instead of 0
    v2 = ([("x", "free")], [1],
          [("a", [1], 0, None), ("b", [1], 1, None)])
    return v1, v2


def _cmd_verify():
    """Version-change verification: scenarios survive and stay recomputable."""
    checks = []

    def check(label, got, want):
        ok = got == want
        checks.append(ok)
        print("%s %-58s got=%r want=%r" % ("PASS" if ok else "FAIL", label, got, want))

    tmp = tempfile.mkdtemp(prefix="archive-verify-")
    path = tmp + "/costs.db"
    v1, v2 = _spec_models()

    with Archive(path) as arch:
        arch.put_model("m", v1)
        check("solve v1", arch.solve("m"), (Fraction(0), {"x": Fraction(0)}))
        arch.save_scenario("m", "s1", [("a", "lower", 1), ("b", "lower", -1)])
        check("sensitivity at v1", arch.sensitivity("m", "s1"), Fraction(1))

    # reopen after "version change": stored scenario must still recompute
    with Archive(path) as arch:
        check("sensitivity after reopen", arch.sensitivity("m", "s1"), Fraction(1))
        arch.put_model("m", v2)  # new model version, scenario kept
        check("solve v2", arch.solve("m"), (Fraction(1), {"x": Fraction(1)}))
        check("sensitivity recomputed on v2", arch.sensitivity("m", "s1"),
              Fraction(-1))

    # error surface: missing records, unbounded, infeasible
    with Archive(path) as arch:
        try:
            arch.solve("missing")
            check("solve missing model raises KeyError", "no raise", "KeyError")
        except KeyError:
            check("solve missing model raises KeyError", "KeyError", "KeyError")
        try:
            arch.sensitivity("m", "missing")
            check("sensitivity missing scenario raises KeyError", "no raise", "KeyError")
        except KeyError:
            check("sensitivity missing scenario raises KeyError", "KeyError", "KeyError")
        arch.put_model("ub", ([("x", "free")], [1], []))
        try:
            arch.solve("ub")
            check("unbounded solve raises Unbounded", "no raise", "Unbounded")
        except Unbounded:
            check("unbounded solve raises Unbounded", "Unbounded", "Unbounded")
        arch.put_model("inf", ([("x", "nonnegative")], [1],
                               [("a", [1], 2, None), ("b", [1], None, 1)]))
        try:
            arch.solve("inf")
            check("infeasible solve raises Infeasible", "no raise", "Infeasible")
        except Infeasible:
            check("infeasible solve raises Infeasible", "Infeasible", "Infeasible")

    if all(checks):
        print("verify: OK (%d checks)" % len(checks))
        return 0
    print("verify: FAILED (%d/%d checks failed)"
          % (checks.count(False), len(checks)))
    return 1


def _cmd_demo():
    """Demonstrate storing, solving and recomputing a scenario across versions."""
    tmp = tempfile.mkdtemp(prefix="archive-demo-")
    path = tmp + "/costs.db"
    v1, v2 = _spec_models()

    print("== archive demo: cost change rates across a version change ==")
    with Archive(path) as arch:
        arch.put_model("m", v1)
        value, point = arch.solve("m")
        print("model v1 stored: minimize x, free x, a: x >= 0, b: x >= 0")
        print("  solve(m)            -> minimum %s at %s"
              % (value, {k: str(v) for k, v in point.items()}))
        arch.save_scenario("m", "s1", [("a", "lower", 1), ("b", "lower", -1)])
        print("scenario s1 stored: a.lower +1/t, b.lower -1/t")
        print("  sensitivity(m, s1)  -> %s" % arch.sensitivity("m", "s1"))

    print("-- archive closed and reopened (version change, scenarios kept) --")
    with Archive(path) as arch:
        print("  sensitivity(m, s1)  -> %s   (recomputed after reopen)"
              % arch.sensitivity("m", "s1"))
        arch.put_model("m", v2)
        print("model v2 stored: same model, but b: x >= 1")
        print("  sensitivity(m, s1)  -> %s   (stored scenario on new version)"
              % arch.sensitivity("m", "s1"))
    print("== demo done ==")
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] not in ("verify", "demo"):
        print("usage: python3 archive.py {verify|demo}", file=sys.stderr)
        return 2
    return _cmd_verify() if argv[1] == "verify" else _cmd_demo()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
