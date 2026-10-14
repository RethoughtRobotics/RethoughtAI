"""A policy's module tree: the layers to peel back, with what each one does in a real forward pass.

The tree is `named_modules()`, so it is the same for every PyTorch policy (ACT, SmolVLA, VITA,
GR00T). One traced forward pass adds, per module, how often it ran and the shapes it took and gave:
a SmolVLA expert layer runs once per flow step, and a module that never runs (an unused LM head,
or one whose weights are used directly) shows 0 calls.
"""

import json
from dataclasses import asdict, dataclass, field

ATTENTION = ("Attention", "MultiheadAttention")  # class names that compute attention


@dataclass
class Node:
    name: str  # path from the root, e.g. "model.vlm_with_expert.lm_expert.layers.3.self_attn"
    type: str
    params: int  # in this module and below
    calls: int = 0
    inputs: list = field(default_factory=list)  # shapes of the first call's tensor inputs
    outputs: list = field(default_factory=list)
    attention: bool = False
    children: list["Node"] = field(default_factory=list)


def tree(module, name: str = "") -> Node:
    node = Node(name, type(module).__name__, sum(p.numel() for p in module.parameters()),
                attention=any(a in type(module).__name__ for a in ATTENTION))
    node.children = [tree(child, f"{name}.{n}" if name else n) for n, child in module.named_children()]
    return node


def _shapes(value) -> list:
    """Shapes of the tensors in a module's inputs or outputs (nested tuples, lists and dicts)."""
    if hasattr(value, "shape"):
        return [list(value.shape)]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (tuple, list)):
        return [s for v in value for s in _shapes(v)]
    return []


def trace(module, run) -> Node:
    """The tree of `module`, with calls and shapes from `run()` (one forward pass through it)."""
    root = tree(module)
    nodes = {n.name: n for n in walk(root)}
    names = {m: n for n, m in module.named_modules()}

    def hook(mod, args, out):
        node = nodes[names[mod]]
        node.calls += 1
        if node.calls == 1:
            node.inputs, node.outputs = _shapes(args), _shapes(out)

    handles = [m.register_forward_hook(hook) for m in names]
    try:
        run()
    finally:
        for h in handles:
            h.remove()
    return root


def walk(node: Node):
    yield node
    for child in node.children:
        yield from walk(child)


def blank_observation(features: dict) -> dict:
    """Zero images and state shaped as a checkpoint's input_features, as LeRobotPolicy takes them."""
    import numpy as np

    out = {}
    for key, f in features.items():
        shape = f["shape"] if isinstance(f, dict) else list(f.shape)
        kind = f["type"] if isinstance(f, dict) else f.type.name
        out[key] = np.zeros((*shape[1:], shape[0]), np.uint8) if kind == "VISUAL" else np.zeros(shape, np.float32)
    return out


def trace_policy(policy) -> Node:
    """A LeRobotPolicy's tree, traced through one inference on a blank observation."""
    features = policy.policy.config.input_features
    observation = blank_observation(features)
    policy.reset()
    root = trace(policy.policy, lambda: policy(observation))
    policy.reset()
    return root


def to_json(root: Node) -> str:
    return json.dumps(asdict(root))


def from_json(text: str) -> Node:
    def build(d):
        return Node(**{**d, "children": [build(c) for c in d["children"]]})

    return build(json.loads(text))


def signature(node: Node) -> tuple:
    """What a module is and does, without its name: equal for repeated layers of a stack."""
    return (node.type, node.params, node.calls, str(node.inputs), str(node.outputs),
            tuple(signature(c) for c in node.children))


def period(node: Node) -> int:
    """The repeating unit of a stack of numbered layers: 1 when all are alike, 2 when they alternate
    (as SmolVLA's expert: self-attention, cross-attention), ...; 0 when it does not repeat."""
    children = node.children
    if len(children) < 2 or not all(c.name.rsplit(".", 1)[-1].isdigit() for c in children):
        return 0
    sigs = [signature(c) for c in children]
    for p in range(1, len(sigs) // 2 + 1):
        if all(sigs[i] == sigs[i % p] for i in range(len(sigs))):
            return p
    return 0


def text(root: Node, depth: int | None = None) -> str:
    """The tree indented, one module per line: name, type, parameters, calls, output shapes.
    A stack of repeated layers shows its repeating unit only (the first layer, or first two)."""
    lines = []

    def line(node, level):
        if depth is not None and level > depth:
            return
        label = node.name.rsplit(".", 1)[-1] or "(policy)"
        shapes = " ".join("×".join(map(str, s)) for s in node.outputs[:2])
        mark = " [attention]" if node.attention else ""
        lines.append(f"{'  ' * level}{label}  {node.type}{mark}  {_count(node.params)}  "
                     f"calls {node.calls}  {shapes}".rstrip())
        p = period(node)
        for child in node.children[:p] if p else node.children:
            line(child, level + 1)
        if p and (depth is None or level < depth):
            unit = "0" if p == 1 else f"0–{p - 1}"
            lines.append(f"{'  ' * (level + 1)}{p}–{len(node.children) - 1}  repeat {unit}")

    line(root, 0)
    return "\n".join(lines)


def _count(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)
