# ouroboros/compression/program_synthesis.py
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class NodeType(Enum):
    CONST = auto()
    TIME  = auto()
    ADD   = auto()
    MUL   = auto()
    MOD   = auto()

@dataclass
class ExprNode:
    node_type: NodeType
    value: Optional[int] = None
    left: Optional['ExprNode'] = None
    right: Optional['ExprNode'] = None

    def evaluate(self, t: int) -> int:
        match self.node_type:
            case NodeType.CONST: return self.value
            case NodeType.TIME:  return t
            case NodeType.ADD:   return self.left.evaluate(t) + self.right.evaluate(t)
            case NodeType.MUL:   return self.left.evaluate(t) * self.right.evaluate(t)
            case NodeType.MOD:
                r = self.right.evaluate(t)
                return self.left.evaluate(t) % r if r != 0 else 0

    def predict_sequence(self, length, alphabet_size=None):
        preds = [self.evaluate(t) for t in range(length)]
        if alphabet_size: preds = [p % alphabet_size for p in preds]
        return preds

    def to_string(self):
        match self.node_type:
            case NodeType.CONST: return str(self.value)
            case NodeType.TIME:  return 't'
            case NodeType.ADD:   return f"({self.left.to_string()} + {self.right.to_string()})"
            case NodeType.MUL:   return f"({self.left.to_string()} * {self.right.to_string()})"
            case NodeType.MOD:   return f"({self.left.to_string()} mod {self.right.to_string()})"

    def to_bytes(self): return self.to_string().encode()

def C(n): return ExprNode(NodeType.CONST, value=n)
def T():  return ExprNode(NodeType.TIME)
def ADD(l, r): return ExprNode(NodeType.ADD, left=l, right=r)
def MUL(l, r): return ExprNode(NodeType.MUL, left=l, right=r)
def MOD(l, r): return ExprNode(NodeType.MOD, left=l, right=r)

def build_linear_modular(slope, intercept, modulus):
    return MOD(ADD(MUL(C(slope), T()), C(intercept)), C(modulus))