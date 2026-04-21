from pathlib import Path

src = Path("ouroboros/environments/base.py").read_text()
src = src.replace(
    "    def __init__(self, alphabet_size: int, seed: int = 42):",
    "    def __init__(self, alphabet_size: int = 256, seed: int = 42):"
)
src = src.replace(
    "    @abstractmethod\n    def _generate_stream(self, length: int) -> List[int]:\n        \"\"\"Generate observation stream of given length.\"\"\"\n        ...",
    "    def _generate_stream(self, length: int = 1000) -> List[int]:\n        \"\"\"Generate observation stream of given length.\"\"\"\n        return []"
)
Path("ouroboros/environments/base.py").write_text(src)
print("Done")
