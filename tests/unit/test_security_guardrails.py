from pathlib import Path


def test_shipped_source_does_not_use_torch_jit() -> None:
    source_root = Path(__file__).parents[2] / "src" / "eleanity"
    offenders = [
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        if "torch.jit" in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"TorchScript is outside Eleanity's supported security boundary: {offenders}"
