# Plugins

Future adapters can register via `importlib.metadata` entry points, for example:

```toml
[project.entry-points."eleanity.adapters"]
sglang = "eleanity_sglang:SGLangAdapter"
```

The core diagnoser and reporters remain adapter-agnostic: plugins only need to implement the adapter protocol and declare capabilities.
