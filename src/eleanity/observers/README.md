# Observers

Observers will encapsulate cross-cutting measurement hooks (template capture, token probes, stream recorders) that adapters can share.

In the MVP, observation is performed directly inside each `BackendAdapter` and assembled by `eleanity.core.run.observe`.
