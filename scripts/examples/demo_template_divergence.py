#!/usr/bin/env python3
"""Compatibility wrapper for the packaged offline divergence demo."""

from eleanity.core.demo import render_template_divergence_demo, run_template_divergence_demo

if __name__ == "__main__":
    print(render_template_divergence_demo(run_template_divergence_demo()))
