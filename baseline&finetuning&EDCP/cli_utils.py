"""Lightweight argument helpers shared by the baseline and EDCP entry points."""
import argparse


def parse_bool(value):
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false, yes/no or 1/0")
