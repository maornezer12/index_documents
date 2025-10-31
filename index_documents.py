#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Index a document into PostgreSQL with Gemini embeddings.")
    p.add_argument("--file", required=False, help="Path to PDF or DOCX file")
    return p.parse_args()

def main():
    args = parse_args()
    print("Hello World")

if __name__ == "__main__":
    main()
