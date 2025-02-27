#!/usr/bin/env python3
import os
import sys

print("Starting Flask server...")
print("Visit http://127.0.0.1:5050 in your browser")
print("To stop the server, press Ctrl+C")
print("-------------------------------------------------")
os.system("flask run --debug --port=5050")