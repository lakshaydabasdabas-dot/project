#!/usr/bin/env python3
"""
Simple script to view visualization files
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIS_DIR = BASE_DIR / "visualization_outputs"
REPORT_FILE = BASE_DIR / "visualization_report.html"

print("Student Score Prediction - Visualization Viewer")
print("=" * 50)

# Check if visualization directory exists
if not VIS_DIR.exists():
    print(f"Error: Visualization directory not found: {VIS_DIR}")
    print("Please run the main script first to generate visualizations.")
    sys.exit(1)

# List all visualization files
print(f"\nVisualization files in '{VIS_DIR.name}/':")
print("-" * 50)

image_files = []
for file in sorted(VIS_DIR.iterdir()):
    if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        size_kb = file.stat().st_size / 1024
        image_files.append(file)
        print(f"  • {file.name:40} ({size_kb:.1f} KB)")

print(f"\nTotal: {len(image_files)} visualization files")

# Check if HTML report exists
if REPORT_FILE.exists():
    print(f"\nHTML Report: {REPORT_FILE.name}")
    print(f"  File size: {REPORT_FILE.stat().st_size / 1024:.1f} KB")
    
    # Ask user what they want to do
    print("\nOptions:")
    print("  1. Open HTML report in browser (recommended)")
    print("  2. List all image files")
    print("  3. Open specific image file")
    print("  4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Try to open HTML report
        print(f"\nOpening HTML report in browser...")
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(REPORT_FILE)])
            elif sys.platform == "win32":
                os.startfile(str(REPORT_FILE))
            else:
                # Linux
                subprocess.run(["xdg-open", str(REPORT_FILE)])
            print("HTML report opened successfully!")
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"You can manually open: file://{REPORT_FILE.resolve()}")
    
    elif choice == "2":
        print("\nImage files:")
        for i, file in enumerate(image_files, 1):
            print(f"  {i:2}. {file.name}")
    
    elif choice == "3":
        print("\nSelect image to open:")
        for i, file in enumerate(image_files, 1):
            print(f"  {i:2}. {file.name}")
        
        try:
            img_choice = int(input("\nEnter image number: ")) - 1
            if 0 <= img_choice < len(image_files):
                img_file = image_files[img_choice]
                print(f"Opening {img_file.name}...")
                
                try:
                    if sys.platform == "darwin":
                        subprocess.run(["open", str(img_file)])
                    elif sys.platform == "win32":
                        os.startfile(str(img_file))
                    else:
                        # Linux
                        subprocess.run(["xdg-open", str(img_file)])
                    print("Image opened successfully!")
                except Exception as e:
                    print(f"Could not open image: {e}")
                    print(f"File location: {img_file.resolve()}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice.")

else:
    print(f"\nHTML report not found: {REPORT_FILE.name}")
    print("You can view individual image files in the visualization_outputs/ folder.")

print("\n" + "=" * 50)
print("Project completed successfully!")
