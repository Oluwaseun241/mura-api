#!/usr/bin/env python3
"""
Test client for the West African Food Classifier API
"""

import argparse
import os
from pathlib import Path
import sys
import time
from contextlib import ExitStack

import requests
import json

class FoodClassifierClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return None
    
    def get_classes(self):
        """Get all available food classes"""
        try:
            response = requests.get(f"{self.base_url}/classes", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get classes: {e}")
            return None
    
    def predict_from_file(self, image_path, top_k=5):
        """Predict food class from image file"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'top_k': top_k}
                response = requests.post(f"{self.base_url}/predict", files=files, data=data, timeout=30)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Prediction failed: {e}")
            return None

    def predict_directory(self, directory, top_k=5, recursive=False, use_batch=False, batch_size=8):
        """Run predictions for all images in a directory"""
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        path = Path(directory)

        if not path.exists() or not path.is_dir():
            print(f"❌ Directory not found: {directory}")
            return []

        files = sorted(path.rglob('*') if recursive else path.glob('*'))
        image_files = [f for f in files if f.suffix.lower() in supported_exts]

        if not image_files:
            print(f"⚠️  No image files found in {directory}")
            return []

        print(f"📁 Running predictions for {len(image_files)} image(s) in '{directory}'{' (recursive)' if recursive else ''}...")

        summary = []
        failures = 0
        start_time = time.time()

        if use_batch:
            total = len(image_files)
            sent = 0
            for chunk in self._chunked(image_files, batch_size):
                chunk_paths = [str(p) for p in chunk]
                sent += len(chunk_paths)
                print(f"🧠 Batch {sent}/{total} images...")
                batch_result = self.predict_batch_from_files(chunk_paths, top_k=top_k)
                if not batch_result or not batch_result.get("success"):
                    print("   ❌ Batch request failed")
                    failures += len(chunk_paths)
                    continue

                for entry in batch_result.get("results", []):
                    if entry.get("success") and entry.get("top_prediction"):
                        top = entry["top_prediction"]
                        print(f"   ✅ {top['class_name']} ({top['confidence_percentage']:.1f}%) - {entry.get('filename')}")
                        summary.append({
                            "path": entry.get("filename"),
                            "result": entry,
                        })
                    else:
                        print(f"   ❌ Prediction failed - {entry.get('filename')}")
                        failures += 1
        else:
            for idx, image_path in enumerate(image_files, 1):
                print(f"[{idx}/{len(image_files)}] 🔍 {image_path}")
                result = self.predict_from_file(str(image_path), top_k=top_k)
                if result and result.get('success') and result.get('top_prediction'):
                    top = result['top_prediction']
                    print(f"   ✅ {top['class_name']} ({top['confidence_percentage']:.1f}%)")
                    summary.append({
                        "path": str(image_path),
                        "result": result
                    })
                else:
                    print("   ❌ Prediction failed")
                    failures += 1

        duration = time.time() - start_time
        print("\n📊 Batch summary")
        print(f"   ✅ Successful predictions: {len(summary)}")
        print(f"   ❌ Failures: {failures}")
        print(f"   ⏱️  Total time: {duration:.1f}s")

        if summary:
            print("\n🏆 Top predictions per file")
            for entry in summary:
                top = entry['result']['top_prediction']
                display_name = top['class_name'].replace('_', ' ').title()
                print(f" - {entry['path']}: {display_name} ({top['confidence_percentage']:.1f}%)")

        return summary

    def predict_batch_from_files(self, image_paths, top_k=5):
        """Predict multiple images in one request."""
        if not image_paths:
            return None

        url = f"{self.base_url}/predict-batch"
        multipart_files = []

        with ExitStack() as stack:
            for p in image_paths:
                f = stack.enter_context(open(p, "rb"))
                multipart_files.append(
                    ("files", (os.path.basename(p), f, "application/octet-stream"))
                )

            data = {"top_k": top_k}
            response = requests.post(url, files=multipart_files, data=data, timeout=120)
            response.raise_for_status()
            return response.json()

    def _chunked(self, items, chunk_size):
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
    
    def predict_from_url(self, image_url, top_k=5):
        """Predict food class from image URL"""
        try:
            data = {
                "image_url": image_url,
                "top_k": top_k
            }
            response = requests.post(f"{self.base_url}/predict-url", json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ URL prediction failed: {e}")
            return None
    
    def print_prediction(self, result):
        """Pretty print prediction results"""
        if not result or not result.get('success'):
            print("❌ No prediction results")
            return
        
        predictions = result.get('predictions', [])
        if not predictions:
            print("❌ No predictions found")
            return
        
        print("\n🍽️ Food Classification Results:")
        print("=" * 40)
        
        for i, pred in enumerate(predictions, 1):
            name = pred['class_name'].replace('_', ' ').title()
            confidence = pred['confidence_percentage']
            print(f"{i}. {name}: {confidence:.1f}%")
        
        top_pred = result.get('top_prediction')
        if top_pred:
            print(f"\n🏆 Top Prediction: {top_pred['class_name'].replace('_', ' ').title()}")
            print(f"   Confidence: {top_pred['confidence_percentage']:.1f}%")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CLI test client for the West African Food Classifier API"
    )

    parser.add_argument("target", nargs="?",
                        help="Image path, image URL, or directory when used with --dir")
    parser.add_argument("--url", action="store_true",
                        help="Interpret target as an image URL")
    parser.add_argument("--dir", action="store_true",
                        help="Interpret target as a directory and run batch predictions")
    parser.add_argument("--use-batch", action="store_true",
                        help="Use /predict-batch when running --dir")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="How many images to send per /predict-batch request")
    parser.add_argument("--recursive", action="store_true",
                        help="When used with --dir, include subdirectories")
    parser.add_argument("--health", action="store_true",
                        help="Run health check and exit")
    parser.add_argument("--classes", action="store_true",
                        help="List available food classes and exit")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to return (default: 5)")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the API (default: http://localhost:8000)")

    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    client = FoodClassifierClient(base_url=args.base_url)

    if args.health:
        print("🔍 Checking API health...")
        health = client.health_check()
        if health:
            print("✅ API is healthy!")
            print(f"   Model loaded: {health.get('model_loaded', 'Unknown')}")
            print(f"   Device: {health.get('device', 'Unknown')}")
            print(f"   Classes: {health.get('total_classes', 'Unknown')}")
        return

    if args.classes:
        print("📋 Getting available food classes...")
        classes = client.get_classes()
        if classes:
            print(f"✅ Found {classes['total_classes']} food classes:")
            for i, class_name in enumerate(classes['classes'][:20], 1):
                print(f"   {i}. {class_name.replace('_', ' ').title()}")
            if classes['total_classes'] > 20:
                remaining = classes['total_classes'] - 20
                print(f"   ... and {remaining} more")
        return

    if not args.target:
        print("❌ Missing target. Provide an image path, URL, or directory.")
        print("   Try: python test_client.py --help")
        sys.exit(1)

    if args.dir:
        if args.use_batch:
            client.predict_directory(
                args.target,
                top_k=args.top_k,
                recursive=args.recursive,
                batch_size=args.batch_size,
                use_batch=True,
            )
        else:
            client.predict_directory(args.target, top_k=args.top_k, recursive=args.recursive)
        return

    if args.url:
        print(f"🔍 Predicting from URL: {args.target}")
        result = client.predict_from_url(args.target, top_k=args.top_k)
        client.print_prediction(result)
        return

    # Default: treat target as image file path
    print(f"🔍 Predicting from file: {args.target}")
    result = client.predict_from_file(args.target, top_k=args.top_k)
    client.print_prediction(result)

if __name__ == "__main__":
    main()
