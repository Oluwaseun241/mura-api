#!/usr/bin/env python3
"""
Upload images to Cloudinary for cloud storage
Organizes images by class name in folders
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Cloudinary free tier allows ~500 API calls per hour (~8.3/minute)
DEFAULT_DELAY_SECONDS = 8
MAX_RETRIES = 3
RETRY_DELAY_BASE = 60  # Start with 1 minute for rate limit errors

def upload_to_cloudinary(images_dir="images", folder_prefix="food-classifier", 
                         cloud_name=None, api_key=None, api_secret=None,
                         delay_seconds=DEFAULT_DELAY_SECONDS, max_retries=MAX_RETRIES):
    """
    Upload images from local directory to Cloudinary
    
    Args:
        images_dir: Local directory containing class folders
        folder_prefix: Prefix for Cloudinary folder structure
        cloud_name: Cloudinary cloud name
        api_key: Cloudinary API key
        api_secret: Cloudinary API secret
        delay_seconds: Delay between uploads to respect rate limits (default: 8)
        max_retries: Maximum retries for rate limit errors (default: 3)
    """
    try:
        import cloudinary
        import cloudinary.uploader
        import cloudinary.api
        
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret
        )
        
        try:
            cloudinary.api.ping()
            logger.info("✅ Connected to Cloudinary successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Cloudinary: {e}")
            return False
        
        if not os.path.exists(images_dir):
            logger.error(f"❌ Images directory not found: {images_dir}")
            return False
        
        uploaded_count = 0
        skipped_count = 0
        error_count = 0
        rate_limited_count = 0
        api_calls_made = 0
        
        class_folders = sorted([f for f in os.listdir(images_dir) 
                               if os.path.isdir(os.path.join(images_dir, f))])
        
        logger.info(f"📁 Found {len(class_folders)} class folders")
        logger.info(f"⏱️  Using {delay_seconds}s delay between uploads to respect rate limits")
        
        for class_folder in class_folders:
            class_path = os.path.join(images_dir, class_folder)
            logger.info(f"\n📦 Uploading class: {class_folder}")
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            
            for filename in image_files:
                file_path = os.path.join(class_path, filename)
                
                image_name = os.path.splitext(filename)[0]
                public_id = f"{folder_prefix}/{class_folder}/{image_name}"
                
                exists = False
                for retry in range(max_retries):
                    try:
                        existing = cloudinary.api.resource(public_id)
                        exists = True
                        api_calls_made += 1
                        break
                    except cloudinary.api.NotFound:
                        exists = False
                        api_calls_made += 1
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if "420" in error_msg or "Rate Limit" in error_msg:
                            rate_limited_count += 1
                            wait_time = RETRY_DELAY_BASE * (2 ** retry)  # Exponential backoff
                            logger.warning(f"   ⚠️  Rate limited checking {filename}. Waiting {wait_time}s before retry {retry+1}/{max_retries}...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"   ❌ Error checking {filename}: {e}")
                            break
                
                if exists:
                    logger.debug(f"⏭️  Skipping {filename} (already exists)")
                    skipped_count += 1
                    time.sleep(delay_seconds)
                    continue
                
                uploaded = False
                for retry in range(max_retries):
                    try:
                        result = cloudinary.uploader.upload(
                            file_path,
                            public_id=public_id,
                            folder=f"{folder_prefix}/{class_folder}",
                            overwrite=False,
                            resource_type="image"
                        )
                        uploaded = True
                        api_calls_made += 1
                        uploaded_count += 1
                        if uploaded_count % 50 == 0:
                            logger.info(f"   ✅ Uploaded {uploaded_count} images so far... (API calls: {api_calls_made})")
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if "420" in error_msg or "Rate Limit" in error_msg:
                            rate_limited_count += 1
                            wait_time = RETRY_DELAY_BASE * (2 ** retry)  # Exponential backoff
                            
                            # Try to extract retry time from error message
                            if "Try again on" in error_msg:
                                try:
                                    # Extract timestamp from error message
                                    time_str = error_msg.split("Try again on ")[1].strip()
                                    retry_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S %Z")
                                    wait_time = max(wait_time, (retry_time - datetime.utcnow()).total_seconds())
                                except:
                                    pass
                            
                            if retry < max_retries - 1:
                                logger.warning(f"   ⚠️  Rate limited uploading {filename}. Waiting {int(wait_time)}s before retry {retry+1}/{max_retries}...")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"   ❌ Rate limit exceeded for {filename} after {max_retries} retries. Will need to resume later.")
                                error_count += 1
                        else:
                            logger.error(f"   ❌ Failed to upload {filename}: {e}")
                            error_count += 1
                            break
                
                if uploaded or exists:
                    time.sleep(delay_seconds)
        
        logger.info("\n" + "="*50)
        logger.info("📊 Upload Summary:")
        logger.info(f"   ✅ Successfully uploaded: {uploaded_count}")
        logger.info(f"   ⏭️  Skipped (already exists): {skipped_count}")
        logger.info(f"   ⚠️  Rate limited: {rate_limited_count}")
        logger.info(f"   ❌ Errors: {error_count}")
        logger.info(f"   📊 Total API calls: {api_calls_made}")
        logger.info(f"   📁 Total classes: {len(class_folders)}")
        logger.info("="*50)
        
        if error_count > 0 and rate_limited_count > 0:
            logger.warning("\n⚠️  Upload incomplete due to rate limits.")
            logger.warning("💡 You can run this script again - it will skip already uploaded images.")
            logger.warning(f"💡 Rate limit resets hourly. Try again in about an hour.")
        
        if uploaded_count > 0 or skipped_count > 0:
            logger.info("\n🎉 Upload session completed!")
            logger.info(f"💡 Your images are now available at: https://res.cloudinary.com/{cloud_name}/image/upload/{folder_prefix}/")
            if error_count == 0:
                return True
            else:
                logger.info("💡 Some images failed. Run the script again to retry failed uploads.")
                return False
        else:
            logger.warning("⚠️  No images were uploaded")
            return False
            
    except ImportError:
        logger.error("❌ Cloudinary library not installed")
        logger.error("   Install it with: pip install cloudinary")
        return False
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Upload images to Cloudinary for cloud storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using .env file (recommended)
  # Create .env file with:
  #   CLOUDINARY_CLOUD_NAME=your_cloud_name
  #   CLOUDINARY_API_KEY=your_api_key
  #   CLOUDINARY_API_SECRET=your_api_secret
  python upload_to_cloudinary.py

  # Using environment variables
  export CLOUDINARY_CLOUD_NAME=your_cloud_name
  export CLOUDINARY_API_KEY=your_api_key
  export CLOUDINARY_API_SECRET=your_api_secret
  python upload_to_cloudinary.py

  # Using command line arguments
  python upload_to_cloudinary.py \\
    --cloud-name your_cloud_name \\
    --api-key your_api_key \\
    --api-secret your_api_secret \\
    --folder food-classifier

  # Upload to specific folder
  python upload_to_cloudinary.py --folder my-food-images

  # Adjust delay for rate limiting (if you have paid tier)
  python upload_to_cloudinary.py --delay 2

  # Resume upload after rate limit (skips already uploaded images)
  python upload_to_cloudinary.py
        """
    )
    
    parser.add_argument("--images-dir", default="images",
                       help="Local images directory (default: images)")
    parser.add_argument("--folder", default="food-classifier",
                       help="Cloudinary folder prefix (default: food-classifier)")
    parser.add_argument("--cloud-name", 
                       help="Cloudinary cloud name (or set in .env file as CLOUDINARY_CLOUD_NAME)")
    parser.add_argument("--api-key",
                       help="Cloudinary API key (or set in .env file as CLOUDINARY_API_KEY)")
    parser.add_argument("--api-secret",
                       help="Cloudinary API secret (or set in .env file as CLOUDINARY_API_SECRET)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                       help=f"Delay in seconds between uploads (default: {DEFAULT_DELAY_SECONDS}, recommended: 8-10 for free tier)")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                       help=f"Maximum retries for rate limit errors (default: {MAX_RETRIES})")
    
    args = parser.parse_args()
    
    # Get credentials from environment or arguments
    cloud_name = args.cloud_name or os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = args.api_key or os.getenv("CLOUDINARY_API_KEY")
    api_secret = args.api_secret or os.getenv("CLOUDINARY_API_SECRET")
    
    # Validate credentials
    if not cloud_name:
        logger.error("❌ Cloudinary cloud name not provided")
        logger.error("   Add CLOUDINARY_CLOUD_NAME to .env file or use --cloud-name")
        sys.exit(1)
    
    if not api_key:
        logger.error("❌ Cloudinary API key not provided")
        logger.error("   Add CLOUDINARY_API_KEY to .env file or use --api-key")
        sys.exit(1)
    
    if not api_secret:
        logger.error("❌ Cloudinary API secret not provided")
        logger.error("   Add CLOUDINARY_API_SECRET to .env file or use --api-secret")
        sys.exit(1)
    
    # Upload images
    success = upload_to_cloudinary(
        images_dir=args.images_dir,
        folder_prefix=args.folder,
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        delay_seconds=args.delay,
        max_retries=args.max_retries
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
