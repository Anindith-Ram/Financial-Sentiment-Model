#!/usr/bin/env python3
"""
Backup Management Script
Check and manage data collection backup files
"""
import os
import glob
from datetime import datetime

def check_backups():
    """Check current backup files"""
    print("📁 Backup File Status")
    print("=" * 50)
    
    # Check main data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("[ERROR] Data directory not found")
        return
    
    # Check main dataset
    main_file = os.path.join(data_dir, "candles.csv")
    if os.path.exists(main_file):
        size_mb = os.path.getsize(main_file) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(main_file))
                    print(f"[SUCCESS] Main dataset: candles.csv ({size_mb:.1f} MB, modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("[ERROR] Main dataset not found")
    
    # Check backup files
    backup_pattern = os.path.join(data_dir, "candles_backup_*.csv")
    backup_files = glob.glob(backup_pattern)
    
    if backup_files:
        print(f"\n📦 Found {len(backup_files)} backup files:")
        for backup in sorted(backup_files, key=lambda x: os.path.getmtime(x), reverse=True):
            size_mb = os.path.getsize(backup) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(backup))
            filename = os.path.basename(backup)
            print(f"  • {filename} ({size_mb:.1f} MB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("\n📦 No dated backup files found")
    
    # Check old-style backup
    old_backup = os.path.join(data_dir, "candles_backup.csv")
    if os.path.exists(old_backup):
        size_mb = os.path.getsize(old_backup) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(old_backup))
                    print(f"\n[WARNING] Old-style backup found: candles_backup.csv ({size_mb:.1f} MB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
        print("   Consider renaming this to candles.csv if you want to use it as your main dataset")
    
    # Check feature summary
    summary_file = os.path.join(data_dir, "candles_feature_summary.txt")
    if os.path.exists(summary_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(summary_file))
                    print(f"\n[CHART] Feature summary: candles_feature_summary.txt (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    print("\n" + "=" * 50)
    print("💡 Next time you run data collection, backups will include timestamps!")
    print("   Example: candles_backup_20240115_143022.csv")


def cleanup_old_backups(keep_recent=3):
    """Clean up old backup files"""
    print(f"🧹 Cleaning up old backups (keeping {keep_recent} most recent)...")
    
    data_dir = "data"
    backup_pattern = os.path.join(data_dir, "candles_backup_*.csv")
    backup_files = glob.glob(backup_pattern)
    
    if len(backup_files) <= keep_recent:
        print("[SUCCESS] No cleanup needed - backup count is within limit")
        return
    
    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Remove old backups
    removed_count = 0
    for old_backup in backup_files[keep_recent:]:
        try:
            os.remove(old_backup)
            filename = os.path.basename(old_backup)
            print(f"🗑️ Removed: {filename}")
            removed_count += 1
        except Exception as e:
            print(f"[WARNING] Could not remove {old_backup}: {e}")
    
            print(f"[SUCCESS] Cleanup complete: removed {removed_count} old backup files")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_old_backups()
    else:
        check_backups() 