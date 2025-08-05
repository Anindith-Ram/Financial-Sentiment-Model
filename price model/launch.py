#!/usr/bin/env python3
"""
Simple Launcher for Price Model Pipeline
Easy-to-use interface for all pipeline functions
"""
import sys
import os
from datetime import datetime

def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("🚀 PRICE MODEL PIPELINE LAUNCHER")
    print("="*60)
    print("Choose an option:")
    print("1. 📊 Build Dataset")
    print("2. 🚀 Progressive Training (Optimized)")
    print("3. 🔮 Predict Patterns")
    print("4. 💰 Generate Trading Signals")
    print("5. 🎬 Run Complete Demo")
    print("6. 📖 Show Help")
    print("7. 🚪 Exit")
    print("="*60)

def run_command(command):
    """Run a command and handle errors"""
    try:
        os.system(command)
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    """Main launcher function"""
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                print("\n📊 Building dataset...")
                run_command("python main.py --mode data")
                
            elif choice == "2":
                experiment = input("Enter experiment name (or press Enter for default): ").strip()
                if not experiment:
                    experiment = f"progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"\n🚀 Running progressive training: {experiment}")
                run_command(f"python main.py --mode progressive --experiment-name {experiment}")
                
            elif choice == "3":
                ticker = input("Enter ticker symbol (or press Enter for AAPL): ").strip()
                if not ticker:
                    ticker = "AAPL"
                print(f"\n🔮 Predicting patterns for {ticker}")
                run_command(f"python main.py --mode predict --ticker {ticker}")
                
            elif choice == "4":
                portfolio = input("Enter portfolio tickers (space-separated, or press Enter for default): ").strip()
                if not portfolio:
                    portfolio = "AAPL MSFT GOOGL TSLA AMZN"
                print(f"\n💰 Generating trading signals for: {portfolio}")
                run_command(f"python main.py --mode signals --portfolio {portfolio}")
                
            elif choice == "5":
                print("\n🎬 Running complete demo...")
                run_command("python main.py --mode demo")
                
            elif choice == "6":
                print("\n📖 Showing help...")
                run_command("python main.py --mode help")
                
            elif choice == "7":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-7.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 