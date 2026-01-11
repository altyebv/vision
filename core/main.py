"""
Simple test interface for receipt OCR system.
"""
import config  # Import first to fix OpenMP
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

from processor import ReceiptProcessor
from database import Database


def test_single(image_path: str):
    """Test single receipt."""
    print("\n" + "=" * 60)
    print("SINGLE RECEIPT TEST")
    print("=" * 60 + "\n")

    processor = ReceiptProcessor(ocr_engine="auto")
    result = processor.process(image_path)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFilename: {result.filename}")
    print(f"Type: {result.receipt_type.value}")
    print(f"Confidence: {result.overall_confidence:.2%}")
    print(f"Status: {'⚠️  NEEDS REVIEW' if result.needs_review else '✅ OK'}")

    print("\nExtracted Fields:")
    print("-" * 60)

    for field in ['transaction_id', 'datetime', 'from_account',
                  'to_account', 'receiver_name', 'comment', 'amount']:
        value = result.data.get_value(field)
        conf = result.data.get_confidence(field)

        if value:
            flag = "⚠️ " if conf < 0.90 else "✅"
            print(f"  {flag} {field:18s}: {value:30s} ({conf:.2%})")
        else:
            print(f"  ❌ {field:18s}: [NOT EXTRACTED]")

    if result.issues:
        print("\nIssues:")
        print("-" * 60)
        for issue in result.issues:
            symbol = "❌" if issue.severity == "error" else "⚠️ "
            print(f"  {symbol} {issue.field}: {issue.message}")

    print(f"\nProcessing time: {result.processing_time:.2f}s\n")

    return result


def test_batch(directory: str):
    """Test batch processing."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING TEST")
    print("=" * 60 + "\n")

    processor = ReceiptProcessor(ocr_engine="auto")
    results = processor.process_directory(directory)

    if not results:
        print("No results to display\n")
        return results

    # Display summary
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60 + "\n")

    for result in results:
        status = "⚠️  REVIEW" if result.needs_review else "✅ OK"
        print(f"{result.filename:30s} | {status} | Conf: {result.overall_confidence:.2%}")

    return results


def save_to_db(results, db_path: str = "receipts.db"):
    """Save results to database."""
    print(f"\nSaving to database: {db_path}")

    db = Database(db_path)
    transaction_ids = db.save_batch(results)

    print(f"✓ Saved {len(transaction_ids)} transactions")

    # Show stats
    stats = db.get_stats(days=1)
    print(f"\nToday's Stats:")
    print(f"  Processed: {stats['total_processed']}")
    print(f"  Auto-verified: {stats['auto_verified']}")
    print(f"  Need review: {stats['manually_reviewed']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2%}")

    db.close()


def export_to_excel(results, output_path: str = None):
    """Export results to Excel."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"receipts_{timestamp}.xlsx"

    print(f"\nExporting to Excel: {output_path}")

    # Convert to DataFrame
    data = [result.to_dict() for result in results]
    df = pd.DataFrame(data)

    # Reorder columns
    columns = [
        'filename', 'receipt_type', 'transaction_id', 'datetime',
        'from_account', 'to_account', 'receiver_name', 'comment',
        'amount', 'confidence', 'needs_review', 'issues'
    ]
    df = df[[col for col in columns if col in df.columns]]

    # Export
    df.to_excel(output_path, index=False, sheet_name='Receipts')

    print(f"✓ Exported {len(df)} receipts to {output_path}\n")


def interactive_menu():
    """Interactive menu."""
    while True:
        print("\n" + "=" * 60)
        print("RECEIPT OCR SYSTEM")
        print("=" * 60)
        print("\n1. Test single receipt")
        print("2. Test batch (directory)")
        print("3. View pending reviews")
        print("4. Exit")

        choice = input("\nChoice (1-4): ").strip()

        if choice == "1":
            path = input("Enter image path: ").strip()
            if Path(path).exists():
                result = test_single(path)

                save = input("\nSave to database? (y/n): ").strip().lower()
                if save == 'y':
                    save_to_db([result])
            else:
                print(f"File not found: {path}")

        elif choice == "2":
            path = input("Enter directory path: ").strip()
            if Path(path).exists():
                results = test_batch(path)

                if results:
                    save = input("\nSave to database? (y/n): ").strip().lower()
                    if save == 'y':
                        save_to_db(results)

                    export = input("Export to Excel? (y/n): ").strip().lower()
                    if export == 'y':
                        export_to_excel(results)
            else:
                print(f"Directory not found: {path}")

        elif choice == "3":
            db = Database()
            pending = db.get_pending_reviews()

            if not pending:
                print("\n✓ No pending reviews!")
            else:
                print(f"\n{len(pending)} transactions need review:")
                for t in pending:
                    print(f"  ID {t['id']}: {t['filename']}")

            db.close()

        elif choice == "4":
            print("\nGoodbye!\n")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1]

        if command == "single" and len(sys.argv) > 2:
            result = test_single(sys.argv[2])
            save_to_db([result])

        elif command == "batch" and len(sys.argv) > 2:
            results = test_batch(sys.argv[2])
            if results:
                save_to_db(results)
                export_to_excel(results)

        else:
            print("Usage:")
            print("  python main.py single <image_path>")
            print("  python main.py batch <directory_path>")
            print("  python main.py  (interactive mode)")

    else:
        # Interactive mode
        interactive_menu()