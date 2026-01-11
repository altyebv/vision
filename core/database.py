"""
SQLite database for storing transactions.
Lightweight and flexible schema.
"""
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from models import ExtractionResult, ValidationIssue


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str = "receipts.db"):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Main transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                receipt_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                overall_confidence REAL,
                needs_review INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Transaction fields (flexible for any field)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transaction_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                raw_value TEXT,
                ocr_confidence REAL,
                manually_edited INTEGER DEFAULT 0,
                edited_at TIMESTAMP,
                FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
            )
        """)

        # Validation issues
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                field_name TEXT,
                severity TEXT,
                message TEXT,
                resolved INTEGER DEFAULT 0,
                FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
            )
        """)

        # Known entities (for autocomplete)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                value TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified INTEGER DEFAULT 0,
                UNIQUE(entity_type, value)
            )
        """)

        # Processing stats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE DEFAULT CURRENT_DATE,
                total_processed INTEGER DEFAULT 0,
                auto_verified INTEGER DEFAULT 0,
                manually_reviewed INTEGER DEFAULT 0,
                avg_processing_time REAL,
                avg_confidence REAL,
                UNIQUE(date)
            )
        """)

        self.conn.commit()
        print(f"✓ Database initialized: {self.db_path}")

    def save_result(self, result: ExtractionResult) -> int:
        """
        Save extraction result to database.

        Args:
            result: ExtractionResult to save

        Returns:
            Transaction ID
        """
        cursor = self.conn.cursor()

        # Insert transaction
        cursor.execute("""
            INSERT INTO transactions 
            (filename, receipt_type, processed_at, overall_confidence, needs_review)
            VALUES (?, ?, ?, ?, ?)
        """, (
            result.filename,
            result.receipt_type.value,
            datetime.now(),
            result.overall_confidence,
            1 if result.needs_review else 0
        ))

        transaction_id = cursor.lastrowid

        # Insert fields
        for field_name in ['transaction_id', 'datetime', 'from_account',
                           'to_account', 'receiver_name', 'comment', 'amount']:
            field = getattr(result.data, field_name, None)

            if field:
                cursor.execute("""
                    INSERT INTO transaction_fields
                    (transaction_id, field_name, field_value, raw_value, ocr_confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    transaction_id,
                    field_name,
                    field.value,
                    field.raw_text,
                    field.confidence
                ))

                # Update known entities
                self._update_known_entity(field_name, field.value)

        # Insert validation issues
        for issue in result.issues:
            cursor.execute("""
                INSERT INTO validation_issues
                (transaction_id, field_name, severity, message)
                VALUES (?, ?, ?, ?)
            """, (
                transaction_id,
                issue.field,
                issue.severity,
                issue.message
            ))

        # Update stats
        self._update_stats(result)

        self.conn.commit()

        return transaction_id

    def save_batch(self, results: List[ExtractionResult]) -> List[int]:
        """
        Save multiple results.

        Args:
            results: List of ExtractionResult

        Returns:
            List of transaction IDs
        """
        ids = []
        for result in results:
            transaction_id = self.save_result(result)
            ids.append(transaction_id)

        return ids

    def get_transaction(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """Get transaction by ID."""
        cursor = self.conn.cursor()

        # Get transaction
        cursor.execute("""
            SELECT * FROM transactions WHERE id = ?
        """, (transaction_id,))

        row = cursor.fetchone()
        if not row:
            return None

        transaction = dict(row)

        # Get fields
        cursor.execute("""
            SELECT * FROM transaction_fields WHERE transaction_id = ?
        """, (transaction_id,))

        fields = {}
        for field_row in cursor.fetchall():
            field_dict = dict(field_row)
            fields[field_dict['field_name']] = field_dict

        transaction['fields'] = fields

        # Get issues
        cursor.execute("""
            SELECT * FROM validation_issues WHERE transaction_id = ?
        """, (transaction_id,))

        issues = [dict(row) for row in cursor.fetchall()]
        transaction['issues'] = issues

        return transaction

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all transactions that need review."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT id FROM transactions 
            WHERE needs_review = 1 AND status = 'pending'
            ORDER BY created_at DESC
        """)

        transactions = []
        for row in cursor.fetchall():
            transaction = self.get_transaction(row['id'])
            if transaction:
                transactions.append(transaction)

        return transactions

    def mark_reviewed(self, transaction_id: int):
        """Mark transaction as reviewed."""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE transactions 
            SET status = 'reviewed', needs_review = 0
            WHERE id = ?
        """, (transaction_id,))

        self.conn.commit()

    def update_field(self, transaction_id: int, field_name: str,
                     new_value: str):
        """Update a field value (manual edit)."""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE transaction_fields
            SET field_value = ?, manually_edited = 1, edited_at = ?
            WHERE transaction_id = ? AND field_name = ?
        """, (new_value, datetime.now(), transaction_id, field_name))

        self.conn.commit()

        # Update known entity
        self._update_known_entity(field_name, new_value, verified=True)

    def get_known_entities(self, entity_type: str,
                           limit: int = 10) -> List[str]:
        """
        Get known entities for autocomplete.

        Args:
            entity_type: Type of entity ('from_account', 'to_account', 'receiver_name')
            limit: Maximum number of results

        Returns:
            List of entity values
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT value FROM known_entities
            WHERE entity_type = ?
            ORDER BY frequency DESC, verified DESC
            LIMIT ?
        """, (entity_type, limit))

        return [row['value'] for row in cursor.fetchall()]

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get processing statistics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary of statistics
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                SUM(total_processed) as total,
                SUM(auto_verified) as auto_verified,
                SUM(manually_reviewed) as manually_reviewed,
                AVG(avg_confidence) as avg_confidence,
                AVG(avg_processing_time) as avg_time
            FROM processing_stats
            WHERE date >= date('now', '-' || ? || ' days')
        """, (days,))

        row = cursor.fetchone()

        return {
            'total_processed': row['total'] or 0,
            'auto_verified': row['auto_verified'] or 0,
            'manually_reviewed': row['manually_reviewed'] or 0,
            'avg_confidence': row['avg_confidence'] or 0.0,
            'avg_processing_time': row['avg_time'] or 0.0
        }

    def _update_known_entity(self, entity_type: str, value: str,
                             verified: bool = False):
        """Update known entity frequency."""
        if entity_type not in ['from_account', 'to_account', 'receiver_name']:
            return

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO known_entities (entity_type, value, verified)
            VALUES (?, ?, ?)
            ON CONFLICT(entity_type, value) DO UPDATE SET
                frequency = frequency + 1,
                last_seen = CURRENT_TIMESTAMP,
                verified = MAX(verified, excluded.verified)
        """, (entity_type, value, 1 if verified else 0))

    def _update_stats(self, result: ExtractionResult):
        """Update daily processing statistics."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO processing_stats 
            (date, total_processed, auto_verified, manually_reviewed,
             avg_processing_time, avg_confidence)
            VALUES (CURRENT_DATE, 1, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_processed = total_processed + 1,
                auto_verified = auto_verified + excluded.auto_verified,
                manually_reviewed = manually_reviewed + excluded.manually_reviewed,
                avg_processing_time = (avg_processing_time * total_processed + excluded.avg_processing_time) / (total_processed + 1),
                avg_confidence = (avg_confidence * total_processed + excluded.avg_confidence) / (total_processed + 1)
        """, (
            0 if result.needs_review else 1,
            1 if result.needs_review else 0,
            result.processing_time,
            result.overall_confidence
        ))

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup."""
        self.close()