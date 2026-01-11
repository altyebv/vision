"""
FastAPI server for receipt processing.
FIXED: Aligned endpoints with frontend API client.
"""
import config  # Import first to fix OpenMP
import io
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import pandas as pd

from processor import ReceiptProcessor
from database import Database
from models import ExtractionResult, ReceiptType, ValidationIssue

# Initialize
app = FastAPI(
    title="Receipt OCR API",
    version="1.0.0",
    description="Automated receipt data extraction system"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor (lazy loading)
processor: Optional[ReceiptProcessor] = None
database: Optional[Database] = None


def get_processor():
    """Get or initialize processor."""
    global processor
    if processor is None:
        print("Initializing processor...")
        processor = ReceiptProcessor(ocr_engine="auto")
    return processor


def get_database():
    """Get or initialize database."""
    global database
    if database is None:
        print("Initializing database...")
        database = Database("receipts.db")
    return database


# === Response Models ===

class FieldData(BaseModel):
    value: str
    confidence: float
    raw_text: str
    needs_review: bool


class IssueData(BaseModel):
    field: str
    severity: str
    message: str


class ExtractionResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    receipt_type: str
    data: dict
    issues: List[IssueData]
    overall_confidence: float
    needs_review: bool
    processing_time: float


class BatchResponse(BaseModel):
    total: int
    successful: int
    needs_review: int
    avg_confidence: float
    results: List[ExtractionResponse]


class StatsResponse(BaseModel):
    total_processed: int
    auto_verified: int
    manually_reviewed: int
    avg_confidence: float
    avg_processing_time: float
    total_amount: Optional[float] = None


class KnownEntityResponse(BaseModel):
    value: str
    display_name: str
    frequency: int
    verified: bool


# === Helper Functions ===

def convert_result_to_response(result: ExtractionResult, db_id: Optional[int] = None) -> ExtractionResponse:
    """Convert ExtractionResult to API response."""
    data_dict = {}

    for field_name in ['transaction_id', 'datetime', 'from_account',
                       'to_account', 'receiver_name', 'comment', 'amount']:
        field_result = getattr(result.data, field_name, None)
        if field_result:
            data_dict[field_name] = {
                'value': field_result.value,
                'confidence': field_result.confidence,
                'raw_text': field_result.raw_text,
                'needs_review': field_result.needs_review
            }
        else:
            data_dict[field_name] = None

    issues_list = [
        IssueData(
            field=issue.field,
            severity=issue.severity,
            message=issue.message
        )
        for issue in result.issues
    ]

    return ExtractionResponse(
        id=db_id,
        filename=result.filename,
        receipt_type=result.receipt_type.value,
        data=data_dict,
        issues=issues_list,
        overall_confidence=result.overall_confidence,
        needs_review=result.needs_review,
        processing_time=result.processing_time
    )


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Cannot decode image")

    return image


# === API Endpoints ===

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Receipt OCR API",
        "version": "1.0.0",
        "endpoints": {
            "extract_single": "/api/extract/single",
            "extract_batch": "/api/extract/batch",
            "receipts": "/api/receipts",
            "transactions": "/api/transactions",
            "accounts": "/api/accounts",
            "stats": "/api/statistics",
            "docs": "/docs"
        }
    }


# === OCR Extraction Endpoints ===

@app.post("/api/extract/single", response_model=ExtractionResponse)
async def extract_single_receipt(
        file: UploadFile = File(...),
        save_to_db: bool = Query(False, description="Save result to database")
):
    """
    Extract data from a single receipt image (without saving).

    Args:
        file: Receipt image (PNG, JPG, JPEG)
        save_to_db: Whether to save to database

    Returns:
        Extracted transaction data with confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")

        # Load image
        image = load_image_from_upload(file)

        # Get processor
        proc = get_processor()

        # Save temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename

        cv2.imwrite(str(temp_path), image)

        # Process
        result = proc.process(temp_path)

        # Save to database if requested
        db_id = None
        if save_to_db:
            db = get_database()
            db_id = db.save_result(result)

        # Cleanup
        temp_path.unlink(missing_ok=True)

        return convert_result_to_response(result, db_id)

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/api/extract/batch", response_model=BatchResponse)
async def extract_batch_receipts(
        files: List[UploadFile] = File(...),
        save_to_db: bool = Query(False, description="Save results to database")
):
    """
    Extract data from multiple receipt images.

    Args:
        files: List of receipt images (max 50)
        save_to_db: Whether to save to database

    Returns:
        Batch processing results with statistics
    """
    try:
        if len(files) > 50:
            raise HTTPException(400, "Maximum 50 files per batch")

        # Get processor
        proc = get_processor()

        # Create temp directory
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        # Save all files
        temp_paths = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue

            temp_path = temp_dir / file.filename

            # Save file
            with open(temp_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)

            temp_paths.append(temp_path)

        # Process batch
        results = proc.process_batch(temp_paths)

        # Save to database if requested
        db_ids = []
        if save_to_db:
            db = get_database()
            db_ids = db.save_batch(results)

        # Convert results
        response_results = [
            convert_result_to_response(r, db_id)
            for r, db_id in zip(results, db_ids if db_ids else [None] * len(results))
        ]

        # Cleanup
        for path in temp_paths:
            path.unlink(missing_ok=True)

        # Calculate stats
        successful = sum(1 for r in results if not r.needs_review)
        needs_review = len(results) - successful
        avg_conf = sum(r.overall_confidence for r in results) / len(results) if results else 0.0

        return BatchResponse(
            total=len(results),
            successful=successful,
            needs_review=needs_review,
            avg_confidence=avg_conf,
            results=response_results
        )

    except Exception as e:
        raise HTTPException(500, f"Batch processing failed: {str(e)}")


# === Receipt Management Endpoints (matches frontend API) ===

@app.post("/api/receipts")
async def save_confirmed_receipt(data: dict):
    """
    Save confirmed receipt data to database.
    This is called after user reviews and confirms the data.

    Args:
        data: Confirmed receipt data

    Returns:
        Save result with transaction ID
    """
    try:
        db = get_database()

        # Create a minimal ExtractionResult from the confirmed data
        from models import TransactionData, FieldResult

        field_results = {}
        for field_name in ['transaction_id', 'datetime', 'from_account',
                          'to_account', 'receiver_name', 'comment', 'amount']:
            if field_name in data and data[field_name]:
                value = data[field_name]
                if isinstance(value, dict):
                    field_results[field_name] = FieldResult(
                        value=value.get('value', ''),
                        confidence=value.get('confidence', 1.0),
                        raw_text=value.get('raw_text', ''),
                        needs_review=False
                    )
                else:
                    field_results[field_name] = FieldResult(
                        value=str(value),
                        confidence=1.0,
                        raw_text=str(value),
                        needs_review=False
                    )

        transaction_data = TransactionData(**field_results)

        result = ExtractionResult(
            filename=data.get('filename', 'manual_entry.jpg'),
            receipt_type=ReceiptType.GREEN,
            data=transaction_data,
            issues=[],
            overall_confidence=1.0,
            needs_review=False,
            processing_time=0.0
        )

        transaction_id = db.save_result(result)

        return {
            "status": "ok",
            "message": "Receipt saved successfully",
            "id": transaction_id
        }

    except Exception as e:
        raise HTTPException(500, f"Save failed: {str(e)}")


@app.get("/api/receipts/check-duplicate/{transaction_id}")
async def check_duplicate_receipt(transaction_id: str):
    """
    Check if transaction ID already exists.

    Args:
        transaction_id: Transaction ID to check

    Returns:
        {"exists": bool, "transaction_id": str}
    """
    try:
        db = get_database()
        exists = db.check_duplicate(transaction_id)

        return {
            "exists": exists,
            "transaction_id": transaction_id
        }

    except Exception as e:
        raise HTTPException(500, f"Duplicate check failed: {str(e)}")


@app.put("/api/receipts/{receipt_id}")
async def update_receipt(receipt_id: int, data: dict):
    """
    Update receipt data after manual review.

    Args:
        receipt_id: Database ID
        data: Updated field values

    Returns:
        Success message
    """
    try:
        db = get_database()
        db.update_receipt(receipt_id, data)

        return {
            "status": "ok",
            "message": "Receipt updated successfully",
            "id": receipt_id
        }

    except Exception as e:
        raise HTTPException(500, f"Update failed: {str(e)}")


@app.delete("/api/receipts/{receipt_id}")
async def delete_receipt(receipt_id: int):
    """Delete a receipt from database."""
    try:
        db = get_database()
        db.delete_receipt(receipt_id)

        return {
            "status": "ok",
            "message": "Receipt deleted successfully",
            "id": receipt_id
        }

    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")


# === Account Management Endpoints (for autocomplete) ===

@app.get("/api/accounts/known", response_model=List[KnownEntityResponse])
async def get_known_accounts():
    """
    Get all known accounts for autocomplete.
    Returns both from_account and to_account entities.

    Returns:
        List of known accounts with frequencies
    """
    try:
        db = get_database()

        # Get known from_accounts
        from_accounts = db.get_known_entities('from_account', limit=50)

        # Get known to_accounts
        to_accounts = db.get_known_entities('to_account', limit=50)

        # Combine and deduplicate
        all_accounts = {}
        for acc in from_accounts + to_accounts:
            key = acc['value']
            if key not in all_accounts or acc['verified']:
                all_accounts[key] = acc

        # Convert to response format
        results = [
            KnownEntityResponse(**acc)
            for acc in all_accounts.values()
        ]

        # Sort by verified, then frequency
        results.sort(key=lambda x: (not x.verified, -x.frequency))

        return results

    except Exception as e:
        raise HTTPException(500, f"Failed to fetch accounts: {str(e)}")


@app.get("/api/accounts/search", response_model=List[KnownEntityResponse])
async def search_accounts(q: str = Query(..., min_length=1)):
    """
    Search for accounts by number or name.

    Args:
        q: Search query

    Returns:
        Matching accounts
    """
    try:
        db = get_database()

        # Search in both from_account and to_account
        from_results = db.search_entities('from_account', q, limit=25)
        to_results = db.search_entities('to_account', q, limit=25)

        # Search in receiver names (they're linked to accounts)
        name_results = db.search_entities('receiver_name', q, limit=25)

        # Combine and deduplicate
        all_results = {}
        for result in from_results + to_results:
            key = result['value']
            if key not in all_results:
                all_results[key] = result

        # Add names with their linked accounts
        for name in name_results:
            # The display_name stores the account number for names
            if name['display_name'] and name['display_name'] != name['value']:
                account_num = name['display_name']
                if account_num not in all_results:
                    all_results[account_num] = {
                        'value': account_num,
                        'display_name': name['value'],  # Use name as display
                        'frequency': name['frequency'],
                        'verified': name['verified']
                    }

        results = [KnownEntityResponse(**acc) for acc in all_results.values()]
        results.sort(key=lambda x: (not x.verified, -x.frequency))

        return results

    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


# === Transaction Query Endpoints ===

@app.get("/api/transactions")
async def query_transactions(
    from_account: Optional[str] = None,
    to_account: Optional[str] = None,
    receiver_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
):
    """
    Query transactions with filters.

    Args:
        from_account: Filter by from account
        to_account: Filter by to account
        receiver_name: Filter by receiver name
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        min_amount: Minimum amount
        max_amount: Maximum amount

    Returns:
        List of matching transactions
    """
    try:
        db = get_database()

        filters = {
            'from_account': from_account,
            'to_account': to_account,
            'receiver_name': receiver_name,
            'date_from': date_from,
            'date_to': date_to,
            'min_amount': min_amount,
            'max_amount': max_amount
        }

        # Remove None values
        filters = {k: v for k, v in filters.items() if v is not None}

        transactions = db.query_transactions(filters)

        return {
            "total": len(transactions),
            "transactions": transactions
        }

    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.get("/api/transactions/{transaction_id}")
async def get_transaction_by_id(transaction_id: str):
    """
    Get transaction by transaction ID (not database ID).

    Args:
        transaction_id: Transaction ID from receipt

    Returns:
        Transaction data
    """
    try:
        db = get_database()

        # Search for transaction by transaction_id field
        filters = {'transaction_id': transaction_id}
        results = db.query_transactions(filters)

        if not results:
            raise HTTPException(404, f"Transaction {transaction_id} not found")

        return results[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch transaction: {str(e)}")


@app.get("/api/transactions/export")
async def export_transactions(
    format: str = Query('json', regex='^(json|csv)$'),
    from_account: Optional[str] = None,
    to_account: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """
    Export transactions to JSON or CSV.

    Args:
        format: 'json' or 'csv'
        from_account: Filter by from account
        to_account: Filter by to account
        date_from: Start date
        date_to: End date

    Returns:
        File download
    """
    try:
        db = get_database()

        filters = {
            'from_account': from_account,
            'to_account': to_account,
            'date_from': date_from,
            'date_to': date_to
        }
        filters = {k: v for k, v in filters.items() if v is not None}

        transactions = db.query_transactions(filters)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == 'json':
            filename = f"transactions_{timestamp}.json"
            content = json.dumps(transactions, indent=2, default=str)

            return StreamingResponse(
                io.BytesIO(content.encode()),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        else:  # csv
            # Flatten transactions for CSV
            flattened = []
            for trans in transactions:
                flat = {
                    'id': trans['id'],
                    'filename': trans['filename'],
                    'receipt_type': trans['receipt_type'],
                    'created_at': trans['created_at']
                }

                # Add field values
                for field_name, field_data in trans.get('fields', {}).items():
                    flat[field_name] = field_data.get('field_value', '')

                flattened.append(flat)

            df = pd.DataFrame(flattened)

            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            filename = f"transactions_{timestamp}.csv"

            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")


# === Statistics Endpoints ===

@app.get("/api/statistics", response_model=StatsResponse)
async def get_statistics(days: int = Query(30, description="Number of days")):
    """
    Get dashboard statistics.

    Args:
        days: Number of days to include (default: 30)

    Returns:
        Statistics summary
    """
    try:
        db = get_database()
        stats = db.get_stats(days=days)

        return StatsResponse(**stats)

    except Exception as e:
        raise HTTPException(500, f"Failed to fetch stats: {str(e)}")


# === Pending Reviews (for review workflow) ===

@app.get("/api/pending")
async def get_pending_reviews():
    """Get all receipts pending manual review."""
    try:
        db = get_database()
        pending = db.get_pending_reviews()

        return {
            "total": len(pending),
            "receipts": pending
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to fetch pending reviews: {str(e)}")


# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# === Startup/Shutdown ===

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("=" * 60)
    print("Receipt OCR API Server")
    print("=" * 60)
    print(f"Starting up...")
    print(f"API docs: http://localhost:8000/docs")
    print(f"Interactive docs: http://localhost:8000/redoc")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global database
    if database:
        database.close()
    print("\nServer shutdown complete.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )