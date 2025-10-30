"""
FastAPI backend for trading dashboard.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from datetime import datetime
import json

from database import get_session, Signal, Trade, Performance
from exchange import AlpacaExchange
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Trading Dashboard")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize exchange
exchange = AlpacaExchange(paper=True)


@app.get("/")
async def root():
    """Serve dashboard."""
    return HTMLResponse(content=open("dashboard/index.html").read())


@app.get("/api/status")
async def get_status():
    """Get system status."""
    try:
        account = exchange.get_account()
        position = exchange.get_position()
        price = exchange.get_latest_price()
        
        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "account": {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            },
            "position": {
                "has_position": position is not None,
                "quantity": float(position.qty) if position else 0,
                "side": position.side if position else None,
                "unrealized_pl": float(position.unrealized_pl) if position else 0,
            } if position else None,
            "current_price": price,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    """Get recent signals."""
    db = get_session()
    signals = db.query(Signal).order_by(Signal.timestamp.desc()).limit(limit).all()
    db.close()
    
    return [
        {
            "id": s.id,
            "timestamp": s.timestamp.isoformat(),
            "signal": s.signal,
            "probability": s.probability,
            "confidence": s.confidence,
            "price": s.price,
        }
        for s in signals
    ]


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    db = get_session()
    trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
    db.close()
    
    return [
        {
            "id": t.id,
            "timestamp": t.timestamp.isoformat(),
            "side": t.side,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "status": t.status,
        }
        for t in trades
    ]


@app.get("/api/performance")
async def get_performance():
    """Get performance metrics."""
    db = get_session()
    
    # Get all closed trades
    trades = db.query(Trade).filter_by(status="closed").all()
    
    if not trades:
        db.close()
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
        }
    
    winning = [t for t in trades if t.pnl and t.pnl > 0]
    total_pnl = sum(t.pnl for t in trades if t.pnl)
    
    db.close()
    
    return {
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(trades) - len(winning),
        "win_rate": len(winning) / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "avg_win": sum(t.pnl for t in winning) / len(winning) if winning else 0,
        "avg_loss": sum(t.pnl for t in trades if t.pnl and t.pnl < 0) / max(1, len(trades) - len(winning)),
    }


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send live data every 2 seconds
            data = {
                "timestamp": datetime.now().isoformat(),
                "price": exchange.get_latest_price(),
                "account": {
                    "equity": float(exchange.get_account().equity),
                },
            }
            
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

