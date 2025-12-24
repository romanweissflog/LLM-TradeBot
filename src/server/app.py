from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from src.server.state import global_state

# Input Model
from pydantic import BaseModel
class ControlCommand(BaseModel):
    action: str  # start, pause, stop, restart, set_interval
    interval: float = None  # Optional: interval in minutes for set_interval action

from fastapi import UploadFile, File
import shutil

app = FastAPI(title="LLM-TradeBot Dashboard")

# Enable CORS (rest unchanged)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get absolute path to the web directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIR = os.path.join(BASE_DIR, 'web')

import math

def clean_nans(obj):
    """Recursively replace NaN/Inf with None for JSON compliance"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(i) for i in obj]
    return obj

# API Endpoints
@app.get("/api/status")
async def get_status():
    data = {
        "system": {
            "running": global_state.is_running,
            "mode": global_state.execution_mode,
            "is_test_mode": global_state.is_test_mode,
            "cycle_counter": global_state.cycle_counter,
            "cycle_interval": global_state.cycle_interval,  # 添加当前间隔
            "current_cycle_id": global_state.current_cycle_id,
            "uptime_start": global_state.start_time,
            "last_heartbeat": global_state.last_update
        },
        "market": {
            "price": global_state.current_price,
            "regime": global_state.market_regime,
            "position": global_state.price_position
        },
        "agents": {
            "strategist_score": global_state.strategist_score,
            "critic_confidence": global_state.critic_confidence,
            "guardian_status": global_state.guardian_status
        },
        "account": global_state.account_overview,
        "virtual_account": {
            "is_test_mode": global_state.is_test_mode,
            "initial_balance": global_state.virtual_initial_balance,
            "current_balance": global_state.virtual_balance,
            "positions": global_state.virtual_positions,
            "total_unrealized_pnl": sum(pos.get('unrealized_pnl', 0) for pos in global_state.virtual_positions.values())
        },
        "account_alert": {
            "active": global_state.account_alert_active,
            "failure_count": global_state.account_failure_count
        },
        "chart_data": {
            "equity": global_state.equity_history
        },
        "decision": global_state.latest_decision,
        "decision_history": global_state.decision_history[:10],
        "trade_history": global_state.trade_history[:20],  # Return recent 20 trades
        "logs": global_state.recent_logs[:50]
    }
    return clean_nans(data)

@app.post("/api/control")
async def control_bot(cmd: ControlCommand):
    action = cmd.action.lower()
    if action == "start":
        global_state.execution_mode = "Running"
        global_state.add_log("▶️ System Resumed by User")
    elif action == "pause":
        global_state.execution_mode = "Paused"
        global_state.add_log("⏸️ System Paused by User")
    elif action == "stop":
        global_state.execution_mode = "Stopped"
        global_state.add_log("⏹️ System Stopped by User")

    elif action == "set_interval":
        if cmd.interval and cmd.interval in [0.5, 1, 3, 5]:
            global_state.cycle_interval = cmd.interval
            global_state.add_log(f"⏱️ Cycle interval updated to {cmd.interval} minutes")
            return {"status": "success", "interval": cmd.interval}
        else:
            raise HTTPException(status_code=400, detail="Invalid interval. Must be 0.5, 1, 3, or 5 minutes.")
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    return {"status": "success", "mode": global_state.execution_mode}

@app.post("/api/upload_prompt")
async def upload_prompt(file: UploadFile = File(...)):
    try:
        # Determine config directory
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        file_path = os.path.join(config_dir, 'custom_prompt.md')
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        global_state.add_log(f"📝 Custom prompt uploaded: {file.filename}")
        log.info(f"Custom prompt saved to: {file_path}")
        return {"status": "success", "message": "Custom prompt uploaded successfully. It will be used in the next decision cycle."}
    except Exception as e:
        log.error(f"Failed to upload prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from src.server.config_manager import ConfigManager
config_manager = ConfigManager(BASE_DIR)

@app.get("/api/config")
async def get_config():
    """Get current configuration (masked)"""
    return config_manager.get_config()

@app.get("/api/config/prompt")
async def get_prompt_content():
    """Get content of custom prompt file"""
    return {"content": config_manager.get_prompt()}

@app.post("/api/config")
async def update_config_endpoint(data: dict = Body(...)):
    """Update .env configuration"""
    success = config_manager.update_config(data)
    if success:
        return {"status": "success", "message": "Configuration updated. Please restart the bot if you changed API keys."}
    else:
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@app.post("/api/config/prompt")
async def update_prompt_text(data: dict = Body(...)):
    """Update custom prompt via text editor"""
    content = data.get("content", "")
    success = config_manager.update_prompt(content)
    if success:
        return {"status": "success", "message": "Prompt updated successfully. Will effect next cycle."}
    else:
        raise HTTPException(status_code=500, detail="Failed to save prompt")

@app.get("/api/config/default_prompt")
async def get_default_prompt():
    """Get the system default prompt template"""
    try:
        from src.config.default_prompt_template import DEFAULT_SYSTEM_PROMPT
        return {"content": DEFAULT_SYSTEM_PROMPT}
    except ImportError:
         raise HTTPException(status_code=500, detail="Default prompt template not found")

# ============================================================================
# Multi-Account API Endpoints
# ============================================================================

from src.exchanges import AccountManager, ExchangeAccount, ExchangeType

# Initialize account manager for API use
_account_manager = None

def get_account_manager():
    """Lazy initialization of account manager"""
    global _account_manager
    if _account_manager is None:
        import os
        from pathlib import Path
        config_path = Path(BASE_DIR) / "config" / "accounts.json"
        _account_manager = AccountManager(str(config_path))
        _account_manager.load_from_file()
    return _account_manager

@app.get("/api/accounts")
async def list_accounts():
    """List all configured trading accounts"""
    manager = get_account_manager()
    accounts = manager.list_accounts()
    return {
        "accounts": [acc.to_dict() for acc in accounts],
        "count": len(accounts)
    }

@app.get("/api/accounts/{account_id}")
async def get_account(account_id: str):
    """Get details of a specific account"""
    manager = get_account_manager()
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    return account.to_dict()

class AccountCreate(BaseModel):
    id: str
    name: str
    exchange: str = "binance"
    enabled: bool = True
    testnet: bool = True

@app.post("/api/accounts")
async def create_account(data: AccountCreate):
    """Create a new trading account"""
    manager = get_account_manager()
    
    # Validate exchange type
    try:
        exchange_type = ExchangeType(data.exchange.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported exchange: {data.exchange}")
    
    # Check if ID already exists
    if manager.get_account(data.id):
        raise HTTPException(status_code=400, detail=f"Account ID already exists: {data.id}")
    
    # Create account (API keys should be set via .env)
    import os
    env_prefix = f"ACCOUNT_{data.id.upper().replace('-', '_')}"
    api_key = os.environ.get(f"{env_prefix}_API_KEY", "")
    secret_key = os.environ.get(f"{env_prefix}_SECRET_KEY", "")
    
    account = ExchangeAccount(
        id=data.id,
        exchange_type=exchange_type,
        account_name=data.name,
        enabled=data.enabled,
        testnet=data.testnet,
        api_key=api_key,
        secret_key=secret_key
    )
    
    manager.add_account(account)
    manager.save_to_file()
    
    global_state.add_log(f"➕ Added account: {data.name} ({data.exchange})")
    return {"status": "success", "account": account.to_dict()}

@app.delete("/api/accounts/{account_id}")
async def delete_account(account_id: str):
    """Delete a trading account"""
    manager = get_account_manager()
    
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    name = account.account_name
    success = manager.remove_account(account_id)
    
    if success:
        manager.save_to_file()
        global_state.add_log(f"➖ Removed account: {name}")
        return {"status": "success", "message": f"Account '{name}' removed"}
    else:
        raise HTTPException(status_code=500, detail="Failed to remove account")

@app.get("/api/exchanges")
async def list_exchanges():
    """List supported exchanges"""
    from src.exchanges import get_supported_exchanges
    return {"exchanges": get_supported_exchanges()}

# Serve Static Files
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Root Route -> Index
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(WEB_DIR, 'index.html'))
