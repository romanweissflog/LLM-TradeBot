"""
数据保存工具模块 - 按日期组织数据文件 (Multi-Agent Refactor)
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from src.utils.logger import log


class DataSaver:
    """数据保存工具类 - 按业务领域和日期自动组织文件
    
    目录结构 (Multi-Agent Architecture):
    data/
      market_data/       (原 step1)
      indicators/        (原 step2)
      features/          (原 step3)
      agent_context/     (原 step4)
      llm_logs/          (原 step5)
      decisions/         (原 step6)
      executions/        (原 step7)
      backtest/          (原 step8)
    """
    
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = base_dir
        
        # 定义业务目录映射 (Strict Multi-Agent Form)
        self.dirs = {
            'market_data': os.path.join(base_dir, 'data_sync_agent'),
            'indicators': os.path.join(base_dir, 'quant_analyst_agent', 'indicators'),
            'features': os.path.join(base_dir, 'quant_analyst_agent', 'features'),
            'agent_context': os.path.join(base_dir, 'quant_analyst_agent', 'context'),
            'llm_logs': os.path.join(base_dir, 'decision_core_agent', 'llm_logs'),
            'decisions': os.path.join(base_dir, 'decision_core_agent', 'decisions'),
            'executions': os.path.join(base_dir, 'execution_engine', 'orders'),
            'backtest': os.path.join(base_dir, 'execution_engine', 'backtest')
        }
        
        # 创建所有目录
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)
            
    def _get_date_folder(self, category: str, date: Optional[str] = None) -> str:
        """获取或创建指定类别的日期文件夹"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        category_dir = self.dirs.get(category)
        if not category_dir:
            # Fallback for unknown categories
            category_dir = os.path.join(self.base_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
        date_folder = os.path.join(category_dir, date)
        os.makedirs(date_folder, exist_ok=True)
        return date_folder
    
    def save_market_data(
        self,
        klines: List[Dict],
        symbol: str,
        timeframe: str,
        save_formats: List[str] = ['json', 'csv', 'parquet']
    ) -> Dict[str, str]:
        """保存原始K线数据 (原 save_step1_klines)"""
        if not klines:
            log.warning("K线数据为空，跳过保存")
            return {}
        
        date_folder = self._get_date_folder('market_data')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 元数据
        df = pd.DataFrame(klines)
        try:
            first_ts = pd.to_datetime(klines[0]['timestamp'], unit='ms')
            last_ts = pd.to_datetime(klines[-1]['timestamp'], unit='ms')
        except:
            first_ts = "unknown"
            last_ts = "unknown"
            
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'count': len(klines),
            'timestamp': timestamp
        }
        
        saved_files = {}
        filename_base = f'market_data_{symbol}_{timeframe}_{timestamp}'
        
        if 'json' in save_formats:
            path = os.path.join(date_folder, f'{filename_base}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'metadata': metadata, 'klines': klines}, f, indent=2)
            saved_files['json'] = path
            
        if 'csv' in save_formats:
            path = os.path.join(date_folder, f'{filename_base}.csv')
            df.to_csv(path, index=False)
            saved_files['csv'] = path
            
        if 'parquet' in save_formats:
            path = os.path.join(date_folder, f'{filename_base}.parquet')
            df.to_parquet(path, index=False)
            saved_files['parquet'] = path
            
        log.debug(f"保存市场数据: {symbol} {timeframe}")
        return saved_files

    def save_indicators(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        snapshot_id: str
    ) -> Dict[str, str]:
        """保存技术指标数据 (原 save_step2_indicators)"""
        date_folder = self._get_date_folder('indicators')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'indicators_{symbol}_{timeframe}_{timestamp}_{snapshot_id}.parquet'
        path = os.path.join(date_folder, filename)
        
        df.to_parquet(path)
        log.debug(f"保存技术指标: {path}")
        return {'parquet': path}

    def save_features(
        self,
        features: pd.DataFrame,
        symbol: str,
        timeframe: str,
        snapshot_id: str,
        version: str = 'v1'
    ) -> Dict[str, str]:
        """保存特征数据 (原 save_step3_features)"""
        date_folder = self._get_date_folder('features')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'features_{symbol}_{timeframe}_{timestamp}_{snapshot_id}_{version}.parquet'
        path = os.path.join(date_folder, filename)
        
        features.to_parquet(path)
        log.debug(f"保存特征数据: {path}")
        return {'parquet': path}

    def save_context(
        self,
        context: Dict,
        symbol: str,
        identifier: str,
        snapshot_id: str
    ) -> Dict[str, str]:
        """保存Agent上下文/分析结果 (原 save_step4_context)"""
        date_folder = self._get_date_folder('agent_context')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'context_{symbol}_{identifier}_{timestamp}_{snapshot_id}.json'
        path = os.path.join(date_folder, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2, ensure_ascii=False)
            
        log.debug(f"保存Agent上下文: {path}")
        return {'json': path}

    def save_llm_log(
        self,
        content: str,
        symbol: str,
        snapshot_id: str
    ) -> Dict[str, str]:
        """保存LLM交互日志 (原 save_step5_markdown)"""
        date_folder = self._get_date_folder('llm_logs')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'llm_log_{symbol}_{timestamp}_{snapshot_id}.md'
        path = os.path.join(date_folder, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        log.debug(f"保存LLM日志: {path}")
        return {'md': path}

    def save_decision(
        self,
        decision: Dict,
        symbol: str,
        snapshot_id: str
    ) -> Dict[str, str]:
        """保存决策结果 (原 save_step6_decision)"""
        date_folder = self._get_date_folder('decisions')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'decision_{symbol}_{timestamp}_{snapshot_id}.json'
        path = os.path.join(date_folder, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)
            
        log.debug(f"保存决策结果: {path}")
        return {'json': path}

    def save_execution(
        self,
        record: Dict,
        symbol: str
    ) -> Dict[str, str]:
        """保存执行记录 (原 save_step7_execution)"""
        date_folder = self._get_date_folder('executions')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'execution_{symbol}_{timestamp}.json'
        path = os.path.join(date_folder, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        # 追加CSV
        csv_path = os.path.join(date_folder, f'executions_{symbol}.csv')
        df = pd.DataFrame([record])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
            
        log.debug(f"保存执行记录: {path}")
        return {'json': path, 'csv': csv_path}

    def list_files(self, category: str, date: str = None) -> List[str]:
        """列出文件"""
        folder = self._get_date_folder(category, date)
        if not os.path.exists(folder):
            return []
        return [os.path.join(folder, f) for f in os.listdir(folder)]

    # 兼容性别名 (Adapters for old code if any remains)
    save_step1_klines = save_market_data
    save_step2_indicators = save_indicators
    save_step3_features = save_features
    save_step4_context = save_context
    save_step5_markdown = save_llm_log
    save_step6_decision = save_decision
    save_step7_execution = save_execution
