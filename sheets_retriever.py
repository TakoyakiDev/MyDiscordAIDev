import gspread
from oauth2client.service_account import ServiceAccountCredentials
import asyncio
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SheetsRetriever:
    """Google Sheetsからセマンティック検索を行うRetriever"""
    
    def __init__(self, sheet_id: str, sheet_name: str, service_account_file: str, embeddings):
        self.sheet_id = sheet_id
        self.sheet_name = sheet_name
        self.service_account_file = service_account_file
        self.embeddings = embeddings
        self.data_cache = []
        self.sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    
    async def load_data(self):
        """スプレッドシートからデータを読み込み"""
        logger.info("スプレッドシートデータ読み込み開始")
        
        def _load():
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.service_account_file, scope
            )
            client = gspread.authorize(creds)
            
            sheet = client.open_by_key(self.sheet_id).worksheet(self.sheet_name)
            return sheet.get_all_values()
        
        # ブロッキング処理を別スレッドで実行
        data = await asyncio.to_thread(_load)
        
        # データをキャッシュに保存（ヘッダー行をスキップ）
        self.data_cache = []
        for i, row in enumerate(data[1:], start=2):  # 1行目はヘッダー
            if len(row) >= 2 and row[0] and row[1]:
                self.data_cache.append({
                    'event': row[0],
                    'point': row[1],
                    'text': f"{row[0]}: {row[1]}",
                    'row_index': i
                })
        
        logger.info(f"スプレッドシートデータ読み込み完了: {len(self.data_cache)}行")
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """セマンティック検索を実行"""
        if not self.data_cache:
            await self.load_data()
        
        # クエリをベクトル化
        query_vector = await self._get_embedding(query)
        
        # 各行のベクトルを取得して類似度計算
        results = []
        for item in self.data_cache:
            item_vector = await self._get_embedding(item['text'])
            similarity = self._cosine_similarity(query_vector, item_vector)
            results.append({
                **item,
                'similarity': similarity
            })
        
        # 類似度でソート
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    async def _get_embedding(self, text: str, max_retries: int = 3) -> np.ndarray:
        """テキストをベクトル化（リトライ付き）"""
        for attempt in range(max_retries):
            try:
                vector = await asyncio.to_thread(
                    lambda: self.embeddings.embed_query(text)
                )
                return np.array(vector)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Embedding取得エラー (試行{attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(2)
                else:
                    logger.error(f"Embedding取得最終失敗: {e}")
                    return np.array([])
        
        return np.array([])
    
    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """コサイン類似度を計算"""
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
