import aiohttp
import asyncio
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class WebSearchRetriever:
    """Google Custom Search APIを使ったWeb検索Retriever"""
    
    def __init__(self, api_key: str, cx: str):
        self.api_key = api_key
        self.cx = cx
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, num_results: int = 3) -> List[Dict]:
        """検索を実行"""
        params = {
            'key': self.api_key,
            'cx': self.cx,
            'q': query,
            'num': num_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"CSE API エラー: {response.status}")
                        return []
                    
                    data = await response.json()
                    items = data.get('items', [])
                    
                    return [
                        {
                            'title': item.get('title'),
                            'link': item.get('link'),
                            'snippet': item.get('snippet')
                        }
                        for item in items
                    ]
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return []
    
    async def fetch_content(self, url: str, timeout: int = 10) -> str:
        """URLからHTMLコンテンツを取得"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"コンテンツ取得失敗 {url}: {response.status}")
                        return ""
        except Exception as e:
            logger.error(f"URL取得エラー {url}: {e}")
            return ""
