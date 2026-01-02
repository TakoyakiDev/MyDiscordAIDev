from bs4 import BeautifulSoup
import re
import logging

logger = logging.getLogger(__name__)


class HTMLParser:
    """HTMLから本文を抽出するパーサー"""
    
    @staticmethod
    def extract_main_text(html: str, max_length: int = 2000) -> str:
        """HTMLから本文テキストを抽出"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # 不要なタグを削除
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # テキスト抽出
            text = soup.get_text(separator=' ', strip=True)
            
            # 空白を正規化
            text = re.sub(r'\s+', ' ', text)
            
            # 長さ制限
            return text[:max_length]
        except Exception as e:
            logger.error(f"HTML解析エラー: {e}")
            return ""
