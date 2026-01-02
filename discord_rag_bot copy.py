import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Tuple
import logging

# LangChain imports (修正版)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document  # ← 修正
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← 修正

# Custom imports
from web_search import WebSearchRetriever
from sheets_retriever import SheetsRetriever
from html_parser import HTMLParser

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

# Discord Bot設定
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# LangChain & Gemini設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    max_retries=3
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

# Retriever初期化
web_retriever = WebSearchRetriever(
    api_key=os.getenv("CSE_API_KEY"),
    cx=os.getenv("CSE_CX")
)
sheets_retriever = SheetsRetriever(
    sheet_id=os.getenv("SHEET_ID"),
    sheet_name=os.getenv("SHEET_NAME"),
    service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
    embeddings=embeddings
)


class RAGSystem:
    """RAGシステムのメインクラス"""
    
    def __init__(self, llm, embeddings, web_retriever, sheets_retriever):
        self.llm = llm
        self.embeddings = embeddings
        self.web_retriever = web_retriever
        self.sheets_retriever = sheets_retriever
        self.html_parser = HTMLParser()
    
    async def get_context(self, query: str) -> Tuple[str, List[str]]:
        """Web検索とスプレッドシートから文脈を取得"""
        logger.info(f"コンテキスト取得開始: {query}")
        
        # 並列で両方のRetrieverを実行
        web_task = asyncio.create_task(self._get_web_context(query))
        sheets_task = asyncio.create_task(self._get_sheets_context(query))
        
        web_result, sheets_result = await asyncio.gather(web_task, sheets_task)
        
        # 結果を統合
        combined_context = web_result['context'] + "\n\n" + sheets_result['context']
        combined_sources = web_result['sources'] + sheets_result['sources']
        
        logger.info(f"コンテキスト取得完了: {len(combined_sources)}件のソース")
        return combined_context, combined_sources
    
    async def _get_web_context(self, query: str) -> Dict:
        """Web検索から文脈を取得"""
        try:
            results = await self.web_retriever.search(query, num_results=3)
            context = ""
            sources = []
            
            for result in results:
                try:
                    # HTML本文を取得・解析
                    html_content = await self.web_retriever.fetch_content(result['link'])
                    clean_text = self.html_parser.extract_main_text(html_content)
                    
                    context += f"【出典】{result['link']}\n{clean_text[:2000]}\n\n"
                    sources.append(result['link'])
                except Exception as e:
                    logger.warning(f"Web取得エラー {result['link']}: {e}")
                    continue
            
            return {'context': context, 'sources': sources}
        except Exception as e:
            logger.error(f"Web検索エラー: {e}")
            return {'context': "", 'sources': []}
    
    async def _get_sheets_context(self, query: str) -> Dict:
        """スプレッドシートから文脈を取得（セマンティック検索）"""
        try:
            results = await self.sheets_retriever.semantic_search(query, top_k=5)
            context = ""
            sources = []
            
            for result in results:
                event = result['event']
                point = result['point'][:500]  # 500文字制限
                row_num = result['row_index']
                similarity = result['similarity']
                
                context += f"【スプレッドシート】{event} (関連度: {similarity:.2f}):\n{point}\n\n"
                sources.append(
                    f"スプレッドシート: {sheets_retriever.sheet_url}#gid=0&range={row_num}"
                )
            
            return {'context': context, 'sources': sources}
        except Exception as e:
            logger.error(f"スプレッドシート検索エラー: {e}")
            return {'context': "", 'sources': []}
    
    async def generate_answer(self, query: str, context: str, max_retries: int = 3) -> str:
        """Geminiで回答を生成（リトライ機能付き）"""
        prompt = f"""
あなたはオンラインゲーム「ホワイトアウト・サバイバル」の攻略AIエージェントです。
以下の参照情報をもとに質問に答えてください。
特にスプレッドシート情報を優先してください。
回答は日本語で簡潔にまとめてください。

参照情報:
{context}

質問:
{query}
"""
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    lambda: self.llm.invoke(prompt)
                )
                return response.content
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Gemini呼び出しエラー (試行{attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(5)
                else:
                    logger.error(f"Gemini呼び出し最終失敗: {e}")
                    return "⚠️ Gemini APIが混雑中のため応答できません。時間を置いて再実行してください。"
        
        return "⚠️ 予期しないエラーが発生しました。"
    
    async def ask(self, query: str) -> Dict:
        """メイン処理: 質問に対して回答を生成"""
        try:
            # コンテキスト取得
            context, sources = await self.get_context(query)
            
            if not context.strip():
                return {
                    'query': query,
                    'answer': "関連する情報が見つかりませんでした。",
                    'sources': []
                }
            
            # 回答生成
            answer = await self.generate_answer(query, context)
            
            return {
                'query': query,
                'answer': answer,
                'sources': sources
            }
        except Exception as e:
            logger.error(f"RAGシステムエラー: {e}")
            return {
                'query': query,
                'answer': f"エラーが発生しました: {str(e)}",
                'sources': []
            }


# RAGシステムのインスタンス化
rag_system = RAGSystem(llm, embeddings, web_retriever, sheets_retriever)


@bot.event
async def on_ready():
    """Bot起動時の処理"""
    logger.info(f'{bot.user} としてログインしました')
    logger.info(f'Discord.py version: {discord.__version__}')
    
    # スプレッドシートデータの初期読み込み
    try:
        await sheets_retriever.load_data()
        logger.info("スプレッドシートデータ読み込み完了")
    except Exception as e:
        logger.error(f"スプレッドシート読み込みエラー: {e}")


@bot.command(name='ask', help='RAGシステムに質問します。使い方: !ask [質問内容]')
async def ask_command(ctx, *, question: str):
    """質問コマンド"""
    logger.info(f"質問受付: {ctx.author} - {question}")
    
    # 処理中メッセージ
    async with ctx.typing():
        # RAG処理実行
        result = await rag_system.ask(question)
    
    # 回答を整形
    answer_text = result['answer']
    sources_text = "\n".join(result['sources']) if result['sources'] else "なし"
    
    # Discord埋め込みメッセージで送信
    embed = discord.Embed(
        title="🤖 回答",
        description=answer_text,
        color=discord.Color.blue()
    )
    embed.add_field(name="📝 質問", value=question, inline=False)
    
    # ソースが長すぎる場合は分割
    if len(sources_text) > 1024:
        sources_parts = [sources_text[i:i+1024] for i in range(0, len(sources_text), 1024)]
        for i, part in enumerate(sources_parts):
            embed.add_field(
                name=f"🔗 参照URL (Part {i+1})",
                value=part,
                inline=False
            )
    else:
        embed.add_field(name="🔗 参照URL", value=sources_text, inline=False)
    
    embed.set_footer(text=f"回答者: {bot.user.name}")
    
    await ctx.send(embed=embed)


@bot.command(name='context', help='質問に関連するコンテキストのみを取得します')
async def context_command(ctx, *, question: str):
    """コンテキスト取得コマンド（デバッグ用）"""
    async with ctx.typing():
        context, sources = await rag_system.get_context(question)
    
    # 文字数制限対応
    if len(context) > 1900:
        context = context[:1900] + "\n...(省略)"
    
    await ctx.send(f"**コンテキスト:**\n```\n{context}\n```")


@bot.command(name='reload', help='スプレッドシートデータを再読み込みします')
@commands.has_permissions(administrator=True)
async def reload_command(ctx):
    """スプレッドシート再読み込みコマンド（管理者専用）"""
    async with ctx.typing():
        try:
            await sheets_retriever.load_data()
            await ctx.send("✅ スプレッドシートデータを再読み込みしました。")
        except Exception as e:
            await ctx.send(f"❌ 再読み込みエラー: {str(e)}")


@bot.event
async def on_command_error(ctx, error):
    """エラーハンドリング"""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("⚠️ 引数が不足しています。`!help`でコマンド一覧を確認してください。")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("⚠️ このコマンドを実行する権限がありません。")
    else:
        logger.error(f"コマンドエラー: {error}")
        await ctx.send(f"⚠️ エラーが発生しました: {str(error)}")


def main():
    """Botを起動"""
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error("DISCORD_TOKENが設定されていません")
        return
    
    try:
        bot.run(token)
    except Exception as e:
        logger.error(f"Bot起動エラー: {e}")


if __name__ == "__main__":
    main()
