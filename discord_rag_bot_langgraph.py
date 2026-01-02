import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Tuple, TypedDict
import logging

# LangChain & LangGraph imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

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

load_dotenv()

# Discord Bot設定
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Ollama設定
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

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


# ============================================
# LangGraph用の状態定義
# ============================================
class RAGState(TypedDict):
    """RAGワークフローの状態"""
    query: str
    web_context: str
    web_sources: List[str]
    sheet_context: str
    sheet_sources: List[str]
    combined_context: str
    all_sources: List[str]
    answer: str
    needs_more_context: bool
    retry_count: int


# ============================================
# LangGraphノード定義
# ============================================
class RAGGraphNodes:
    """RAGワークフローのノード集"""
    
    def __init__(self, web_retriever, sheets_retriever, llm):
        self.web_retriever = web_retriever
        self.sheets_retriever = sheets_retriever
        self.llm = llm
        self.html_parser = HTMLParser()
    
    async def analyze_query(self, state: RAGState) -> RAGState:
        """質問分析ノード"""
        logger.info(f"[ノード] 質問分析: {state['query']}")
        state['retry_count'] = 0
        return state
    
    async def search_web(self, state: RAGState) -> RAGState:
        """Web検索ノード"""
        logger.info("[ノード] Web検索実行")
        try:
            results = await self.web_retriever.search(state['query'], num_results=3)
            context = ""
            sources = []
            
            for result in results:
                try:
                    html_content = await self.web_retriever.fetch_content(result['link'])
                    clean_text = self.html_parser.extract_main_text(html_content)
                    context += f"【出典】{result['link']}\n{clean_text[:1000]}\n\n"
                    sources.append(result['link'])
                except Exception as e:
                    logger.warning(f"Web取得エラー {result['link']}: {e}")
            
            state['web_context'] = context
            state['web_sources'] = sources
        except Exception as e:
            logger.error(f"Web検索エラー: {e}")
            state['web_context'] = ""
            state['web_sources'] = []
        
        return state
    
    async def search_sheets(self, state: RAGState) -> RAGState:
        """スプレッドシート検索ノード"""
        logger.info("[ノード] スプレッドシート検索実行")
        try:
            results = await self.sheets_retriever.semantic_search(state['query'], top_k=3)
            context = ""
            sources = []
            
            for result in results:
                event = result['event']
                point = result['point'][:300]
                row_num = result['row_index']
                similarity = result['similarity']
                
                context += f"【スプレッドシート】{event} (関連度: {similarity:.2f}):\n{point}\n\n"
                sources.append(
                    f"スプレッドシート: {sheets_retriever.sheet_url}#gid=0&range={row_num}"
                )
            
            state['sheet_context'] = context
            state['sheet_sources'] = sources
        except Exception as e:
            logger.error(f"スプレッドシート検索エラー: {e}")
            state['sheet_context'] = ""
            state['sheet_sources'] = []
        
        return state
    
    async def evaluate_context(self, state: RAGState) -> RAGState:
        """コンテキスト評価ノード"""
        logger.info("[ノード] コンテキスト評価")
        
        state['combined_context'] = state['web_context'] + "\n\n" + state['sheet_context']
        state['all_sources'] = state['web_sources'] + state['sheet_sources']
        
        min_context_length = 100
        if len(state['combined_context'].strip()) < min_context_length:
            state['needs_more_context'] = True
            logger.warning("コンテキスト不足")
        else:
            state['needs_more_context'] = False
            logger.info(f"コンテキスト十分: {len(state['combined_context'])}文字")
        
        return state
    
    async def generate_answer(self, state: RAGState) -> RAGState:
        """回答生成ノード"""
        logger.info("[ノード] 回答生成")
        
        max_context_length = 2000
        context = state['combined_context']
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n...(以下省略)"
        
        prompt = f"""以下の参照情報をもとに質問に日本語で答えてください。
スプレッドシート情報を優先し、簡潔にまとめてください。

参照情報:
{context}

質問: {state['query']}

回答:"""
        
        try:
            response = await asyncio.to_thread(lambda: self.llm.invoke(prompt))
            state['answer'] = response.content
            logger.info("回答生成成功")
        except Exception as e:
            logger.error(f"回答生成エラー: {e}")
            state['answer'] = f"⚠️ エラーが発生しました: {str(e)}"
        
        return state
    
    async def handle_insufficient_context(self, state: RAGState) -> RAGState:
        """コンテキスト不足時の処理ノード"""
        logger.warning("[ノード] コンテキスト不足処理")
        state['answer'] = "関連する情報が見つかりませんでした。質問を変えて再度お試しください。"
        return state
    
    async def quality_check(self, state: RAGState) -> RAGState:
        """回答品質チェックノード"""
        logger.info("[ノード] 品質チェック")
        
        if len(state['answer']) < 20 and state['retry_count'] < 2:
            logger.warning("回答が短すぎるため再生成")
            state['retry_count'] += 1
        
        return state


# ============================================
# LangGraphワークフロー構築
# ============================================
def create_rag_graph(web_retriever, sheets_retriever, llm):
    """RAGワークフローのグラフを作成"""
    
    nodes = RAGGraphNodes(web_retriever, sheets_retriever, llm)
    
    workflow = StateGraph(RAGState)
    
    # ノード追加
    workflow.add_node("analyze_query", nodes.analyze_query)
    workflow.add_node("search_web", nodes.search_web)
    workflow.add_node("search_sheets", nodes.search_sheets)
    workflow.add_node("evaluate_context", nodes.evaluate_context)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("handle_insufficient_context", nodes.handle_insufficient_context)
    workflow.add_node("quality_check", nodes.quality_check)
    
    # エッジ定義
    workflow.set_entry_point("analyze_query")
    
    workflow.add_edge("analyze_query", "search_web")
    workflow.add_edge("analyze_query", "search_sheets")
    workflow.add_edge("search_web", "evaluate_context")
    workflow.add_edge("search_sheets", "evaluate_context")
    
    # 条件分岐
    def should_generate_answer(state: RAGState) -> str:
        if state['needs_more_context']:
            return "handle_insufficient_context"
        return "generate_answer"
    
    workflow.add_conditional_edges(
        "evaluate_context",
        should_generate_answer,
        {
            "generate_answer": "generate_answer",
            "handle_insufficient_context": "handle_insufficient_context"
        }
    )
    
    workflow.add_edge("generate_answer", "quality_check")
    workflow.add_edge("handle_insufficient_context", END)
    workflow.add_edge("quality_check", END)
    
    return workflow.compile()


# グラフをコンパイル
rag_graph = create_rag_graph(web_retriever, sheets_retriever, llm)


# ============================================
# Discordコマンド
# ============================================
@bot.event
async def on_ready():
    """Bot起動時の処理"""
    logger.info(f'{bot.user} としてログインしました（LangGraph版）')
    logger.info(f'Ollama Model: {OLLAMA_MODEL}')
    
    try:
        await sheets_retriever.load_data()
        logger.info("✅ スプレッドシートデータ読み込み完了")
    except Exception as e:
        logger.error(f"❌ スプレッドシート読み込みエラー: {e}")


@bot.command(name='ask', help='RAGシステムに質問します（LangGraph版）')
async def ask_command(ctx, *, question: str):
    """質問コマンド"""
    logger.info(f"質問受付: {ctx.author} - {question}")
    
    async with ctx.typing():
        initial_state = RAGState(
            query=question,
            web_context="",
            web_sources=[],
            sheet_context="",
            sheet_sources=[],
            combined_context="",
            all_sources=[],
            answer="",
            needs_more_context=False,
            retry_count=0
        )
        
        result = await asyncio.to_thread(
            lambda: rag_graph.invoke(initial_state)
        )
    
    answer_text = result['answer']
    sources_text = "\n".join(result['all_sources']) if result['all_sources'] else "なし"
    
    embed = discord.Embed(
        title="🤖 回答 (LangGraph版)",
        description=answer_text,
        color=discord.Color.purple()
    )
    embed.add_field(name="📝 質問", value=question, inline=False)
    
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
    
    embed.set_footer(text=f"回答者: {bot.user.name} (Powered by LangGraph + Ollama)")
    
    await ctx.send(embed=embed)


@bot.command(name='graph', help='LangGraphのワークフローを可視化します')
async def graph_command(ctx):
    """グラフ可視化コマンド"""
    graph_text = (
        "```\n"
        "START\n"
        "  ↓\n"
        "[質問分析]\n"
        "  ↓\n"
        "  ├→ [Web検索]\n"
        "  └→ [スプレッドシート検索]\n"
        "  ↓\n"
        "[コンテキスト評価]\n"
        "  ↓\n"
        "  ├→ 十分 → [回答生成] → [品質チェック] → END\n"
        "  └→ 不足 → [コンテキスト不足処理] → END\n"
        "```"
    )
    
    embed = discord.Embed(
        title="📊 LangGraphワークフロー",
        description=graph_text,
        color=discord.Color.blue()
    )
    await ctx.send(embed=embed)


@bot.command(name='reload', help='スプレッドシートデータを再読み込みします')
@commands.has_permissions(administrator=True)
async def reload_command(ctx):
    """スプレッドシート再読み込みコマンド"""
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
