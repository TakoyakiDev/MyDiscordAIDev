import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Tuple, TypedDict, Literal, Annotated
import logging
from datetime import datetime
import operator

# LangChain & LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

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

# Gemini設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-exp",
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

# 会話履歴管理（ユーザーごと）
conversation_memory: Dict[int, List[BaseMessage]] = {}


# ============================================
# LangGraph用の状態定義（Gemini版）
# ============================================
class RAGState(TypedDict):
    """RAGワークフローの状態"""
    query: str
    user_id: int
    conversation_history: Annotated[List[BaseMessage], operator.add]
    
    # 検索結果
    web_context: str
    web_sources: List[str]
    web_error: str
    sheet_context: str
    sheet_sources: List[str]
    sheet_error: str
    
    # コンテキスト統合
    combined_context: str
    all_sources: List[str]
    context_quality_score: float
    
    # 回答生成
    answer: str
    answer_quality: str  # "good", "retry", "insufficient"
    
    # フロー制御
    needs_web: bool
    needs_sheet: bool
    retry_count: int
    max_retries: int
    
    # メタデータ
    start_time: str
    total_execution_time: float
    node_execution_times: Dict[str, float]


# ============================================
# LangGraphノード定義（Gemini版）
# ============================================
class RAGGraphNodes:
    """RAGワークフローのノード集（Gemini版）"""
    
    def __init__(self, web_retriever, sheets_retriever, llm):
        self.web_retriever = web_retriever
        self.sheets_retriever = sheets_retriever
        self.llm = llm
        self.html_parser = HTMLParser()
    
    async def initialize_state(self, state: RAGState) -> RAGState:
        """状態初期化ノード"""
        logger.info(f"[初期化] クエリ: {state['query']}")
        
        # 初期値設定
        state.setdefault('retry_count', 0)
        state.setdefault('max_retries', 2)
        state.setdefault('web_error', '')
        state.setdefault('sheet_error', '')
        state.setdefault('web_context', '')
        state.setdefault('sheet_context', '')
        state.setdefault('web_sources', [])
        state.setdefault('sheet_sources', [])
        state.setdefault('context_quality_score', 0.0)
        state.setdefault('node_execution_times', {})
        state.setdefault('start_time', datetime.now().isoformat())
        
        return state
    
    async def analyze_query(self, state: RAGState) -> RAGState:
        """質問分析ノード - どの情報源が必要かを判断"""
        start = datetime.now()
        logger.info(f"[質問分析] {state['query']}")
        
        query_lower = state['query'].lower()
        
        # キーワードベースの簡易ルーティング
        web_keywords = ['最新', 'ニュース', 'アップデート', '公式', 'wiki']
        sheet_keywords = ['イベント', 'ポイント', '攻略', 'おすすめ', 'データ']
        
        state['needs_web'] = any(keyword in query_lower for keyword in web_keywords) or len(query_lower) > 20
        state['needs_sheet'] = any(keyword in query_lower for keyword in sheet_keywords) or True  # デフォルトで有効
        
        # 短い質問は両方検索
        if len(query_lower) < 10:
            state['needs_web'] = True
            state['needs_sheet'] = True
        
        logger.info(f"[ルーティング] Web: {state['needs_web']}, Sheet: {state['needs_sheet']}")
        
        elapsed = (datetime.now() - start).total_seconds()
        state['node_execution_times']['analyze_query'] = elapsed
        return state
    
    async def search_web(self, state: RAGState) -> RAGState:
        """Web検索ノード（エラーハンドリング強化）"""
        start = datetime.now()
        logger.info("[Web検索] 実行中")
        
        if not state['needs_web']:
            logger.info("[Web検索] スキップ")
            elapsed = (datetime.now() - start).total_seconds()
            state['node_execution_times']['search_web'] = elapsed
            return state
        
        try:
            results = await self.web_retriever.search(state['query'], num_results=3)
            context = ""
            sources = []
            
            for result in results:
                try:
                    html_content = await self.web_retriever.fetch_content(result['link'])
                    clean_text = self.html_parser.extract_main_text(html_content)
                    
                    if clean_text:
                        context += f"【出典】{result['title']}\nURL: {result['link']}\n{clean_text[:2000]}\n\n"
                        sources.append(result['link'])
                except Exception as e:
                    logger.warning(f"Web取得エラー {result.get('link', 'unknown')}: {e}")
                    continue
            
            state['web_context'] = context
            state['web_sources'] = sources
            logger.info(f"[Web検索] 完了 - {len(sources)}件取得")
            
        except Exception as e:
            logger.error(f"[Web検索] エラー: {e}")
            state['web_error'] = str(e)
            state['web_context'] = ""
            state['web_sources'] = []
        
        elapsed = (datetime.now() - start).total_seconds()
        state['node_execution_times']['search_web'] = elapsed
        return state
    
    async def search_sheets(self, state: RAGState) -> RAGState:
        """スプレッドシート検索ノード（エラーハンドリング強化）"""
        start = datetime.now()
        logger.info("[Sheet検索] 実行中")
        
        if not state['needs_sheet']:
            logger.info("[Sheet検索] スキップ")
            elapsed = (datetime.now() - start).total_seconds()
            state['node_execution_times']['search_sheets'] = elapsed
            return state
        
        try:
            results = await self.sheets_retriever.semantic_search(state['query'], top_k=5)
            context = ""
            sources = []
            
            for result in results:
                event = result['event']
                point = result['point'][:500]  # 500文字
                row_num = result['row_index']
                similarity = result['similarity']
                
                # 類似度が低すぎる場合はスキップ
                if similarity < 0.3:
                    continue
                
                context += f"【スプレッドシート】{event} (関連度: {similarity:.2f})\n{point}\n\n"
                sources.append(
                    f"スプレッドシート行{row_num}: {sheets_retriever.sheet_url}#gid=0&range={row_num}"
                )
            
            state['sheet_context'] = context
            state['sheet_sources'] = sources
            logger.info(f"[Sheet検索] 完了 - {len(sources)}件取得")
            
        except Exception as e:
            logger.error(f"[Sheet検索] エラー: {e}")
            state['sheet_error'] = str(e)
            state['sheet_context'] = ""
            state['sheet_sources'] = []
        
        elapsed = (datetime.now() - start).total_seconds()
        state['node_execution_times']['search_sheets'] = elapsed
        return state
    
    async def evaluate_context(self, state: RAGState) -> RAGState:
        """コンテキスト評価ノード（品質スコアリング）"""
        start = datetime.now()
        logger.info("[コンテキスト評価] 実行中")
        
        state['combined_context'] = state['web_context'] + "\n\n" + state['sheet_context']
        state['all_sources'] = state['web_sources'] + state['sheet_sources']
        
        # 品質スコア計算
        score = 0.0
        if len(state['combined_context'].strip()) > 100:
            score += 30.0
        if len(state['combined_context'].strip()) > 500:
            score += 30.0
        if len(state['all_sources']) >= 2:
            score += 20.0
        if len(state['all_sources']) >= 4:
            score += 10.0
        if state['sheet_context']:
            score += 10.0  # スプレッドシート情報がある場合はボーナス
        
        state['context_quality_score'] = score
        logger.info(f"[コンテキスト評価] 品質スコア: {score}/100")
        
        elapsed = (datetime.now() - start).total_seconds()
        state['node_execution_times']['evaluate_context'] = elapsed
        return state
    
    async def generate_answer(self, state: RAGState) -> RAGState:
        """回答生成ノード（Gemini用プロンプト）"""
        start = datetime.now()
        logger.info("[回答生成] 実行中")
        
        # Geminiは長いコンテキストに対応（最大8192トークン）
        max_context_length = 6000
        context = state['combined_context']
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n...(以下省略)"
        
        # 会話履歴を含む
        history_text = ""
        if state.get('user_id') and state['user_id'] in conversation_memory:
            recent_history = conversation_memory[state['user_id']][-4:]  # 最新2往復
            for msg in recent_history:
                if isinstance(msg, HumanMessage):
                    history_text += f"ユーザー: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"AI: {msg.content}\n"
        
        prompt = f"""あなたは「ホワイトアウト・サバイバル」の専門攻略AIアシスタントです。
以下の参照情報をもとに、質問に日本語で正確かつ詳細に答えてください。

【重要ルール】
1. スプレッドシート情報を最優先してください
2. 関連度の高い情報から順に説明してください
3. 具体的な数値やデータがあれば必ず含めてください
4. 情報が不足している場合は「情報がありません」と明記してください
5. 読みやすく、構造化された回答を心がけてください

{f"【会話履歴】\\n{history_text}\\n" if history_text else ""}

【参照情報】
{context}

【質問】
{state['query']}

【回答】"""
        
        try:
            response = await asyncio.to_thread(lambda: self.llm.invoke(prompt))
            answer = response.content.strip()
            state['answer'] = answer
            
            # 回答品質チェック
            if len(answer) < 20:
                state['answer_quality'] = "retry"
                logger.warning("[回答生成] 回答が短すぎます")
            elif "情報がありません" in answer or "わかりません" in answer:
                state['answer_quality'] = "insufficient"
                logger.info("[回答生成] 情報不足")
            else:
                state['answer_quality'] = "good"
                logger.info("[回答生成] 良質な回答を生成")
            
        except Exception as e:
            logger.error(f"[回答生成] エラー: {e}")
            state['answer'] = f"⚠️ Gemini API呼び出しエラー: {str(e)}"
            state['answer_quality'] = "insufficient"
        
        elapsed = (datetime.now() - start).total_seconds()
        state['node_execution_times']['generate_answer'] = elapsed
        return state
    
    async def handle_insufficient_context(self, state: RAGState) -> RAGState:
        """コンテキスト不足時の処理ノード"""
        start = datetime.now()
        logger.warning("[コンテキスト不足] 処理中")
        
        error_info = []
        if state['web_error']:
            error_info.append(f"Web検索エラー: {state['web_error']}")
        if state['sheet_error']:
            error_info.append(f"スプレッドシートエラー: {state['sheet_error']}")
        
        if error_info:
            state['answer'] = f"⚠️ 情報取得中にエラーが発生しました:\n" + "\n".join(error_info)
        else:
            state['answer'] = """関連する情報が見つかりませんでした。

以下をお試しください:
• より具体的なキーワードを使用する
• 質問を言い換える
• イベント名やゲーム機能の正式名称を使用する"""
        
        state['answer_quality'] = "insufficient"
        
        elapsed = (datetime.now() - start).total_seconds()
        state['node_execution_times']['handle_insufficient_context'] = elapsed
        return state
    
    async def retry_node(self, state: RAGState) -> RAGState:
        """リトライ処理ノード"""
        logger.info(f"[リトライ] {state['retry_count'] + 1}回目")
        state['retry_count'] += 1
        
        # より多くの結果を取得するように調整
        state['needs_web'] = True
        state['needs_sheet'] = True
        
        return state
    
    async def finalize(self, state: RAGState) -> RAGState:
        """最終処理ノード - 会話履歴保存とメトリクス計算"""
        start = datetime.now()
        logger.info("[最終処理] 実行中")
        
        # 会話履歴に追加
        if state.get('user_id'):
            if state['user_id'] not in conversation_memory:
                conversation_memory[state['user_id']] = []
            
            conversation_memory[state['user_id']].append(HumanMessage(content=state['query']))
            conversation_memory[state['user_id']].append(AIMessage(content=state['answer']))
            
            # 履歴を最新10件に制限
            if len(conversation_memory[state['user_id']]) > 20:
                conversation_memory[state['user_id']] = conversation_memory[state['user_id']][-20:]
        
        # 総実行時間計算
        if state.get('start_time'):
            start_dt = datetime.fromisoformat(state['start_time'])
            state['total_execution_time'] = (datetime.now() - start_dt).total_seconds()
        
        logger.info(f"[最終処理] 完了 - 実行時間: {state.get('total_execution_time', 0):.2f}秒")
        
        elapsed = (datetime.now() - start).total_seconds()
        state['node_execution_times']['finalize'] = elapsed
        return state


# ============================================
# LangGraphワークフロー構築（Gemini版）
# ============================================
def create_rag_graph(web_retriever, sheets_retriever, llm):
    """RAGワークフローのグラフを作成（Gemini版）"""
    
    nodes = RAGGraphNodes(web_retriever, sheets_retriever, llm)
    
    workflow = StateGraph(RAGState)
    
    # ノード追加
    workflow.add_node("initialize", nodes.initialize_state)
    workflow.add_node("analyze_query", nodes.analyze_query)
    workflow.add_node("search_web", nodes.search_web)
    workflow.add_node("search_sheets", nodes.search_sheets)
    workflow.add_node("evaluate_context", nodes.evaluate_context)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("handle_insufficient_context", nodes.handle_insufficient_context)
    workflow.add_node("retry", nodes.retry_node)
    workflow.add_node("finalize", nodes.finalize)
    
    # エントリーポイント
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_query")
    
    # 並列検索
    workflow.add_edge("analyze_query", "search_web")
    workflow.add_edge("analyze_query", "search_sheets")
    
    # 検索結果の統合
    workflow.add_edge("search_web", "evaluate_context")
    workflow.add_edge("search_sheets", "evaluate_context")
    
    # コンテキスト評価後の条件分岐
    def route_after_evaluation(state: RAGState) -> Literal["generate_answer", "handle_insufficient_context"]:
        if state['context_quality_score'] >= 30.0:
            return "generate_answer"
        return "handle_insufficient_context"
    
    workflow.add_conditional_edges(
        "evaluate_context",
        route_after_evaluation,
        {
            "generate_answer": "generate_answer",
            "handle_insufficient_context": "handle_insufficient_context"
        }
    )
    
    # 回答品質チェック後の条件分岐
    def route_after_answer(state: RAGState) -> Literal["retry", "finalize"]:
        if state['answer_quality'] == "retry" and state['retry_count'] < state['max_retries']:
            return "retry"
        return "finalize"
    
    workflow.add_conditional_edges(
        "generate_answer",
        route_after_answer,
        {
            "retry": "retry",
            "finalize": "finalize"
        }
    )
    
    # リトライループ
    workflow.add_edge("retry", "search_web")
    workflow.add_edge("retry", "search_sheets")
    
    # 終了
    workflow.add_edge("handle_insufficient_context", "finalize")
    workflow.add_edge("finalize", END)
    
    # メモリセーバー付きでコンパイル
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# グラフをコンパイル
rag_graph = create_rag_graph(web_retriever, sheets_retriever, llm)


# ============================================
# Discordコマンド
# ============================================
@bot.event
async def on_ready():
    """Bot起動時の処理"""
    logger.info(f'{bot.user} としてログインしました（LangGraph + Gemini版）')
    logger.info(f'Gemini Model: gemini-2.0-flash-exp')
    logger.info(f'Embedding Model: text-embedding-004')
    
    try:
        await sheets_retriever.load_data()
        logger.info("✅ スプレッドシートデータ読み込み完了")
    except Exception as e:
        logger.error(f"❌ スプレッドシート読み込みエラー: {e}")


@bot.command(name='ask', help='RAGシステムに質問します（LangGraph + Gemini版）')
async def ask_command(ctx, *, question: str):
    """質問コマンド"""
    logger.info(f"質問受付: {ctx.author.name}({ctx.author.id}) - {question}")
    
    async with ctx.typing():
        initial_state: RAGState = {
            'query': question,
            'user_id': ctx.author.id,
            'conversation_history': [],
            'web_context': "",
            'web_sources': [],
            'web_error': "",
            'sheet_context': "",
            'sheet_sources': [],
            'sheet_error': "",
            'combined_context': "",
            'all_sources': [],
            'context_quality_score': 0.0,
            'answer': "",
            'answer_quality': "good",
            'needs_web': True,
            'needs_sheet': True,
            'retry_count': 0,
            'max_retries': 2,
            'start_time': datetime.now().isoformat(),
            'total_execution_time': 0.0,
            'node_execution_times': {}
        }
        
        config = {"configurable": {"thread_id": str(ctx.author.id)}}
        
        try:
            result = await rag_graph.ainvoke(initial_state, config)
        except Exception as e:
            logger.error(f"グラフ実行エラー: {e}")
            await ctx.send(f"⚠️ システムエラーが発生しました: {str(e)}")
            return
    
    # 回答を整形
    answer_text = result['answer']
    sources_text = "\n".join(result['all_sources']) if result['all_sources'] else "なし"
    
    # 実行時間情報
    exec_time = result.get('total_execution_time', 0)
    quality_score = result.get('context_quality_score', 0)
    
    embed = discord.Embed(
        title="🤖 回答 (LangGraph + Gemini版)",
        description=answer_text[:2000],  # Discord制限
        color=discord.Color.green()
    )
    embed.add_field(name="📝 質問", value=question[:1024], inline=False)
    
    # ソース情報
    if len(sources_text) > 1024:
        sources_parts = [sources_text[i:i+1024] for i in range(0, len(sources_text), 1024)]
        for i, part in enumerate(sources_parts[:3]):  # 最大3パート
            embed.add_field(
                name=f"🔗 参照元 (Part {i+1})",
                value=part,
                inline=False
            )
    else:
        embed.add_field(name="🔗 参照元", value=sources_text[:1024], inline=False)
    
    # メトリクス情報
    metrics = f"⏱️ {exec_time:.2f}秒 | 📊 品質スコア: {quality_score:.0f}/100"
    if result.get('retry_count', 0) > 0:
        metrics += f" | 🔄 リトライ: {result['retry_count']}回"
    
    embed.set_footer(text=f"{metrics} | Powered by LangGraph + Gemini")
    
    await ctx.send(embed=embed)


@bot.command(name='history', help='会話履歴を表示します（最新5件）')
async def history_command(ctx):
    """会話履歴表示コマンド"""
    user_id = ctx.author.id
    
    if user_id not in conversation_memory or not conversation_memory[user_id]:
        await ctx.send("会話履歴がありません。")
        return
    
    history = conversation_memory[user_id][-10:]  # 最新5往復
    
    embed = discord.Embed(
        title="📜 会話履歴",
        color=discord.Color.blue()
    )
    
    for i, msg in enumerate(history):
        if isinstance(msg, HumanMessage):
            embed.add_field(
                name=f"🧑 質問 #{i//2 + 1}",
                value=msg.content[:200],
                inline=False
            )
        elif isinstance(msg, AIMessage):
            embed.add_field(
                name=f"🤖 回答 #{i//2 + 1}",
                value=msg.content[:200],
                inline=False
            )
    
    await ctx.send(embed=embed)


@bot.command(name='clear', help='会話履歴をクリアします')
async def clear_command(ctx):
    """会話履歴クリアコマンド"""
    user_id = ctx.author.id
    
    if user_id in conversation_memory:
        del conversation_memory[user_id]
        await ctx.send("✅ 会話履歴をクリアしました。")
    else:
        await ctx.send("会話履歴がありません。")


@bot.command(name='graph', help='LangGraphのワークフローを可視化します')
async def graph_command(ctx):
    """グラフ可視化コマンド"""
    graph_text = """```
START
  ↓
[初期化]
  ↓
[質問分析] ← 動的ルーティング
  ↓
  ├→ [Web検索]     ⎤
  └→ [Sheet検索]   ⎦ 並列実行
       ↓
  [コンテキスト評価] ← 品質スコアリング
       ↓
  ┌────┴────┐
  │ スコア >= 30?
  ├─ YES → [回答生成]
  │          ↓
  │     ┌───┴───┐
  │     │品質OK? │
  │     ├─ NO → [リトライ] ──┐
  │     │         (最大2回)   │
  │     └─ YES               │
  │          ↓               │
  └─ NO → [コンテキスト不足] │
            ↓               │
       [最終処理] ←──────────┘
            ↓
          END
```"""
    
    embed = discord.Embed(
        title="📊 LangGraph + Geminiワークフロー",
        description=graph_text,
        color=discord.Color.blue()
    )
    
    features = """
    **主要機能:**
    • 動的ルーティング（質問内容に応じて検索先を決定）
    • 並列検索実行（Web + スプレッドシート）
    • 品質スコアリング（0-100点）
    • 自動リトライ（最大2回）
    • 会話履歴管理（ユーザーごと）
    • Gemini 2.0 Flash（高速・高品質）
    """
    embed.add_field(name="✨ 特徴", value=features, inline=False)
    
    await ctx.send(embed=embed)


@bot.command(name='stats', help='システム統計情報を表示します')
async def stats_command(ctx):
    """統計情報表示コマンド"""
    total_users = len(conversation_memory)
    total_messages = sum(len(history) for history in conversation_memory.values())
    
    embed = discord.Embed(
        title="📈 システム統計",
        color=discord.Color.green()
    )
    embed.add_field(name="👥 会話ユーザー数", value=f"{total_users}人", inline=True)
    embed.add_field(name="💬 総メッセージ数", value=f"{total_messages}件", inline=True)
    embed.add_field(name="🤖 使用モデル", value="LLM: Gemini 2.0 Flash\nEmbed: text-embedding-004", inline=False)
    
    # スプレッドシートデータ数
    if sheets_retriever.data_cache:
        embed.add_field(name="📊 スプレッドシートデータ", value=f"{len(sheets_retriever.data_cache)}行", inline=True)
    
    await ctx.send(embed=embed)


@bot.command(name='reload', help='スプレッドシートデータを再読み込みします')
@commands.has_permissions(administrator=True)
async def reload_command(ctx):
    """スプレッドシート再読み込みコマンド"""
    async with ctx.typing():
        try:
            await sheets_retriever.load_data()
            data_count = len(sheets_retriever.data_cache)
            await ctx.send(f"✅ スプレッドシートデータを再読み込みしました。（{data_count}行）")
        except Exception as e:
            await ctx.send(f"❌ 再読み込みエラー: {str(e)}")


@bot.event
async def on_command_error(ctx, error):
    """エラーハンドリング"""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("⚠️ 引数が不足しています。`!help`でコマンド一覧を確認してください。")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("⚠️ このコマンドを実行する権限がありません。")
    elif isinstance(error, commands.CommandNotFound):
        pass  # コマンドが見つからない場合は無視
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
