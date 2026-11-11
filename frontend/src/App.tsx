import {
  ArrowRight,
  BookOpen,
  Check,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Copy,
  Cpu,
  Database,
  FileSearch,
  Gauge,
  GitFork,
  Layers3,
  Link2,
  LoaderCircle,
  LockKeyhole,
  Network,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  Upload,
  X,
  Zap,
} from 'lucide-react'
import { type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import {
  enqueueIndex,
  fetchChunkGraph,
  fetchDocumentPageAudit,
  fetchDocuments,
  fetchJob,
  fetchRuntimeConfig,
  fetchSources,
  searchKnowledge,
  uploadDocuments,
} from './api'
import type {
  Citation,
  DocumentPageAudit,
  DocumentSummary,
  EvidenceItem,
  GovernedSource,
  GraphResponse,
  IndexJob,
  RuntimeMode,
  SearchHit,
  SearchResponse,
  ViewId,
} from './types'

const MODE_DETAILS: Record<
  RuntimeMode,
  { label: string; compact: string; description: string; icon: typeof Search }
> = {
  'strict-lexical': {
    label: '严格词法',
    compact: 'Lexical',
    description: 'SQLite FTS5 + TF-IDF；不加载 Torch、向量模型或生成模型。',
    icon: FileSearch,
  },
  competition: {
    label: '竞赛模式',
    compact: 'Competition',
    description: 'BGE-M3 检索可用，禁用生成式模型；不可用时自动退化到词法检索。',
    icon: ShieldCheck,
  },
  enhanced: {
    label: '增强检索',
    compact: 'Enhanced',
    description: '混合向量检索、知识图谱扩展与可选重排，不生成答案。',
    icon: Network,
  },
  llm: {
    label: 'Phi-4 辅助',
    compact: 'Local LLM',
    description: '在增强检索后请求显式配置的本地 Phi-4；无云端回退。',
    icon: Sparkles,
  },
}

const MODE_CAPABILITY: Record<RuntimeMode, number> = {
  'strict-lexical': 0,
  competition: 1,
  enhanced: 2,
  llm: 3,
}

const NAV_ITEMS: { id: ViewId; label: string; icon: typeof Search }[] = [
  { id: 'search', label: '知识检索', icon: Search },
  { id: 'sources', label: '知识源', icon: Database },
  { id: 'settings', label: '运行设置', icon: Settings2 },
]

const EXAMPLE_QUERIES = [
  '解释 ROP 链的构造步骤与常见约束',
  'CVE 与 CWE 在漏洞分类中的关系是什么？',
  'ELF 重定位表如何辅助逆向分析？',
]

function formatLocation(citation: Citation): string {
  const parts: string[] = []
  if (citation.page_start) {
    parts.push(
      citation.page_end && citation.page_end !== citation.page_start
        ? `第 ${citation.page_start}–${citation.page_end} 页`
        : `第 ${citation.page_start} 页`,
    )
  }
  if (citation.line_start) {
    parts.push(
      citation.line_end && citation.line_end !== citation.line_start
        ? `行 ${citation.line_start}–${citation.line_end}`
        : `行 ${citation.line_start}`,
    )
  }
  if (citation.section_path?.length) parts.push(citation.section_path.join(' / '))
  return parts.join(' · ') || citation.relative_path || '文档位置未标注'
}

function pagePreviewUrl(citation: Citation): string | null {
  if (!citation.relative_path?.toLocaleLowerCase().endsWith('.pdf') || !citation.page_start) {
    return null
  }
  const encodedPath = citation.relative_path
    .replaceAll('\\', '/')
    .split('/')
    .map(encodeURIComponent)
    .join('/')
  return `/api/pages/${encodedPath}?page=${citation.page_start}&dpi=144`
}

function pageAuditPreviewUrl(relativePath: string, page: number): string | null {
  if (!relativePath.toLocaleLowerCase().endsWith('.pdf')) return null
  const encodedPath = relativePath
    .replaceAll('\\', '/')
    .split('/')
    .map(encodeURIComponent)
    .join('/')
  return `/api/pages/${encodedPath}?page=${page}&dpi=144`
}

function formatScore(score: number): string {
  if (!Number.isFinite(score)) return '—'
  if (score > 0 && score < 0.01) return score.toFixed(4)
  return score.toFixed(3)
}

function App() {
  const [view, setView] = useState<ViewId>('search')
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<RuntimeMode>('strict-lexical')
  const [serverMode, setServerMode] = useState<RuntimeMode>('strict-lexical')
  const [generativeAvailable, setGenerativeAvailable] = useState(false)
  const [capabilities, setCapabilities] = useState<Record<string, boolean>>({})
  const [phiEnabled, setPhiEnabled] = useState(false)
  const [topK, setTopK] = useState(12)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastQuery, setLastQuery] = useState('')
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => () => controllerRef.current?.abort(), [])

  useEffect(() => {
    const controller = new AbortController()
    void fetchRuntimeConfig(controller.signal)
      .then((config) => {
        setServerMode(config.mode)
        setMode(config.mode)
        setGenerativeAvailable(config.generative_enabled)
        setCapabilities(config.capabilities)
        if (!config.generative_enabled) setPhiEnabled(false)
      })
      .catch(() => {
        setServerMode('strict-lexical')
        setMode('strict-lexical')
        setGenerativeAvailable(false)
        setPhiEnabled(false)
      })
    return () => controller.abort()
  }, [])

  const runSearch = async (event?: FormEvent, suggestedQuery?: string) => {
    event?.preventDefault()
    const nextQuery = (suggestedQuery ?? query).trim()
    if (!nextQuery || loading) return
    if (suggestedQuery) setQuery(suggestedQuery)
    setLastQuery(nextQuery)
    setLoading(true)
    setError(null)
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    try {
      const result = await searchKnowledge(
        { query: nextQuery, mode, top_k: topK, use_llm: phiEnabled },
        controller.signal,
      )
      setResponse(result)
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') return
      setResponse(null)
      setError(reason instanceof Error ? reason.message : '检索请求失败，请稍后重试。')
    } finally {
      if (controllerRef.current === controller) setLoading(false)
    }
  }

  const switchMode = (nextMode: RuntimeMode) => {
    if (MODE_CAPABILITY[nextMode] > MODE_CAPABILITY[serverMode]) return
    setMode(nextMode)
    if (nextMode !== 'llm') setPhiEnabled(false)
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        跳到主要内容
      </a>
      <aside className="sidebar" aria-label="主导航">
        <button className="brand" type="button" onClick={() => setView('search')}>
          <span className="brand-mark" aria-hidden="true">
            <ShieldCheck size={22} strokeWidth={2.2} />
          </span>
          <span>
            <strong>SecKnow</strong>
            <small>SECURITY INTELLIGENCE</small>
          </span>
        </button>

        <nav className="nav-list" aria-label="功能">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.id}
                type="button"
                className={view === item.id ? 'nav-item active' : 'nav-item'}
                aria-current={view === item.id ? 'page' : undefined}
                onClick={() => setView(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
                {view === item.id && <ChevronRight className="nav-arrow" size={16} />}
              </button>
            )
          })}
        </nav>

        <div className="sidebar-spacer" />
        <div className="privacy-card">
          <div className="privacy-icon" aria-hidden="true">
            <LockKeyhole size={17} />
          </div>
          <div>
            <strong>离线优先</strong>
            <p>搜索内容仅发送至本机 API，不包含任何云端回退。</p>
          </div>
        </div>
        <div className="sidebar-footer">
          <span className="status-dot" />
          <span>本地工作台</span>
          <span className="version">v0.1</span>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb">
            <span>工作台</span>
            <ChevronRight size={14} />
            <strong>{NAV_ITEMS.find((item) => item.id === view)?.label}</strong>
          </div>
          <div className="topbar-actions">
            <div className="mode-pill" title={MODE_DETAILS[mode].description}>
              <span className={mode === 'llm' && phiEnabled ? 'mode-dot llm' : 'mode-dot'} />
              {MODE_DETAILS[mode].label}
            </div>
            <button
              className={phiEnabled ? 'model-button enabled' : 'model-button'}
              type="button"
              aria-pressed={phiEnabled}
              disabled={!generativeAvailable}
              onClick={() => {
                if (!generativeAvailable) return
                setPhiEnabled((current) => !current)
                if (!phiEnabled) switchMode('llm')
              }}
            >
              <Cpu size={16} />
              Phi-4 {phiEnabled ? '已挂载' : '未挂载'}
            </button>
          </div>
        </header>

        <main id="main-content" tabIndex={-1}>
          {view === 'search' && (
            <SearchView
              query={query}
              setQuery={setQuery}
              mode={mode}
              setMode={switchMode}
              serverMode={serverMode}
              generativeAvailable={generativeAvailable}
              phiEnabled={phiEnabled}
              setPhiEnabled={setPhiEnabled}
              loading={loading}
              error={error}
              response={response}
              lastQuery={lastQuery}
              onSearch={runSearch}
            />
          )}
          {view === 'sources' && <SourcesView />}
          {view === 'settings' && (
            <SettingsView
              mode={mode}
              setMode={switchMode}
              serverMode={serverMode}
              generativeAvailable={generativeAvailable}
              phiEnabled={phiEnabled}
              setPhiEnabled={setPhiEnabled}
              topK={topK}
              setTopK={setTopK}
              capabilities={capabilities}
            />
          )}
        </main>
      </div>
    </div>
  )
}

interface SearchViewProps {
  query: string
  setQuery: (value: string) => void
  mode: RuntimeMode
  setMode: (value: RuntimeMode) => void
  serverMode: RuntimeMode
  generativeAvailable: boolean
  phiEnabled: boolean
  setPhiEnabled: (value: boolean) => void
  loading: boolean
  error: string | null
  response: SearchResponse | null
  lastQuery: string
  onSearch: (event?: FormEvent, suggestedQuery?: string) => Promise<void>
}

function SearchView({
  query,
  setQuery,
  mode,
  setMode,
  serverMode,
  generativeAvailable,
  phiEnabled,
  setPhiEnabled,
  loading,
  error,
  response,
  lastQuery,
  onSearch,
}: SearchViewProps) {
  const hasResults = Boolean(response || error || loading)
  return (
    <div className={hasResults ? 'search-page results-state' : 'search-page'}>
      <section className="search-hero" aria-labelledby="search-title">
        <div className="eyebrow">
          <span /> OFFLINE SECURITY KNOWLEDGE
        </div>
        <h1 id="search-title">
          从证据出发，<em>找到可信答案。</em>
        </h1>
        <p>
          面向 CTF 与安全研究的本地知识检索。融合词法、向量与图关系，所有结论可回溯到原始资料。
        </p>

        <form className="search-form" onSubmit={(event) => void onSearch(event)}>
          <label className="sr-only" htmlFor="knowledge-query">
            输入安全知识问题
          </label>
          <div className="search-box">
            <Search className="search-leading" size={22} aria-hidden="true" />
            <input
              id="knowledge-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索漏洞、利用技术、代码符号或安全概念…"
              autoComplete="off"
              spellCheck={false}
              disabled={loading}
            />
            {query && !loading && (
              <button
                className="clear-button"
                type="button"
                aria-label="清空搜索内容"
                onClick={() => setQuery('')}
              >
                <X size={17} />
              </button>
            )}
            <button className="search-submit" type="submit" disabled={!query.trim() || loading}>
              {loading ? <LoaderCircle className="spin" size={18} /> : <ArrowRight size={18} />}
              <span>{loading ? '检索中' : '开始检索'}</span>
            </button>
          </div>

          <div className="search-options">
            <label>
              <span>检索模式</span>
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value as RuntimeMode)}
                disabled={loading}
              >
                {(Object.keys(MODE_DETAILS) as RuntimeMode[]).map((value) => (
                  <option
                    value={value}
                    key={value}
                    disabled={MODE_CAPABILITY[value] > MODE_CAPABILITY[serverMode]}
                  >
                    {MODE_DETAILS[value].label}
                  </option>
                ))}
              </select>
            </label>
            <span className="option-divider" />
            <label className="compact-toggle">
              <input
                type="checkbox"
                checked={phiEnabled}
                onChange={(event) => {
                  if (!generativeAvailable) return
                  setPhiEnabled(event.target.checked)
                  if (event.target.checked) setMode('llm')
                }}
                disabled={loading || !generativeAvailable}
              />
              <span className="toggle-track" aria-hidden="true">
                <span />
              </span>
              <span>使用本地 Phi-4</span>
            </label>
            <span className="safe-label">
              <LockKeyhole size={13} /> 默认关闭
            </span>
          </div>
        </form>

        {!hasResults && (
          <div className="suggestions" aria-label="示例问题">
            <span>试试搜索</span>
            <div>
              {EXAMPLE_QUERIES.map((example) => (
                <button key={example} type="button" onClick={() => void onSearch(undefined, example)}>
                  {example}
                  <ArrowRight size={13} />
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {!hasResults && <CapabilityStrip />}
      {loading && <SearchProgress query={lastQuery} mode={mode} />}
      {error && <ErrorState message={error} onRetry={() => void onSearch()} />}
      {response && !loading && <Results response={response} />}
    </div>
  )
}

function CapabilityStrip() {
  return (
    <section className="capabilities" aria-label="检索能力">
      <article>
        <span className="capability-icon cyan"><FileSearch size={19} /></span>
        <div><strong>可验证检索</strong><small>BM25 · TF-IDF · 精确标识符</small></div>
      </article>
      <article>
        <span className="capability-icon violet"><Network size={19} /></span>
        <div><strong>图关系扩展</strong><small>实体 · 路径 · 跨文档关联</small></div>
      </article>
      <article>
        <span className="capability-icon green"><Link2 size={19} /></span>
        <div><strong>原文级引用</strong><small>页码 · 行号 · 版本 · 可信度</small></div>
      </article>
      <article>
        <span className="capability-icon amber"><Zap size={19} /></span>
        <div><strong>竞赛友好</strong><small>无生成模型也可完整运行</small></div>
      </article>
    </section>
  )
}

function SearchProgress({ query, mode }: { query: string; mode: RuntimeMode }) {
  return (
    <section className="progress-card" aria-live="polite" aria-busy="true">
      <div className="progress-spinner"><LoaderCircle className="spin" size={26} /></div>
      <div className="progress-copy">
        <strong>正在检索“{query}”</strong>
        <span>{MODE_DETAILS[mode].description}</span>
        <div className="progress-line"><span /></div>
      </div>
      <div className="progress-steps" aria-hidden="true">
        <span className="done"><Check size={12} /> 查询分析</span>
        <span className="active"><span className="pulse-dot" /> 多路召回</span>
        <span>证据重组</span>
      </div>
    </section>
  )
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section className="error-card" role="alert">
      <span><CircleAlert size={22} /></span>
      <div>
        <strong>无法完成检索</strong>
        <p>{message}</p>
      </div>
      <button type="button" onClick={onRetry}>重新尝试</button>
    </section>
  )
}

function Results({ response }: { response: SearchResponse }) {
  const [copied, setCopied] = useState<string | null>(null)
  const evidenceItems = response.answer?.sections.flatMap((section) => section.items) ?? []
  const hits = evidenceItems.length
    ? evidenceItems.map((item, index) => evidenceToHit(item, index))
    : response.hits
  const generatedText = response.generated_answer

  const copyText = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(id)
      window.setTimeout(() => setCopied(null), 1600)
    } catch {
      setCopied(null)
    }
  }

  return (
    <section className="results" aria-labelledby="results-title">
      <div className="results-summary">
        <div>
          <span className="result-kicker">RETRIEVAL COMPLETE</span>
          <h2 id="results-title">检索证据</h2>
          <p>{response.answer?.summary || `找到 ${hits.length} 条可追溯证据。`}</p>
        </div>
        <dl>
          <div><dt>证据</dt><dd>{hits.length}</dd></div>
          <div><dt>耗时</dt><dd>{Math.round(response.elapsed_ms)} ms</dd></div>
          <div>
            <dt>模式</dt>
            <dd>{MODE_DETAILS[response.mode as RuntimeMode]?.compact ?? response.mode}</dd>
          </div>
        </dl>
      </div>

      {response.warnings.length > 0 && (
        <div className="warning-stack" role="status">
          {Array.from(new Set(response.warnings)).map((warning) => (
            <div key={warning}><CircleAlert size={16} /><span>{warning}</span></div>
          ))}
        </div>
      )}

      {generatedText && (
        <article className="generated-card">
          <div className="generated-heading">
            <span><Sparkles size={16} /> 本地 Phi-4 辅助整理</span>
            <small>请依据下方引用核验</small>
          </div>
          <p>{generatedText}</p>
        </article>
      )}

      {hits.length ? (
        <div className="evidence-layout">
          <div className="evidence-list">
            {hits.map((hit) => (
              <EvidenceCard
                key={hit.chunk.chunk_id}
                hit={hit}
                copied={copied === hit.chunk.chunk_id}
                onCopy={() => void copyText(hit.chunk.chunk_id, hit.chunk.text)}
                query={response.query}
              />
            ))}
          </div>
          <aside className="result-aside" aria-label="结果说明">
            <div className="aside-card">
              <div className="aside-title"><Gauge size={16} /> 排名说明</div>
              <p>
                {response.ranking_score_calibrated
                  ? '当前分数已经过校准，可作为相关度参考。'
                  : '当前分数用于结果排序，未经概率校准，不代表事实置信度。'}
              </p>
            </div>
            <div className="aside-card">
              <div className="aside-title"><ShieldCheck size={16} /> 证据原则</div>
              <ul>
                <li>优先核对原文与版本</li>
                <li>交叉验证多个独立来源</li>
                <li>语料内容不会被执行</li>
              </ul>
            </div>
          </aside>
        </div>
      ) : (
        <div className="empty-results">
          <FileSearch size={28} />
          <strong>未找到匹配证据</strong>
          <p>尝试使用 CVE/CWE 编号、函数名或更具体的安全术语。</p>
        </div>
      )}
    </section>
  )
}

function evidenceToHit(item: EvidenceItem, index: number): SearchHit {
  return {
    chunk: {
      chunk_id: item.evidence_id,
      source_id: item.citation.source_id,
      source_version: item.citation.source_version,
      text: item.text,
      citation: item.citation,
    },
    score: item.score,
    rank: index + 1,
    channels: item.channels ?? {},
    graph_paths: item.graph_paths ?? [],
    explanation: item.explanation,
  }
}

const CODE_HINT =
  /^\s|\{|\}|;\s*$|\b0x[0-9a-f]+\b|\btext:|\b(?:mov|call|push|pop|ret|retn|jmp|leave|lea|add|sub|xor|cmp|test|fgets|printf|puts|read|write|malloc|free|eax|ebx|ecx|edx|esi|edi|ebp|esp|dword|qword)\b/i

const CJK_HINT = /[一-鿿]/
const FENCE_HINT = /^```/
const LIST_HINT = /^\s*(?:\d+[.)]|[-*+])\s+/
const HEADING_HINT = /^\s*#{1,6}\s+/
const TABLE_HINT = /^\s*\|.*\|\s*$/

type EvidenceBlock = { kind: 'prose' | 'code' | 'heading'; text: string }

function splitEvidence(text: string): EvidenceBlock[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const blocks: EvidenceBlock[] = []
  let prose = ''
  let code: string[] = []
  let inCode = false
  let inFence = false
  let fence: string[] = []
  const flushProse = () => {
    const value = prose.trim()
    if (value) blocks.push({ kind: 'prose', text: value })
    prose = ''
  }
  const flushCode = () => {
    if (code.length) blocks.push({ kind: 'code', text: code.join('\n') })
    code = []
    inCode = false
  }
  for (const raw of lines) {
    const line = raw.trimEnd()
    if (FENCE_HINT.test(line.trim())) {
      if (inFence) {
        if (fence.length) blocks.push({ kind: 'code', text: fence.join('\n') })
        fence = []
        inFence = false
      } else {
        flushProse()
        flushCode()
        inFence = true
      }
      continue
    }
    if (inFence) {
      fence.push(raw)
      continue
    }
    if (!line.trim()) {
      if (!inCode) flushProse()
      else code.push('')
      continue
    }
    if (HEADING_HINT.test(line)) {
      flushProse()
      flushCode()
      blocks.push({ kind: 'heading', text: line.replace(HEADING_HINT, '') })
      continue
    }
    if (LIST_HINT.test(line)) {
      flushProse()
      flushCode()
      blocks.push({ kind: 'prose', text: line.trim() })
      continue
    }
    if (TABLE_HINT.test(line)) {
      // Markdown 表格按等宽行渲染，至少保持列对齐；跳过分隔行
      if (/^\s*\|[\s:|-]+\|\s*$/.test(line)) continue
      flushProse()
      code.push(raw)
      inCode = true
      continue
    }
    const explicitCode = CODE_HINT.test(line)
    // 已经在代码块里时，短的、不含中文的行视为续行（汇编写法千奇百怪）
    const continuation = inCode && !CJK_HINT.test(line) && line.trim().length <= 48
    if (explicitCode || continuation) {
      flushProse()
      code.push(raw)
      inCode = true
    } else {
      flushCode()
      prose = prose ? `${prose} ${line.trim()}` : line.trim()
    }
  }
  if (inFence && fence.length) blocks.push({ kind: 'code', text: fence.join('\n') })
  flushProse()
  flushCode()
  return blocks
}

function queryTerms(query: string): string[] {
  const terms = query
    .split(/[\s，。、；：？！?,;:!()（）[\]【】]+/)
    .map((term) => term.trim())
    .filter((term) => term.length >= 2)
  return Array.from(new Set(terms)).slice(0, 6)
}

function highlightTerms(text: string, terms: string[]): ReactNode {
  const usable = terms.filter((term) => term.length >= 2)
  if (!usable.length) return text
  const escaped = usable.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const matcher = new RegExp(`(${escaped.join('|')})`, 'gi')
  const lowered = new Set(usable.map((term) => term.toLowerCase()))
  return (
    <>
      {text.split(matcher).map((part, index) =>
        lowered.has(part.toLowerCase()) ? (
          <mark key={index}>{part}</mark>
        ) : (
          <span key={index}>{part}</span>
        ),
      )}
    </>
  )
}

const INLINE_HINT = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g

function renderInline(text: string, terms: string[]): ReactNode {
  const parts = text.split(INLINE_HINT)
  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={index}>{highlightTerms(part.slice(2, -2), terms)}</strong>
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return <code key={index} className="inline-code">{part.slice(1, -1)}</code>
        }
        const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(part)
        if (link) return <span key={index}>{link[1]}</span>
        return <span key={index}>{highlightTerms(part, terms)}</span>
      })}
    </>
  )
}

function EvidenceText({
  text,
  expanded,
  terms,
}: {
  text: string
  expanded: boolean
  terms: string[]
}) {
  const blocks = useMemo(() => splitEvidence(text), [text])
  return (
    <div className={expanded ? 'evidence-body' : 'evidence-body collapsed'}>
      {blocks.map((block, index) =>
        block.kind === 'code' ? (
          <pre key={index} className="evidence-code">{highlightTerms(block.text, terms)}</pre>
        ) : block.kind === 'heading' ? (
          <p key={index} className="evidence-heading">{renderInline(block.text, terms)}</p>
        ) : (
          <p key={index}>{renderInline(block.text, terms)}</p>
        ),
      )}
    </div>
  )
}

function EvidenceCard({
  hit,
  copied,
  onCopy,
  query,
}: {
  hit: SearchHit
  copied: boolean
  onCopy: () => void
  query: string
}) {
  const [expanded, setExpanded] = useState(false)
  const [graphOpen, setGraphOpen] = useState(false)
  const [graph, setGraph] = useState<GraphResponse | null>(null)
  const [graphError, setGraphError] = useState<string | null>(null)
  const citation = hit.chunk.citation
  const channels = Object.entries(hit.channels).sort((left, right) => right[1] - left[1])
  const previewUrl = pagePreviewUrl(citation)
  const terms = useMemo(() => queryTerms(query), [query])

  useEffect(() => {
    if (!graphOpen || graph || graphError) return
    const controller = new AbortController()
    fetchChunkGraph(hit.chunk.chunk_id, 1, controller.signal)
      .then(setGraph)
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setGraphError(error instanceof Error ? error.message : '查询子图加载失败。')
      })
    return () => controller.abort()
  }, [graph, graphError, graphOpen, hit.chunk.chunk_id])

  return (
    <article className="evidence-card">
      <header>
        <div className="rank-badge">{String(hit.rank).padStart(2, '0')}</div>
        <div className="citation-heading">
          <strong>{citation.title}</strong>
          <span><BookOpen size={13} /> {formatLocation(citation)}</span>
        </div>
        <div className="score-block">
          <span>排序分</span>
          <strong>{formatScore(hit.score)}</strong>
        </div>
      </header>
      <EvidenceText text={hit.chunk.text} expanded={expanded} terms={terms} />
      {hit.chunk.text.length > 320 && (
        <button className="expand-button" type="button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? '收起原文' : '展开原文'}
        </button>
      )}
      <footer>
        <div className="channel-list" aria-label="召回通道">
          {channels.length ? (
            channels.slice(0, 4).map(([name, score]) => (
              <span key={name}>{name}<b>{formatScore(score)}</b></span>
            ))
          ) : (
            <span>evidence</span>
          )}
        </div>
        <div className="evidence-actions">
          {previewUrl && (
            <a
              className="page-preview-link"
              href={previewUrl}
              target="_blank"
              rel="noreferrer"
            >
              <BookOpen size={14} /> 查看原页
            </a>
          )}
          <button className="copy-button" type="button" onClick={onCopy}>
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? '已复制' : '复制证据'}
          </button>
        </div>
      </footer>
      <details className="graph-detail" onToggle={(event) => setGraphOpen(event.currentTarget.open)}>
        <summary><Network size={14} /> 查看查询子图与排名解释</summary>
        {hit.explanation && <p>{hit.explanation}</p>}
        {hit.graph_paths.map((path, pathIndex) => (
          <div className="graph-path" key={`${path.join('-')}-${pathIndex}`}>
            {path.map((part, partIndex) => (
              <span key={`${part}-${partIndex}`}>
                <b>{part}</b>{partIndex < path.length - 1 && <ArrowRight size={12} />}
              </span>
            ))}
          </div>
        ))}
        {graphOpen && !graph && !graphError && (
          <div className="graph-loading"><LoaderCircle size={14} className="spin" /> 加载查询相关子图…</div>
        )}
        {graphError && <div className="graph-error">{graphError}</div>}
        {graph && <MiniGraph graph={graph} seedId={hit.chunk.chunk_id} />}
      </details>
    </article>
  )
}

function MiniGraph({ graph, seedId }: { graph: GraphResponse; seedId: string }) {
  const nodes = graph.nodes.slice(0, 18)
  const allowed = new Set(nodes.map((node) => node.id))
  const edges = graph.edges.filter(
    (edge) => allowed.has(edge.source) && allowed.has(edge.target),
  )
  if (!nodes.length) return <div className="graph-error">当前块没有可显示的图邻居。</div>
  const centerX = 210
  const centerY = 105
  const radiusX = 165
  const radiusY = 72
  const positioned = nodes.map((node, index) => ({
    ...node,
    x: nodes.length === 1 ? centerX : centerX + radiusX * Math.cos((index * Math.PI * 2) / nodes.length),
    y: nodes.length === 1 ? centerY : centerY + radiusY * Math.sin((index * Math.PI * 2) / nodes.length),
  }))
  const byId = new Map(positioned.map((node) => [node.id, node]))
  return (
    <div className="mini-graph">
      <svg viewBox="0 0 420 210" role="img" aria-label={`查询子图，${nodes.length} 个节点，${edges.length} 条边`}>
        {edges.map((edge) => {
          const source = byId.get(edge.source)
          const target = byId.get(edge.target)
          if (!source || !target) return null
          return <line key={edge.id} x1={source.x} y1={source.y} x2={target.x} y2={target.y} />
        })}
        {positioned.map((node) => {
          const label = node.label || node.name || node.title || node.kind || node.id
          const seed = node.id === seedId
          return (
            <g key={node.id} transform={`translate(${node.x} ${node.y})`}>
              <circle r={seed ? 11 : 8} className={seed ? 'seed' : ''} />
              <title>{label}</title>
              <text y={18}>{label.length > 18 ? `${label.slice(0, 17)}…` : label}</text>
            </g>
          )
        })}
      </svg>
      <span>显示 {nodes.length}/{graph.nodes.length} 个节点 · {edges.length} 条关系</span>
    </div>
  )
}

function SourcesView() {
  const [sources, setSources] = useState<GovernedSource[]>([])
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState('')
  const [audit, setAudit] = useState<DocumentPageAudit | null>(null)
  const [attentionOnly, setAttentionOnly] = useState(true)
  const [auditLoading, setAuditLoading] = useState(true)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<IndexJob | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadNotice, setUploadNotice] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    Promise.all([fetchSources(controller.signal), fetchDocuments(controller.signal, false)])
      .then(([manifest, nextDocuments]) => {
        setSources(manifest.sources)
        setDocuments(nextDocuments)
        setSelectedDocumentId((current) => current || nextDocuments[0]?.source_id || '')
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setError(reason instanceof Error ? reason.message : '知识源状态加载失败。')
      })
      .finally(() => setLoading(false))
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      fetchJob(job.job_id, controller.signal)
        .then(setJob)
        .catch((reason: unknown) => {
          if (reason instanceof DOMException && reason.name === 'AbortError') return
          setError(reason instanceof Error ? reason.message : '索引作业状态读取失败。')
        })
    }, 1000)
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [job])

  useEffect(() => {
    if (job?.status !== 'succeeded') return
    const controller = new AbortController()
    fetchDocuments(controller.signal, false)
      .then((nextDocuments) => {
        setDocuments(nextDocuments)
        setSelectedDocumentId((current) => current || nextDocuments[0]?.source_id || '')
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setError(reason instanceof Error ? reason.message : '已完成索引，但文档状态刷新失败。')
      })
    return () => controller.abort()
  }, [job?.status])

  useEffect(() => {
    if (!selectedDocumentId) return
    const controller = new AbortController()
    fetchDocumentPageAudit(
      selectedDocumentId,
      attentionOnly,
      0,
      100,
      controller.signal,
    )
      .then(setAudit)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setError(reason instanceof Error ? reason.message : '逐页质量审计加载失败。')
      })
      .finally(() => setAuditLoading(false))
    return () => controller.abort()
  }, [attentionOnly, selectedDocumentId])

  const loadMoreAudit = async () => {
    if (!audit || auditLoading) return
    setAuditLoading(true)
    setError(null)
    try {
      const next = await fetchDocumentPageAudit(
        audit.source_id,
        attentionOnly,
        audit.pages.length,
        audit.limit,
      )
      setAudit({ ...next, pages: [...audit.pages, ...next.pages] })
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : '更多页质量记录加载失败。')
    } finally {
      setAuditLoading(false)
    }
  }

  const startIndex = async () => {
    setError(null)
    try {
      setJob(await enqueueIndex())
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : '索引作业创建失败。')
    }
  }
  const onUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? [])
    event.target.value = ''
    if (!selected.length) return
    setUploading(true)
    setUploadNotice(null)
    setError(null)
    try {
      const result = await uploadDocuments(selected)
      if (result.job_id) setJob(await fetchJob(result.job_id))
      setUploadNotice(`已接收 ${result.files.length} 个文件，等待索引 worker 处理。`)
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : '文件上传失败。')
    } finally {
      setUploading(false)
    }
  }
  const defaultSources = sources.filter((source) => source.risk_state === 'default')
  const activeDocumentCount = documents.filter((document) => document.active).length
  const quarantinedSources = sources.filter((source) => source.risk_state === 'quarantine')
  const publishedCount = sources.filter((source) => source.current_snapshot).length
  const indexing = job && ['queued', 'running'].includes(job.status)

  return (
    <div className="section-page sources-page">
      <section className="section-heading">
        <div>
          <span className="page-kicker">KNOWLEDGE FOUNDATION</span>
          <h1>知识源</h1>
          <p>本地文档与受治理公开资料共同构成检索底座；网络获取始终需要独立的显式操作。</p>
        </div>
        <div className="governance-badge"><ShieldCheck size={17} /> 来源治理已启用</div>
      </section>

      <div className="source-grid">
        <article className="source-card primary-source">
          <header>
            <span className="source-icon"><BookOpen size={20} /></span>
            <span className="source-state local"><span /> LOCAL CORPUS</span>
          </header>
          <h2>本地安全资料库</h2>
          <p>PDF、DOCX、Markdown、TXT 与 Python 源码按原始位置建立可追溯引用。</p>
          <ul>
            <li><Check size={14} /> 已索引 {activeDocumentCount} 本；已审计 {documents.length} 本</li>
            <li><Check size={14} /> Python AST 只解析、不执行</li>
            <li><Check size={14} /> 原始文件始终保留版本哈希</li>
          </ul>
          <div className="source-actions">
            <button className="source-action" type="button" onClick={startIndex} disabled={Boolean(indexing) || uploading}>
              {indexing ? <LoaderCircle size={14} className="spin" /> : <Database size={14} />}
              {job?.status === 'queued' ? '等待 worker' : job?.status === 'running' ? `索引中 ${Math.round(job.progress * 100)}%` : '开始增量索引'}
            </button>
            <label className="source-action secondary" aria-label="上传本地文档">
              {uploading ? <LoaderCircle size={14} className="spin" /> : <Upload size={14} />}
              {uploading ? '上传中…' : '上传文档'}
              <input type="file" multiple accept=".pdf,.docx,.txt,.py,.md" onChange={onUpload} disabled={uploading || Boolean(indexing)} />
            </label>
          </div>
          <div className="source-foot"><span>目录</span><code>book/</code></div>
        </article>

        <article className="source-card">
          <header>
            <span className="source-icon blue"><Database size={20} /></span>
            <span className="source-state governed"><span /> GOVERNED</span>
          </header>
          <h2>权威漏洞数据</h2>
          <p>manifest 中 {defaultSources.length} 个默认来源；已有 {publishedCount} 个活动快照。</p>
          <div className="tag-cloud">
            {defaultSources.slice(0, 6).map((source) => <span key={source.id}>{source.id.toUpperCase()}</span>)}
          </div>
          <div className="source-foot"><span>策略</span><strong>白名单 · 哈希 · 许可证</strong></div>
        </article>

        <article className="source-card">
          <header>
            <span className="source-icon purple"><GitFork size={20} /></span>
            <span className="source-state quarantine"><span /> QUARANTINED</span>
          </header>
          <h2>公开安全仓库</h2>
          <p>{quarantinedSources.length} 个来源要求隔离或人工确认；脚本与二进制不会被执行。</p>
          <ul>
            <li><Check size={14} /> 不运行 Git hook 或仓库代码</li>
            <li><Check size={14} /> 内容哈希与来源版本绑定</li>
            <li><Check size={14} /> 未知许可证默认拒绝发布</li>
          </ul>
          <div className="source-foot"><span>网络</span><strong>离线时完全禁用</strong></div>
        </article>
      </div>

      <section className="audit-panel" aria-labelledby="page-audit-title">
        <div className="audit-heading">
          <div>
            <span className="capability-icon cyan"><FileSearch size={19} /></span>
            <div>
              <h2 id="page-audit-title">逐页提取质量</h2>
              <p>坏文本页不会进入检索；即使 OCR 尚未安装，也可以打开原始 PDF 页核对。</p>
            </div>
          </div>
          <label className="audit-filter">
            <input
              type="checkbox"
              checked={attentionOnly}
              onChange={(event) => {
                setAuditLoading(true)
                setAttentionOnly(event.target.checked)
              }}
            />
            仅显示需关注页
          </label>
        </div>
        <label className="audit-document-select">
          <span>文档</span>
          <select
            value={selectedDocumentId}
            onChange={(event) => {
              setAuditLoading(true)
              setSelectedDocumentId(event.target.value)
            }}
          >
            {documents.map((document) => (
              <option value={document.source_id} key={`${document.source_id}:${document.version_id}`}>
                {document.title}{document.active ? '' : ' · 待 OCR'}
              </option>
            ))}
          </select>
        </label>

        {audit && (
          <div className="audit-stats" aria-label="页质量计数">
            <span><b>{audit.counts.total}</b>总页数</span>
            <span className="native"><b>{audit.counts.native + audit.counts.ocr}</b>可索引</span>
            <span className="required"><b>{audit.counts.ocr_required}</b>待 OCR</span>
            <span className="quarantine"><b>{audit.counts.quarantine}</b>已隔离</span>
          </div>
        )}

        {auditLoading && !audit ? (
          <div className="audit-empty"><LoaderCircle size={16} className="spin" /> 读取逐页审计…</div>
        ) : audit && audit.pages.length > 0 ? (
          <div className="audit-page-list">
            {audit.pages.map((page) => {
              const preview = pageAuditPreviewUrl(audit.relative_path, page.page)
              return (
                <article className={`audit-page-row ${page.indexable ? 'indexable' : 'attention'}`} key={page.page}>
                  <div className="audit-page-number"><small>PAGE</small><strong>{page.page}</strong></div>
                  <div className="audit-page-copy">
                    <div>
                      <strong>{page.indexable ? '可索引' : page.disposition === 'quarantine' ? '已隔离' : '待 OCR'}</strong>
                      <span>{page.parser} · 置信度 {Math.round(page.extraction_confidence * 100)}%</span>
                    </div>
                    {page.diagnostics.length > 0 && <p>{page.diagnostics.join(' · ')}</p>}
                  </div>
                  {preview && (
                    <a href={preview} target="_blank" rel="noreferrer" className="audit-preview-link">
                      <Link2 size={13} /> 查看原页
                    </a>
                  )}
                </article>
              )
            })}
          </div>
        ) : audit ? (
          <div className="audit-empty">
            <CircleCheck size={16} /> {attentionOnly ? '当前文档没有需关注页。' : '当前文档没有逐页审计记录。'}
          </div>
        ) : (
          <div className="audit-empty">尚无可读取的文档审计。</div>
        )}

        {audit && audit.pages.length < audit.total_matching && (
          <button className="audit-load-more" type="button" onClick={loadMoreAudit} disabled={auditLoading}>
            {auditLoading && <LoaderCircle size={13} className="spin" />}
            继续加载（{audit.pages.length}/{audit.total_matching}）
          </button>
        )}
      </section>

      <section className="pipeline-panel">
        <div className="pipeline-copy">
          <span className="capability-icon cyan"><Layers3 size={19} /></span>
          <div><h2>不可变快照管线</h2><p>外部内容只有经过完整校验后，才会原子切换为活动版本。</p></div>
        </div>
        <div className="pipeline-flow" aria-label="来源处理流程">
          {['白名单', '隔离区', '规范化', '去重', '索引', '发布'].map((step, index) => (
            <span key={step}><b>{String(index + 1).padStart(2, '0')}</b>{step}{index < 5 && <ArrowRight size={13} />}</span>
          ))}
        </div>
        <div className="source-catalog">
          {loading ? (
            <span><LoaderCircle size={14} className="spin" /> 读取来源清单…</span>
          ) : sources.map((source) => (
            <span key={source.id}>
              <b>{source.name}</b>
              <small>{source.license.expression} · {source.current_snapshot?.source_revision ?? '未发布'}</small>
            </span>
          ))}
        </div>
      </section>

      {(error || job?.error || uploadNotice) && (
        <div className="sources-note"><CircleAlert size={16} /><p>{error || job?.error || uploadNotice}</p></div>
      )}
      {!error && (
        <div className="sources-note">
          <CircleAlert size={16} />
          <p>此页只读取本地清单与控制面。点击索引会处理 book/，不会下载公开来源或启动生成式模型。</p>
        </div>
      )}
    </div>
  )
}

interface SettingsViewProps {
  mode: RuntimeMode
  setMode: (value: RuntimeMode) => void
  serverMode: RuntimeMode
  generativeAvailable: boolean
  phiEnabled: boolean
  setPhiEnabled: (value: boolean) => void
  topK: number
  setTopK: (value: number) => void
  capabilities: Record<string, boolean>
}

function SettingsView({
  mode,
  setMode,
  serverMode,
  generativeAvailable,
  phiEnabled,
  setPhiEnabled,
  topK,
  setTopK,
  capabilities,
}: SettingsViewProps) {
  return (
    <div className="section-page settings-page">
      <section className="section-heading">
        <div>
          <span className="page-kicker">RUNTIME POLICY</span>
          <h1>运行设置</h1>
          <p>本页设置仅影响当前浏览器会话中的搜索请求，不会安装、下载或卸载模型。</p>
        </div>
      </section>

      <section className="settings-section" aria-labelledby="runtime-mode-title">
        <div className="settings-section-heading">
          <div><h2 id="runtime-mode-title">检索运行模式</h2><p>选择与当前资源及比赛规则匹配的能力边界。</p></div>
          <span className="required-label">当前：{MODE_DETAILS[mode].label}</span>
        </div>
        <div className="mode-grid">
          {(Object.keys(MODE_DETAILS) as RuntimeMode[]).map((value) => {
            const detail = MODE_DETAILS[value]
            const Icon = detail.icon
            const selected = mode === value
            const unavailable = MODE_CAPABILITY[value] > MODE_CAPABILITY[serverMode]
            return (
              <button
                key={value}
                type="button"
                className={selected ? 'mode-card selected' : 'mode-card'}
                aria-pressed={selected}
                disabled={unavailable}
                onClick={() => setMode(value)}
              >
                <span className="mode-card-icon"><Icon size={20} /></span>
                <span className="mode-card-copy"><strong>{detail.label}</strong><small>{detail.description}</small></span>
                <span className="radio-mark">{selected && <Check size={13} />}</span>
              </button>
            )
          })}
        </div>
      </section>

      <section className="settings-section" aria-labelledby="capability-title">
        <div className="settings-section-heading">
          <div><h2 id="capability-title">实际运行能力</h2><p>这里反映当前机器上已安装并可被离线运行时加载的组件，而不是配置开关。</p></div>
        </div>
        <div className="capability-grid">
          {([
            ['llamaindex', 'LlamaIndex 统一解析'],
            ['unstructured', 'Unstructured 文档结构'],
            ['tree_sitter', 'tree-sitter 代码结构'],
            ['bge_m3', 'BAAI bge-m3 向量'],
            ['reranker', 'bge-reranker 重排'],
            ['tesseract', 'Tesseract 离线 OCR'],
            ['milvus_driver', 'Milvus 混合索引驱动'],
            ['neo4j_driver', 'Neo4j 图存储驱动'],
            ['deterministic_reasoning', 'NetworkX 确定性推理图'],
          ] as [string, string][]).map(([key, label]) => {
            const ready = capabilities[key] === true
            return <span className={ready ? 'state-badge on' : 'state-badge'} key={key}><span />{label} · {ready ? '可用' : '未就绪'}</span>
          })}
        </div>
      </section>

      <section className="settings-section model-settings" aria-labelledby="local-model-title">
        <div className="settings-section-heading">
          <div><h2 id="local-model-title">本地生成模型</h2><p>Phi-4 是外部可挂载能力，不属于 SecKnow 必需组件。</p></div>
          <span className={phiEnabled ? 'state-badge on' : 'state-badge'}>
            <span /> {phiEnabled ? '当前请求启用' : '默认关闭'}
          </span>
        </div>
        <div className="model-control">
          <div className="model-identity">
            <span className="model-glyph"><Cpu size={25} /></span>
            <div><strong>Phi-4 · 本地 OpenAI 兼容端点</strong><small>由后端配置端点与模型文件；前端不接入模型 SDK。</small></div>
          </div>
          <label className="large-toggle">
            <input
              type="checkbox"
              checked={phiEnabled}
              disabled={!generativeAvailable}
              onChange={(event) => {
                if (!generativeAvailable) return
                setPhiEnabled(event.target.checked)
                if (event.target.checked) setMode('llm')
              }}
            />
            <span className="toggle-track" aria-hidden="true"><span /></span>
            <span>{phiEnabled ? '允许本次会话请求生成' : '禁止生成式请求'}</span>
          </label>
        </div>
        <div className="boundary-list">
          <span><CircleCheck size={15} /> 关闭时搜索不发送生成请求</span>
          <span><CircleCheck size={15} /> 仅允许后端配置的本地/私有地址</span>
          <span><CircleCheck size={15} /> 服务不可用时不回退到云端</span>
        </div>
      </section>

      <section className="settings-section retrieval-settings" aria-labelledby="result-count-title">
        <div className="settings-section-heading">
          <div><h2 id="result-count-title">返回证据数量</h2><p>控制单次请求的 top_k，较大值会增加重排和阅读成本。</p></div>
          <output htmlFor="top-k-slider">{topK} 条</output>
        </div>
        <div className="slider-wrap">
          <span>1</span>
          <input
            id="top-k-slider"
            type="range"
            min="1"
            max="30"
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
            style={{ '--slider-value': `${((topK - 1) / 29) * 100}%` } as React.CSSProperties}
          />
          <span>30</span>
        </div>
      </section>

      <aside className="competition-callout">
        <span><TerminalSquare size={20} /></span>
        <div><strong>比赛前检查</strong><p>切换到“严格词法”或“竞赛模式”，确认 Phi-4 显示为关闭，并遵守赛事对嵌入模型的具体定义。</p></div>
      </aside>
    </div>
  )
}

export default App
