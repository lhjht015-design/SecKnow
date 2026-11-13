import type {
  ChainRequest,
  ChainResponse,
  Citation,
  DocumentPageAudit,
  DocumentSummary,
  EvidenceAnswer,
  EvidenceItem,
  EvidenceSection,
  GovernedSource,
  GraphEdge,
  GraphNode,
  GraphResponse,
  IndexJob,
  RuntimeConfig,
  RuntimeMode,
  SearchHit,
  SearchRequest,
  SearchResponse,
  SourcesResponse,
  UploadResponse,
} from './types'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null

const stringValue = (value: unknown, fallback = ''): string =>
  typeof value === 'string' ? value : fallback

const numberValue = (value: unknown, fallback = 0): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback

const stringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []

const RUNTIME_MODES = new Set<RuntimeMode>([
  'strict-lexical',
  'competition',
  'enhanced',
  'llm',
])

const runtimeMode = (value: unknown): RuntimeMode | null =>
  typeof value === 'string' && RUNTIME_MODES.has(value as RuntimeMode)
    ? value as RuntimeMode
    : null

function decodeCitation(value: unknown): Citation {
  const raw = isRecord(value) ? value : {}
  return {
    source_id: stringValue(raw.source_id, 'unknown'),
    source_version: stringValue(raw.source_version, 'unknown'),
    title: stringValue(raw.title, '未命名来源'),
    relative_path: typeof raw.relative_path === 'string' ? raw.relative_path : null,
    page_start: typeof raw.page_start === 'number' ? raw.page_start : null,
    page_end: typeof raw.page_end === 'number' ? raw.page_end : null,
    line_start: typeof raw.line_start === 'number' ? raw.line_start : null,
    line_end: typeof raw.line_end === 'number' ? raw.line_end : null,
    section_path: stringArray(raw.section_path),
    url: typeof raw.url === 'string' ? raw.url : null,
    extraction_confidence: numberValue(raw.extraction_confidence, 1),
  }
}

function decodePaths(value: unknown): string[][] {
  if (!Array.isArray(value)) return []
  return value
    .filter(Array.isArray)
    .map((path) => path.filter((part): part is string => typeof part === 'string'))
    .filter((path) => path.length > 0)
}

function decodeHit(value: unknown, index: number): SearchHit | null {
  if (!isRecord(value)) return null
  const chunk = isRecord(value.chunk) ? value.chunk : value
  const citation = decodeCitation(chunk.citation ?? value.citation)
  const text = stringValue(chunk.text ?? value.text)
  if (!text) return null

  const channels = isRecord(value.channels)
    ? Object.fromEntries(
        Object.entries(value.channels).filter(
          (entry): entry is [string, number] =>
            typeof entry[1] === 'number' && Number.isFinite(entry[1]),
        ),
      )
    : {}

  return {
    chunk: {
      chunk_id: stringValue(chunk.chunk_id, `result-${index + 1}`),
      source_id: stringValue(chunk.source_id, citation.source_id),
      source_version: stringValue(chunk.source_version, citation.source_version),
      text,
      section_path: stringArray(chunk.section_path),
      order: numberValue(chunk.order),
      token_count: numberValue(chunk.token_count),
      previous_chunk_id: typeof chunk.previous_chunk_id === 'string' ? chunk.previous_chunk_id : null,
      next_chunk_id: typeof chunk.next_chunk_id === 'string' ? chunk.next_chunk_id : null,
      citation,
      metadata: isRecord(chunk.metadata) ? chunk.metadata : {},
    },
    score: numberValue(value.score),
    rank: numberValue(value.rank, index + 1),
    channels,
    graph_paths: decodePaths(value.graph_paths),
    explanation: typeof value.explanation === 'string' ? value.explanation : null,
  }
}

function decodeEvidenceItem(value: unknown, index: number): EvidenceItem | null {
  if (!isRecord(value)) return null
  const text = stringValue(value.text)
  if (!text) return null
  return {
    evidence_id: stringValue(value.evidence_id, `E${index + 1}`),
    text,
    score: numberValue(value.score),
    citation: decodeCitation(value.citation),
    channels: isRecord(value.channels)
      ? Object.fromEntries(
          Object.entries(value.channels).filter(
            (entry): entry is [string, number] => typeof entry[1] === 'number',
          ),
        )
      : {},
    graph_paths: decodePaths(value.graph_paths),
    explanation: typeof value.explanation === 'string' ? value.explanation : null,
  }
}

function decodeAnswer(value: unknown): EvidenceAnswer | null {
  if (!isRecord(value)) return null
  const sections: EvidenceSection[] = Array.isArray(value.sections)
    ? value.sections.flatMap((section, sectionIndex) => {
        if (!isRecord(section)) return []
        const items = Array.isArray(section.items)
          ? section.items.flatMap((item, itemIndex) => {
              const decoded = decodeEvidenceItem(item, itemIndex)
              return decoded ? [decoded] : []
            })
          : []
        return [
          {
            title: stringValue(section.title, `证据组 ${sectionIndex + 1}`),
            score: numberValue(section.score),
            items,
          },
        ]
      })
    : []

  return {
    query: stringValue(value.query),
    mode: stringValue(value.mode),
    summary: stringValue(value.summary),
    sections,
    warnings: stringArray(value.warnings),
    elapsed_ms: numberValue(value.elapsed_ms),
    generated: value.generated === true,
  }
}

function decodeResponse(payload: unknown): SearchResponse {
  if (!isRecord(payload)) throw new Error('服务返回了无法识别的数据格式。')
  const nested = isRecord(payload.data) ? payload.data : payload
  const hitValue = nested.hits ?? nested.results
  const hits = Array.isArray(hitValue)
    ? hitValue.flatMap((hit, index) => {
        const decoded = decodeHit(hit, index)
        return decoded ? [decoded] : []
      })
    : []
  const answer = decodeAnswer(nested.answer ?? nested.evidence)

  return {
    query: stringValue(nested.query),
    mode: stringValue(nested.mode),
    hits,
    elapsed_ms: numberValue(nested.elapsed_ms),
    ranking_score_calibrated: nested.ranking_score_calibrated === true,
    warnings: [...stringArray(nested.warnings), ...(answer?.warnings ?? [])],
    answer,
    generated_answer:
      typeof nested.generated_answer === 'string' ? nested.generated_answer : null,
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (isRecord(body)) {
      if (typeof body.detail === 'string') return body.detail
      if (typeof body.message === 'string') return body.message
      if (Array.isArray(body.detail)) {
        const messages = body.detail.flatMap((item) => {
          if (!isRecord(item)) return []
          return typeof item.msg === 'string' ? [item.msg] : []
        })
        if (messages.length) return messages.join('；')
      }
    }
  } catch {
    // Fall through to the status text when an error body is not JSON.
  }
  return response.statusText || `HTTP ${response.status}`
}

export async function fetchChunkGraph(
  chunkId: string,
  radius = 1,
  signal?: AbortSignal,
): Promise<GraphResponse> {
  const response = await fetch(
    `/api/graph/${encodeURIComponent(chunkId)}?radius=${radius}&limit=80`,
    { headers: { Accept: 'application/json' }, signal },
  )
  if (!response.ok) throw new Error(await errorMessage(response))
  const payload: unknown = await response.json()
  if (!isRecord(payload)) throw new Error('图接口返回了无法识别的数据格式。')
  const nodes: GraphNode[] = Array.isArray(payload.nodes)
    ? payload.nodes.flatMap((value) => {
        if (!isRecord(value) || typeof value.id !== 'string') return []
        return [{
          id: value.id,
          kind: stringValue(value.kind),
          label: stringValue(value.label),
          name: stringValue(value.name),
          title: stringValue(value.title),
        }]
      })
    : []
  const edges: GraphEdge[] = Array.isArray(payload.edges)
    ? payload.edges.flatMap((value) => {
        if (
          !isRecord(value) ||
          typeof value.id !== 'string' ||
          typeof value.source !== 'string' ||
          typeof value.target !== 'string'
        ) return []
        return [{
          id: value.id,
          source: value.source,
          target: value.target,
          kind: stringValue(value.kind),
          weight: typeof value.weight === 'number' ? value.weight : undefined,
        }]
      })
    : []
  return { nodes, edges }
}

export async function fetchRuntimeConfig(signal?: AbortSignal): Promise<RuntimeConfig> {
  const response = await fetch('/api/config', {
    headers: { Accept: 'application/json' },
    signal,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  const payload: unknown = await response.json()
  if (!isRecord(payload)) throw new Error('配置接口返回了无法识别的数据格式。')
  const mode = runtimeMode(payload.mode)
  if (!mode) throw new Error('配置接口返回了未知的运行模式。')
  return {
    mode,
    offline: payload.offline === true,
    semantic_enabled: payload.semantic_enabled === true,
    semantic_stack_installed: payload.semantic_stack_installed === true,
    effective_semantic_enabled: payload.effective_semantic_enabled === true,
    milvus_service: typeof payload.milvus_service === 'string' ? payload.milvus_service : 'unknown',
    generative_enabled: payload.generative_enabled === true,
    reranker_enabled: payload.reranker_enabled === true,
    effective_reranker_enabled: payload.effective_reranker_enabled === true,
    deterministic_reasoning_enabled: payload.deterministic_reasoning_enabled === true,
    capabilities: isRecord(payload.capabilities)
      ? Object.fromEntries(
          Object.entries(payload.capabilities).filter(
            (entry): entry is [string, boolean] => typeof entry[1] === 'boolean',
          ),
        )
      : {},
  }
}

export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  if (!files.length) throw new Error('请选择至少一个文件。')
  const form = new FormData()
  files.forEach((file) => form.append('files', file, file.name))
  form.append('auto_index', 'true')
  const response = await fetch('/api/documents/upload', {
    method: 'POST',
    headers: { Accept: 'application/json' },
    body: form,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  const payload: unknown = await response.json()
  if (!isRecord(payload) || !Array.isArray(payload.files)) {
    throw new Error('上传接口返回了无法识别的数据格式。')
  }
  return {
    files: payload.files.flatMap((value) => {
      if (!isRecord(value) || typeof value.relative_path !== 'string') return []
      return [{
        name: stringValue(value.name, value.relative_path),
        relative_path: value.relative_path,
        format: stringValue(value.format),
        size_bytes: numberValue(value.size_bytes),
        sha256: stringValue(value.sha256),
      }]
    }),
    job_id: typeof payload.job_id === 'string' ? payload.job_id : null,
    status: stringValue(payload.status, 'stored'),
  }
}

export async function fetchSources(signal?: AbortSignal): Promise<SourcesResponse> {
  const response = await fetch('/api/sources', { headers: { Accept: 'application/json' }, signal })
  if (!response.ok) throw new Error(await errorMessage(response))
  const payload: unknown = await response.json()
  if (!isRecord(payload)) throw new Error('来源接口返回了无法识别的数据格式。')
  const sources: GovernedSource[] = Array.isArray(payload.sources)
    ? payload.sources.flatMap((value) => {
        if (!isRecord(value) || typeof value.id !== 'string') return []
        const license = isRecord(value.license) ? value.license : {}
        const snapshot = isRecord(value.current_snapshot) ? value.current_snapshot : null
        const risk = value.risk_state
        if (risk !== 'default' && risk !== 'opt-in' && risk !== 'quarantine') return []
        return [{
          id: value.id,
          name: stringValue(value.name, value.id),
          description: stringValue(value.description),
          adapter: stringValue(value.adapter),
          risk_state: risk,
          license: {
            expression: stringValue(license.expression),
            name: stringValue(license.name),
            url: stringValue(license.url),
            attribution_required: license.attribution_required === true,
            redistribution_allowed: license.redistribution_allowed === true,
          },
          current_snapshot: snapshot ? {
            snapshot_id: stringValue(snapshot.snapshot_id),
            source_revision: stringValue(snapshot.source_revision),
            created_at: stringValue(snapshot.created_at),
            risk_state: stringValue(snapshot.risk_state),
          } : null,
        }]
      })
    : []
  return { version: numberValue(payload.version, 1), sources }
}

export async function fetchDocuments(
  signal?: AbortSignal,
  activeOnly = true,
): Promise<DocumentSummary[]> {
  const response = await fetch(`/api/documents?active_only=${activeOnly}`, {
    headers: { Accept: 'application/json' },
    signal,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  const payload: unknown = await response.json()
  if (!Array.isArray(payload)) throw new Error('文档接口返回了无法识别的数据格式。')
  return payload.flatMap((value) => {
    if (!isRecord(value) || typeof value.source_id !== 'string') return []
    return [{
      source_id: value.source_id,
      title: stringValue(value.title, '未命名文档'),
      relative_path: stringValue(value.relative_path),
      version_id: stringValue(value.version_id),
      active: value.active === 1 || value.active === true,
    }]
  })
}

export async function fetchDocumentPageAudit(
  sourceId: string,
  attentionOnly: boolean,
  offset = 0,
  limit = 100,
  signal?: AbortSignal,
): Promise<DocumentPageAudit> {
  const query = new URLSearchParams({
    attention_only: String(attentionOnly),
    offset: String(offset),
    limit: String(limit),
  })
  const response = await fetch(
    `/api/documents/${encodeURIComponent(sourceId)}/pages?${query}`,
    { headers: { Accept: 'application/json' }, signal },
  )
  if (!response.ok) throw new Error(await errorMessage(response))
  const payload: unknown = await response.json()
  if (!isRecord(payload) || !isRecord(payload.counts)) {
    throw new Error('页质量接口返回了无法识别的数据格式。')
  }
  const pages = Array.isArray(payload.pages)
    ? payload.pages.flatMap((value) => {
        if (!isRecord(value) || typeof value.page !== 'number') return []
        return [{
          page: value.page,
          disposition: stringValue(value.disposition, 'unknown'),
          indexable: value.indexable === true,
          parser: stringValue(value.parser),
          extraction_confidence: numberValue(value.extraction_confidence),
          diagnostics: stringArray(value.diagnostics),
        }]
      })
    : []
  return {
    source_id: stringValue(payload.source_id, sourceId),
    version_id: stringValue(payload.version_id),
    title: stringValue(payload.title, '未命名文档'),
    relative_path: stringValue(payload.relative_path),
    format: stringValue(payload.format),
    counts: {
      total: numberValue(payload.counts.total),
      native: numberValue(payload.counts.native),
      ocr: numberValue(payload.counts.ocr),
      ocr_required: numberValue(payload.counts.ocr_required),
      quarantine: numberValue(payload.counts.quarantine),
    },
    total_matching: numberValue(payload.total_matching),
    offset: numberValue(payload.offset),
    limit: numberValue(payload.limit, limit),
    pages,
  }
}

export async function enqueueIndex(): Promise<IndexJob> {
  const response = await fetch('/api/index', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ path: '.', recursive: true }),
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return decodeJob(await response.json())
}

export async function fetchJob(jobId: string, signal?: AbortSignal): Promise<IndexJob> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
    headers: { Accept: 'application/json' },
    signal,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return decodeJob(await response.json())
}

function decodeJob(value: unknown): IndexJob {
  if (!isRecord(value) || typeof value.job_id !== 'string') {
    throw new Error('作业接口返回了无法识别的数据格式。')
  }
  return {
    job_id: value.job_id,
    status: stringValue(value.status, 'unknown'),
    stage: typeof value.stage === 'string' ? value.stage : null,
    progress: numberValue(value.progress),
    error: typeof value.error === 'string' && value.error ? value.error : null,
  }
}

export async function searchKnowledge(
  request: SearchRequest,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }
  return decodeResponse(await response.json())
}

function decodeChainResponse(raw: Partial<ChainResponse>): ChainResponse {
  return {
    entries: Array.isArray(raw.entries) ? raw.entries : [],
    chains: Array.isArray(raw.chains) ? raw.chains : [],
    binary: raw.binary ?? null,
    warnings: Array.isArray(raw.warnings) ? raw.warnings : [],
  }
}

export async function recommendChain(
  request: ChainRequest,
  signal?: AbortSignal,
): Promise<ChainResponse> {
  const response = await fetch('/api/chain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }
  return decodeChainResponse(await response.json() as Partial<ChainResponse>)
}

export async function analyzeBinary(
  file: File,
  signal?: AbortSignal,
): Promise<ChainResponse> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch('/api/chain/binary', {
    method: 'POST',
    body: form,
    signal,
  })
  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }
  return decodeChainResponse(await response.json() as Partial<ChainResponse>)
}

export async function fetchChunk(chunkId: string, signal?: AbortSignal): Promise<SearchHit> {
  const response = await fetch(`/api/chunks/${encodeURIComponent(chunkId)}`, {
    headers: { Accept: 'application/json' },
    signal,
  })
  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }
  const hit = decodeHit(await response.json(), 0)
  if (!hit) throw new Error('块数据解析失败')
  return hit
}
