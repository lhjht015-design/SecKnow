export type RuntimeMode = 'strict-lexical' | 'competition' | 'enhanced' | 'llm'

export interface BoundingBox {
  x0: number
  y0: number
  x1: number
  y1: number
  page_width?: number | null
  page_height?: number | null
}

export interface Citation {
  source_id: string
  source_version: string
  title: string
  relative_path?: string | null
  page_start?: number | null
  page_end?: number | null
  line_start?: number | null
  line_end?: number | null
  section_path?: string[]
  url?: string | null
  bounding_boxes?: BoundingBox[]
  extraction_confidence?: number
}

export interface Chunk {
  chunk_id: string
  source_id: string
  source_version: string
  text: string
  content_sha256?: string
  element_ids?: string[]
  order?: number
  token_count?: number
  section_path?: string[]
  previous_chunk_id?: string | null
  next_chunk_id?: string | null
  citation: Citation
  metadata?: Record<string, unknown>
}

export interface SearchHit {
  chunk: Chunk
  score: number
  rank: number
  channels: Record<string, number>
  graph_paths: string[][]
  explanation?: string | null
}

export interface EvidenceItem {
  evidence_id: string
  text: string
  score: number
  citation: Citation
  channels?: Record<string, number>
  graph_paths?: string[][]
  explanation?: string | null
}

export interface EvidenceSection {
  title: string
  score: number
  items: EvidenceItem[]
}

export interface EvidenceAnswer {
  query: string
  mode: string
  summary: string
  sections: EvidenceSection[]
  warnings: string[]
  elapsed_ms: number
  generated: boolean
}

export interface SearchResponse {
  query: string
  mode: string
  hits: SearchHit[]
  elapsed_ms: number
  ranking_score_calibrated: boolean
  warnings: string[]
  answer?: EvidenceAnswer | null
  generated_answer?: string | null
}

export interface GraphNode {
  id: string
  kind?: string
  label?: string
  name?: string
  title?: string
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  kind?: string
  weight?: number
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface SourceLicense {
  expression: string
  name: string
  url: string
  attribution_required: boolean
  redistribution_allowed: boolean
}

export interface SourceSnapshot {
  snapshot_id: string
  source_revision: string
  created_at: string
  risk_state: string
}

export interface GovernedSource {
  id: string
  name: string
  description: string
  adapter: string
  risk_state: 'default' | 'opt-in' | 'quarantine'
  license: SourceLicense
  current_snapshot: SourceSnapshot | null
}

export interface SourcesResponse {
  version: number
  sources: GovernedSource[]
}

export interface DocumentSummary {
  source_id: string
  title: string
  relative_path: string
  version_id: string
  active: boolean
}

export interface PageAuditCounts {
  total: number
  native: number
  ocr: number
  ocr_required: number
  quarantine: number
}

export interface PageAuditItem {
  page: number
  disposition: string
  indexable: boolean
  parser: string
  extraction_confidence: number
  diagnostics: string[]
}

export interface DocumentPageAudit {
  source_id: string
  version_id: string
  title: string
  relative_path: string
  format: string
  counts: PageAuditCounts
  total_matching: number
  offset: number
  limit: number
  pages: PageAuditItem[]
}

export interface IndexJob {
  job_id: string
  status: string
  stage: string | null
  progress: number
  error: string | null
}

export interface RuntimeConfig {
  mode: RuntimeMode
  offline: boolean
  semantic_enabled: boolean
  semantic_stack_installed: boolean
  effective_semantic_enabled: boolean
  milvus_service: string
  generative_enabled: boolean
  reranker_enabled: boolean
  effective_reranker_enabled: boolean
  deterministic_reasoning_enabled: boolean
  capabilities: Record<string, boolean>
}

export interface UploadResponse {
  files: Array<{
    name: string
    relative_path: string
    format: string
    size_bytes: number
    sha256: string
  }>
  job_id: string | null
  status: string
}

export interface SearchRequest {
  query: string
  mode: RuntimeMode
  top_k: number
  use_llm: boolean
}

export interface ChainRequest {
  source?: string
  filename?: string
  relative_path?: string
  max_evidence?: number
}

export interface SourceEntryInfo {
  category: string
  technique: string
  line?: number | null
  text: string
  confidence: string
}

export interface ChainStepInfo {
  unit: string
  title: string
  status: string
  detail: string
}

export interface ChainEvidence {
  chunk_id: string
  score: number
  title: string
  relative_path?: string | null
  page_start?: number | null
  section_path: string[]
  text: string
}

export interface ChainRecommendation {
  chain_id: string
  title: string
  category: string
  status: string
  reason: string
  steps: ChainStepInfo[]
  evidence: ChainEvidence[]
}

export interface BinarySummary {
  bits: number
  is_pie: boolean
  has_canary: boolean
  has_nx: boolean
  relro: string
  imports: number
  win_candidates: Array<{ address: number; symbol?: string | null; source: string }>
  stack_hints: number[]
  call_graph?: {
    functions: number
    edges: number
    dead: number
    dead_output: number
    main?: string | null
  }
}

export interface ChainResponse {
  entries: SourceEntryInfo[]
  chains: ChainRecommendation[]
  binary?: BinarySummary | null
  warnings: string[]
}

export type ViewId = 'search' | 'chain' | 'sources' | 'settings'
