import { CircleAlert, LoaderCircle, ShieldCheck } from 'lucide-react'
import { type FormEvent, useRef, useState } from 'react'
import { analyzeBinary, recommendChain } from '../api'
import type { ChainResponse } from '../types'

const CHAIN_STATUS_LABEL: Record<string, string> = {
  static: '静态确认',
  'needs-binary': '需要二进制',
  'needs-execution': '需执行验证',
  'needs-leak-chain': '需泄漏链',
  'needs-oracle': '需 Oracle',
  refused: '已拒绝',
  blocked: '受阻',
  unsupported: '不支持',
}

const STEP_STATUS_LABEL: Record<string, string> = {
  ok: '完成',
  blocked: '阻断',
  'requires-execution': '需执行',
  unknown: '未实现',
}

function evidenceLocation(item: ChainResponse['chains'][number]['evidence'][number]): string {
  const parts: string[] = []
  if (item.relative_path) parts.push(item.relative_path)
  if (item.page_start) parts.push(`第 ${item.page_start} 页`)
  if (item.section_path.length > 0) parts.push(item.section_path.join(' / '))
  return parts.join(' · ') || '来源未标注'
}

export function ChainView() {
  const [source, setSource] = useState('')
  const [filename, setFilename] = useState('')
  const [relativePath, setRelativePath] = useState('')
  const [binaryFile, setBinaryFile] = useState<File | null>(null)
  const [result, setResult] = useState<ChainResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!source.trim() && !relativePath.trim() && binaryFile === null) {
      setError('粘贴源码、上传 ELF，或填写语料中的二进制相对路径。')
      return
    }
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setLoading(true)
    setError(null)
    try {
      const response = binaryFile !== null
        ? await analyzeBinary(binaryFile, controller.signal)
        : await recommendChain(
          {
            source: source.trim() || undefined,
            filename: filename.trim() || undefined,
            relative_path: relativePath.trim() || undefined,
          },
          controller.signal,
        )
      setResult(response)
    } catch (reason) {
      if ((reason as Error).name !== 'AbortError') {
        setError(reason instanceof Error ? reason.message : '验证链分析失败。')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="section-page chain-page">
      <header className="section-heading">
        <div>
          <span className="page-kicker">CHAIN RECOMMENDATION</span>
          <h1>验证链</h1>
          <p>
            粘贴源码或指定语料内的二进制，识别漏洞入口，给出候选验证链与知识证据。
            产品内只做静态分析，不执行目标程序。
          </p>
        </div>
        <span className="governance-badge"><ShieldCheck size={14} /> 只读分析</span>
      </header>

      <form className="chain-form" onSubmit={submit}>
        <label className="chain-source">
          <span>粘贴源码（C / Python，可选）</span>
          <textarea
            rows={7}
            value={source}
            placeholder={'例如：\nint main(void) {\n  char buf[64];\n  gets(buf);\n}'}
            onChange={(event) => setSource(event.target.value)}
          />
        </label>
        <div className="chain-inline">
          <label>
            <span>文件名</span>
            <input
              value={filename}
              placeholder="challenge.c"
              onChange={(event) => setFilename(event.target.value)}
            />
          </label>
          <label>
            <span>语料内二进制路径</span>
            <input
              value={relativePath}
              placeholder="uploads/target"
              onChange={(event) => setRelativePath(event.target.value)}
            />
          </label>
          <label className="chain-file">
            <span>或上传 ELF</span>
            <input
              type="file"
              accept=".elf,.so,.bin,application/octet-stream"
              onChange={(event) => setBinaryFile(event.target.files?.[0] ?? null)}
            />
            <small>{binaryFile ? binaryFile.name : '仅做只读分析，不执行'}</small>
          </label>
          <button className="chain-submit" type="submit" disabled={loading}>
            {loading ? <LoaderCircle size={15} className="spin" /> : <ShieldCheck size={15} />}
            {loading ? '分析中' : '分析'}
          </button>
        </div>
      </form>

      {error && (
        <div className="chain-error">
          <CircleAlert size={15} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="chain-results">
          {result.binary && (
            <div className="chain-binary">
              <div className="chain-binary-head">
                <strong>二进制画像</strong>
                <span>{result.binary.imports} 个导入</span>
              </div>
              <div className="chain-badges">
                <span>{result.binary.bits} 位</span>
                <span className={result.binary.is_pie ? 'on' : ''}>
                  PIE {result.binary.is_pie ? '开' : '关'}
                </span>
                <span className={result.binary.has_canary ? 'on' : ''}>
                  canary {result.binary.has_canary ? '开' : '关'}
                </span>
                <span className={result.binary.has_nx ? 'on' : ''}>
                  NX {result.binary.has_nx ? '开' : '关'}
                </span>
                <span>RELRO {result.binary.relro}</span>
              </div>
              {result.binary.win_candidates.length > 0 && (
                <ul className="chain-wins">
                  {result.binary.win_candidates.map((candidate) => (
                    <li key={`${candidate.address}-${candidate.source}`}>
                      <code>{candidate.symbol ?? '未命名函数'}</code>
                      <span>{`0x${candidate.address.toString(16)}`}</span>
                      <em>{candidate.source}</em>
                    </li>
                  ))}
                </ul>
              )}
              {result.binary.stack_hints.length > 0 && (
                <p className="chain-hints">
                  栈缓冲候选：{result.binary.stack_hints.join(' / ')} 字节
                </p>
              )}
              {result.binary.call_graph && (
                <p className="chain-hints">
                  调用图：{result.binary.call_graph.functions} 个函数 /{' '}
                  {result.binary.call_graph.edges} 条边；主流程不可达{' '}
                  {result.binary.call_graph.dead} 个，其中会输出{' '}
                  {result.binary.call_graph.dead_output} 个
                </p>
              )}
            </div>
          )}

          <div className="chain-entries">
            {result.entries.length === 0 ? (
              <p className="chain-empty">未识别到入口。</p>
            ) : (
              result.entries.map((entry) => (
                <span className="chain-entry" key={`${entry.category}-${entry.line ?? 'binary'}`}>
                  <b>{entry.technique}</b>
                  <small>{entry.line ? `第 ${entry.line} 行` : entry.text}</small>
                </span>
              ))
            )}
          </div>

          {result.chains.map((chain) => (
            <article className="chain-card" key={chain.chain_id}>
              <header>
                <div>
                  <h2>{chain.title}</h2>
                  <p>{chain.reason}</p>
                </div>
                <span className={`chain-status ${chain.status}`}>
                  {CHAIN_STATUS_LABEL[chain.status] ?? chain.status}
                </span>
              </header>
              <ol className="chain-steps">
                {chain.steps.map((step) => (
                  <li key={step.unit} className={step.status}>
                    <span className="chain-step-dot" />
                    <div>
                      <b>{step.title}</b>
                      <small>{step.detail}</small>
                    </div>
                    <em>{STEP_STATUS_LABEL[step.status] ?? step.status}</em>
                  </li>
                ))}
              </ol>
              {chain.evidence.length > 0 && (
                <div className="chain-evidence">
                  <h3>知识证据</h3>
                  {chain.evidence.map((item) => (
                    <div className="chain-evidence-item" key={item.chunk_id}>
                      <div>
                        <strong>{item.title}</strong>
                        <span>{evidenceLocation(item)}</span>
                      </div>
                      <p>{item.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </article>
          ))}

          {result.warnings.length > 0 && (
            <ul className="chain-warnings">
              {result.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}

          <p className="chain-note">
            “需执行验证”的步骤没有在产品内运行：产品只做静态分析，避免执行未知二进制；
            请在靶场环境复核。
          </p>
        </div>
      )}
    </section>
  )
}
