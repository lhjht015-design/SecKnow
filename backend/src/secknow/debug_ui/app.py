from __future__ import annotations

"""基于 tkinter 的本地调试界面。"""

import json
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .service_adapter import (
    UiSettings,
    ingest_file,
    inspect_file_metadata,
    inspect_qdrant,
    semantic_search,
)


class DebugUiApp:
    """SecKnow 本地调试工具窗口。"""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("SecKnow Debug UI")
        self.root.geometry("1280x920")
        self.root.report_callback_exception = self.report_callback_exception

        self.mode_var = tk.StringVar(value="online")
        self.qdrant_host_var = tk.StringVar(value="localhost")
        self.qdrant_port_var = tk.StringVar(value="6333")
        self.sparse_db_path_var = tk.StringVar(value="db/online_sparse.sqlite3")
        self.db_path_var = tk.StringVar(value="db/secknow.sqlite3")
        self.index_dir_var = tk.StringVar(value="db/faiss")
        self.embedding_mode_var = tk.StringVar(value="sbert")
        self.embedding_model_var = tk.StringVar(
            value="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_dim_var = tk.StringVar(value="384")
        self.status_var = tk.StringVar(value="准备就绪")

        self.ingest_file_var = tk.StringVar()
        self.ingest_zone_var = tk.StringVar(value="cyber")
        self.ingest_record_type_var = tk.StringVar(value="knowledge")
        self.ingest_chunk_strategy_var = tk.StringVar(value="hybrid")
        self.ingest_dedup_strategy_var = tk.StringVar(value="exact")
        self.ingest_max_tokens_var = tk.StringVar(value="300")
        self.ingest_overlap_var = tk.StringVar(value="50")
        self.ingest_language_var = tk.StringVar(value="")
        self.ingest_write_var = tk.BooleanVar(value=True)

        self.query_text_var = tk.StringVar()
        self.query_zone_var = tk.StringVar(value="cyber")
        self.query_top_k_var = tk.StringVar(value="5")
        self.query_record_type_var = tk.StringVar(value="knowledge")
        self.schema_filter_var = tk.StringVar(value="")
        self.meta_doc_id_var = tk.StringVar(value="")
        self.meta_filename_var = tk.StringVar(value="")
        self._last_schema_result = None

        self._build_layout()
        self.log("桌面调试 UI 已初始化")

    def run(self) -> None:
        """进入主事件循环。"""
        self.root.mainloop()

    def report_callback_exception(
        self,
        exc: type[BaseException],
        val: BaseException,
        tb,
    ) -> None:
        """捕获 tkinter 主线程回调异常并输出。"""
        formatted = "".join(traceback.format_exception(exc, val, tb))
        self.log(f"Tk 回调异常：{val}")
        self.log(formatted)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        title = ttk.Label(
            container,
            text="SecKnow Debug UI",
            font=("Arial", 18, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            container,
            text="当前分支仅用于调试 4.1 / 4.3 / 4.4 联调。",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 10))

        status_label = ttk.Label(container, textvariable=self.status_var)
        status_label.grid(row=2, column=0, sticky="w", pady=(8, 4))

        notebook = ttk.Notebook(container)
        notebook.grid(row=3, column=0, sticky="nsew")
        container.rowconfigure(3, weight=1)

        self.config_tab = ttk.Frame(notebook, padding=10)
        self.ingest_tab = ttk.Frame(notebook, padding=10)
        self.query_tab = ttk.Frame(notebook, padding=10)
        self.schema_tab = ttk.Frame(notebook, padding=10)
        self.file_meta_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.config_tab, text="公共配置")
        notebook.add(self.ingest_tab, text="单文件入库")
        notebook.add(self.query_tab, text="语义检索")
        notebook.add(self.schema_tab, text="数据库结构")
        notebook.add(self.file_meta_tab, text="文件元信息")

        self._build_config_tab()
        self._build_ingest_tab()
        self._build_query_tab()
        self._build_schema_tab()
        self._build_file_meta_tab()

    def _build_config_tab(self) -> None:
        self.config_tab.columnconfigure(0, weight=1)
        self.config_tab.rowconfigure(1, weight=1)

        settings_frame = ttk.LabelFrame(self.config_tab, text="公共调试配置", padding=10)
        settings_frame.grid(row=0, column=0, sticky="nsew")
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        self._add_labeled_combobox(
            settings_frame,
            "存储模式",
            self.mode_var,
            ["online", "offline"],
            row=0,
            column=0,
            width=12,
        )
        self._add_labeled_entry(
            settings_frame,
            "Qdrant Host",
            self.qdrant_host_var,
            row=0,
            column=2,
        )
        self._add_labeled_entry(
            settings_frame,
            "Qdrant Port",
            self.qdrant_port_var,
            row=0,
            column=4,
            width=10,
        )
        self._add_labeled_entry(
            settings_frame,
            "在线稀疏索引 SQLite",
            self.sparse_db_path_var,
            row=1,
            column=0,
            colspan=5,
        )
        self._add_labeled_entry(
            settings_frame,
            "离线 SQLite",
            self.db_path_var,
            row=2,
            column=0,
            colspan=3,
        )
        self._add_labeled_entry(
            settings_frame,
            "离线 FAISS 目录",
            self.index_dir_var,
            row=2,
            column=3,
            colspan=2,
        )
        self._add_labeled_combobox(
            settings_frame,
            "编码模式",
            self.embedding_mode_var,
            ["sbert", "fake"],
            row=3,
            column=0,
            width=12,
        )
        self._add_labeled_entry(
            settings_frame,
            "向量维度",
            self.embedding_dim_var,
            row=3,
            column=2,
            width=10,
        )
        self._add_labeled_entry(
            settings_frame,
            "编码模型",
            self.embedding_model_var,
            row=3,
            column=3,
            colspan=2,
        )

        log_frame = ttk.LabelFrame(self.config_tab, text="调试日志", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state=tk.DISABLED)

    def _build_ingest_tab(self) -> None:
        self.ingest_tab.columnconfigure(0, weight=1)
        self.ingest_tab.rowconfigure(3, weight=1)

        file_row = ttk.Frame(self.ingest_tab)
        file_row.grid(row=0, column=0, sticky="ew")
        file_row.columnconfigure(0, weight=1)
        ttk.Entry(file_row, textvariable=self.ingest_file_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(file_row, text="选择文件", command=self.on_pick_file).grid(
            row=0, column=1
        )

        options_row = ttk.Frame(self.ingest_tab)
        options_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for idx in range(6):
            options_row.columnconfigure(idx, weight=1)
        self._add_labeled_combobox(
            options_row, "分区", self.ingest_zone_var, ["cyber", "ai", "crypto"], 0, 0
        )
        self._add_labeled_combobox(
            options_row,
            "记录类型",
            self.ingest_record_type_var,
            ["knowledge", "baseline"],
            0,
            1,
        )
        self._add_labeled_combobox(
            options_row,
            "分块策略",
            self.ingest_chunk_strategy_var,
            ["hybrid", "fixed_window", "paragraph", "line", "semantic"],
            0,
            2,
        )
        self._add_labeled_combobox(
            options_row,
            "去重策略",
            self.ingest_dedup_strategy_var,
            ["exact", "minhash"],
            0,
            3,
        )
        self._add_labeled_entry(options_row, "max_tokens", self.ingest_max_tokens_var, 0, 4)
        self._add_labeled_entry(options_row, "overlap", self.ingest_overlap_var, 0, 5)

        action_row = ttk.Frame(self.ingest_tab)
        action_row.grid(row=2, column=0, sticky="ew", pady=(10, 10))
        ttk.Label(action_row, text="language").pack(side=tk.LEFT)
        ttk.Entry(action_row, textvariable=self.ingest_language_var, width=16).pack(
            side=tk.LEFT, padx=(6, 16)
        )
        ttk.Button(action_row, text="预览切块", command=self.on_preview_chunks).pack(
            side=tk.RIGHT
        )
        ttk.Button(action_row, text="开始入库", command=self.on_ingest).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        log_frame = ttk.LabelFrame(self.ingest_tab, text="入库日志", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.ingest_text = tk.Text(log_frame, wrap="word")
        self.ingest_text.grid(row=0, column=0, sticky="nsew")

    def _build_query_tab(self) -> None:
        self.query_tab.columnconfigure(0, weight=1)
        self.query_tab.rowconfigure(2, weight=1)

        query_input_frame = ttk.Frame(self.query_tab)
        query_input_frame.grid(row=0, column=0, sticky="ew")
        query_input_frame.columnconfigure(0, weight=1)

        query_box = ttk.LabelFrame(query_input_frame, text="查询文本", padding=8)
        query_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        query_box.columnconfigure(0, weight=1)
        query_box.rowconfigure(0, weight=1)
        self.query_text = tk.Text(query_box, height=4, wrap="word")
        self.query_text.grid(row=0, column=0, sticky="nsew")

        side_box = ttk.LabelFrame(query_input_frame, text="查询参数", padding=8)
        side_box.grid(row=0, column=1, sticky="ns")
        self._add_labeled_combobox(
            side_box, "分区", self.query_zone_var, ["cyber", "ai", "crypto"], 0, 0
        )
        self._add_labeled_entry(side_box, "top_k", self.query_top_k_var, 1, 0)
        self._add_labeled_combobox(
            side_box,
            "记录类型",
            self.query_record_type_var,
            ["knowledge", "baseline"],
            2,
            0,
        )
        ttk.Button(side_box, text="开始查询", command=self.on_query).grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )

        vector_frame = ttk.LabelFrame(self.query_tab, text="Query 向量调试信息", padding=8)
        vector_frame.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        vector_frame.columnconfigure(0, weight=1)
        vector_frame.rowconfigure(0, weight=1)
        self.query_vector_text = tk.Text(vector_frame, height=8, wrap="word")
        self.query_vector_text.grid(row=0, column=0, sticky="nsew")

        result_frame = ttk.LabelFrame(self.query_tab, text="检索结果", padding=8)
        result_frame.grid(row=2, column=0, sticky="nsew")
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self.query_result_text = tk.Text(result_frame, wrap="word")
        self.query_result_text.grid(row=0, column=0, sticky="nsew")

    def _build_schema_tab(self) -> None:
        self.schema_tab.columnconfigure(0, weight=1)
        self.schema_tab.rowconfigure(1, weight=1)

        top_row = ttk.Frame(self.schema_tab)
        top_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(top_row, text="文件名过滤").pack(side=tk.LEFT)
        ttk.Entry(top_row, textvariable=self.schema_filter_var, width=28).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Label(
            top_row,
            text="读取当前 Qdrant 的 collection 结构、向量配置和 payload 示例。",
        ).pack(side=tk.LEFT)
        ttk.Button(top_row, text="刷新结构", command=self.on_refresh_schema).pack(
            side=tk.RIGHT
        )

        schema_frame = ttk.LabelFrame(self.schema_tab, text="Qdrant 结构信息", padding=8)
        schema_frame.grid(row=1, column=0, sticky="nsew")
        schema_frame.columnconfigure(0, weight=1)
        schema_frame.rowconfigure(0, weight=1)
        self.schema_text = tk.Text(schema_frame, wrap="word")
        self.schema_text.grid(row=0, column=0, sticky="nsew")

    def _build_file_meta_tab(self) -> None:
        self.file_meta_tab.columnconfigure(0, weight=1)
        self.file_meta_tab.rowconfigure(1, weight=1)

        top_row = ttk.Frame(self.file_meta_tab)
        top_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(top_row, text="doc_id").pack(side=tk.LEFT)
        ttk.Entry(top_row, textvariable=self.meta_doc_id_var, width=36).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Label(top_row, text="文件名").pack(side=tk.LEFT)
        ttk.Entry(top_row, textvariable=self.meta_filename_var, width=28).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Button(top_row, text="查询文件详情", command=self.on_query_file_metadata).pack(
            side=tk.RIGHT
        )

        meta_frame = ttk.LabelFrame(self.file_meta_tab, text="文件元信息与 chunk 明细", padding=8)
        meta_frame.grid(row=1, column=0, sticky="nsew")
        meta_frame.columnconfigure(0, weight=1)
        meta_frame.rowconfigure(0, weight=1)
        self.file_meta_text = tk.Text(meta_frame, wrap="word")
        self.file_meta_text.grid(row=0, column=0, sticky="nsew")

    def _add_labeled_entry(
        self,
        parent: tk.Misc,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
        *,
        width: int = 24,
        colspan: int = 1,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, columnspan=colspan, sticky="ew", padx=4, pady=4)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=variable, width=width).grid(
            row=1, column=0, sticky="ew"
        )

    def _add_labeled_combobox(
        self,
        parent: tk.Misc,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        row: int,
        column: int,
        *,
        width: int = 18,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky="ew", padx=4, pady=4)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        box = ttk.Combobox(
            frame,
            textvariable=variable,
            values=values,
            state="readonly",
            width=width,
        )
        box.grid(row=1, column=0, sticky="ew")

    def current_settings(self) -> UiSettings:
        return UiSettings(
            mode=self.mode_var.get().strip() or "online",
            qdrant_host=self.qdrant_host_var.get().strip() or "localhost",
            qdrant_port=int((self.qdrant_port_var.get() or "6333").strip()),
            sparse_db_path=self.sparse_db_path_var.get().strip()
            or "db/online_sparse.sqlite3",
            db_path=self.db_path_var.get().strip() or "db/secknow.sqlite3",
            index_dir=self.index_dir_var.get().strip() or "db/faiss",
            embedding_mode=self.embedding_mode_var.get().strip() or "sbert",
            embedding_model=self.embedding_model_var.get().strip()
            or "sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=int((self.embedding_dim_var.get() or "384").strip()),
        )

    def log(self, message: str) -> None:
        print(f"[debug_ui] {message}")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.status_var.set(message)
        self.root.update_idletasks()

    def set_text(self, widget: tk.Text, value: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        self.root.update_idletasks()

    def on_pick_file(self) -> None:
        self.log("点击了“选择文件”按钮")
        path = filedialog.askopenfilename(title="选择待入库文件")
        if path:
            self.ingest_file_var.set(path)
            self.log(f"已选择文件：{path}")

    def on_ingest(self) -> None:
        self._execute_ingest(write_to_store=True)

    def on_preview_chunks(self) -> None:
        self.log("点击了“预览切块”按钮")
        self._execute_ingest(write_to_store=False)

    def _execute_ingest(self, *, write_to_store: bool) -> None:
        self.log("点击了“开始入库”按钮")
        file_path = self.ingest_file_var.get().strip()
        if not file_path:
            messagebox.showwarning("提示", "请先选择或填写待入库文件路径。")
            return
        if not Path(file_path).exists():
            messagebox.showerror("错误", f"文件不存在：{file_path}")
            return

        settings = self.current_settings()
        zone_id = self.ingest_zone_var.get()
        record_type = self.ingest_record_type_var.get()
        chunk_strategy = self.ingest_chunk_strategy_var.get()
        embedding_mode = settings.embedding_mode
        dedup_strategy = self.ingest_dedup_strategy_var.get()
        max_tokens = int(self.ingest_max_tokens_var.get().strip())
        overlap = int(self.ingest_overlap_var.get().strip())
        language = self.ingest_language_var.get().strip() or None
        if chunk_strategy == "semantic" and embedding_mode == "sbert":
            self.log("已拦截 semantic + sbert 组合，避免调试工具进程崩溃")
            messagebox.showwarning(
                "调试保护",
                "当前调试工具已禁用 semantic + sbert 组合。\n\n"
                "原因：该路径会触发 LangChain/HuggingFace 真实 embedding 初始化，"
                "本机上已出现直接导致进程崩溃的情况。\n\n"
                "建议改用：\n"
                "1. 分块策略切回 hybrid\n"
                "2. 或把编码模式改成 fake 做联调",
            )
            return
        try:
            if write_to_store:
                self.log("开始执行单文件入库")
            else:
                self.log("开始执行切块预览")
            debug_result = ingest_file(
                settings=settings,
                file_path=file_path,
                zone_id=zone_id,
                record_type=record_type,
                chunk_strategy=chunk_strategy,
                dedup_strategy=dedup_strategy,
                max_tokens=max_tokens,
                overlap=overlap,
                language=language,
                write_to_store=write_to_store,
            )
            self.render_ingest_result(debug_result)
            if write_to_store:
                self.log("单文件入库完成")
            else:
                self.log("切块预览完成")
        except Exception as exc:  # noqa: BLE001
            self.log(f"入库失败：{exc}")
            self.set_text(self.ingest_text, f"入库失败：{exc}")

    def render_ingest_result(self, debug_result) -> None:
        result = debug_result.pipeline_result
        preview_limit = 700
        lines = [
            "=== 4.1 流水线摘要 ===",
            f"source_path={result.loaded_document.source_path}",
            f"filename={result.loaded_document.filename}",
            f"file_type={result.loaded_document.file_type}",
            f"doc_id={result.loaded_document.doc_id}",
            f"raw_text_len={len(result.raw_document.text)}",
            f"chunks_before_dedup={len(result.chunks)}",
            f"chunks_after_dedup={len(result.deduped_chunks)}",
            f"dropped_duplicates={result.dropped_duplicates}",
            f"records_ready={len(result.records)}",
            "",
            "=== chunk 明细 ===",
        ]
        for idx, chunk in enumerate(result.deduped_chunks, start=1):
            record = result.records[idx - 1] if idx - 1 < len(result.records) else None
            metadata = record.metadata if record is not None else None
            preview = chunk.text.strip()
            overflow = len(preview) > preview_limit
            if overflow:
                preview = f"{preview[:preview_limit]}...[已截断，原 chunk 超过 700 字]"
            lines.extend(
                [
                    f"[chunk {idx}] chunk_index={chunk.chunk_index} char_len={chunk.char_len}",
                    (
                        "warning=当前 chunk 文本较长，预览已截断；这通常说明该 chunk 体积偏大，"
                        "建议关注 max_tokens / 分块策略"
                        if overflow
                        else "warning=无"
                    ),
                    "text=",
                    preview,
                    f"vector_dim={len(record.vector) if record is not None else 0}",
                    (
                        "metadata="
                        f"doc_id={metadata.doc_id} "
                        f"zone_id={metadata.zone_id} "
                        f"record_type={metadata.record_type} "
                        f"file_type={metadata.file_type} "
                        f"chunk_id={metadata.chunk_id}"
                    )
                    if metadata is not None
                    else "metadata=无",
                    "",
                ]
            )
        if debug_result.upsert_result is None:
            lines.extend(
                [
                    "=== 4.3 写库摘要 ===",
                    "本次未写库（可能是未勾选“执行写库”或没有 records）。",
                ]
            )
        else:
            upsert = debug_result.upsert_result
            lines.extend(
                [
                    "=== 4.3 写库摘要 ===",
                    f"zone_id={upsert.zone_id}",
                    f"attempted={upsert.attempted}",
                    f"inserted={upsert.inserted}",
                    f"chunk_id_count={len(upsert.chunk_ids)}",
                ]
            )
        self.set_text(self.ingest_text, "\n".join(lines))

    def on_query(self) -> None:
        self.log("点击了“开始查询”按钮")
        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("提示", "请输入查询文本。")
            return

        settings = self.current_settings()
        zone_id = self.query_zone_var.get()
        top_k = int(self.query_top_k_var.get().strip())
        record_type = self.query_record_type_var.get()
        try:
            self.log("开始执行语义检索")
            debug_result = semantic_search(
                settings=settings,
                query=query,
                zone_id=zone_id,
                top_k=top_k,
                record_type=record_type,
            )
            self.render_query_result(debug_result)
            self.log("语义检索完成")
        except Exception as exc:  # noqa: BLE001
            self.log(f"查询失败：{exc}")
            self.set_text(self.query_result_text, f"查询失败：{exc}")

    def render_query_result(self, debug_result) -> None:
        vector_preview = ", ".join(
            f"{item:.6f}" for item in debug_result.query_vector[:8]
        )
        vector_lines = [
            "=== Query 向量 ===",
            f"query={debug_result.query}",
            f"vector_dim={len(debug_result.query_vector)}",
            f"vector_preview=[{vector_preview}]",
            f"raw_hit_count={len(debug_result.raw_hits)}",
            "retrieval_mode=qdrant_dense_only",
        ]
        self.set_text(self.query_vector_text, "\n".join(vector_lines))

        result_lines = []
        if not debug_result.formatted_results:
            result_lines.append("没有检索到结果。")
        for idx, result in enumerate(debug_result.formatted_results, start=1):
            result_lines.extend(
                [
                    f"[{idx}] score={result.score:.6f}",
                    f"doc_id={result.doc_id}",
                    f"filename={result.filename}",
                    f"source_path={result.source_path}",
                    f"record_type={result.record_type}",
                    f"text={result.text}",
                    "",
                ]
            )
        self.set_text(self.query_result_text, "\n".join(result_lines))

    def on_refresh_schema(self) -> None:
        self.log("点击了“刷新结构”按钮")
        settings = self.current_settings()
        try:
            self.log("开始读取 Qdrant 结构")
            result = inspect_qdrant(settings)
            self._last_schema_result = result
            self.render_schema_result(result)
            self.log("Qdrant 结构读取完成")
        except Exception as exc:  # noqa: BLE001
            self.log(f"读取结构失败：{exc}")
            self.set_text(self.schema_text, f"读取结构失败：{exc}")

    def render_schema_result(self, result) -> None:
        filename_filter = self.schema_filter_var.get().strip().lower()
        lines = [
            "=== Qdrant 结构概览 ===",
            f"host={result.host}",
            f"port={result.port}",
            f"collection_count={len(result.collections)}",
            "",
        ]
        if not result.collections:
            lines.append("当前没有 collection。")
        for item in result.collections:
            lines.extend(
                [
                    f"=== collection: {item.name} ===",
                    f"points_count={item.points_count}",
                    f"vectors_count={item.vectors_count}",
                    f"indexed_vectors_count={item.indexed_vectors_count}",
                    f"vector_size={item.vector_size}",
                    f"distance={item.distance}",
                    f"doc_count={item.doc_count}",
                    f"filename_count={item.filename_count}",
                    f"record_type_counts={json.dumps(item.record_type_counts, ensure_ascii=False)}",
                    f"payload_schema={json.dumps(item.payload_schema, ensure_ascii=False)}",
                    f"sample_payload_keys={item.sample_payload_keys}",
                    "sample_payload=这只是一条样本记录，不代表全部文件",
                    f"sample_payload={json.dumps(item.sample_payload, ensure_ascii=False)}",
                    "按文件名分组的 chunk 数=",
                    "",
                ]
            )
            visible = item.filename_chunk_counts
            if filename_filter:
                visible = [
                    pair for pair in visible if filename_filter in pair[0].lower()
                ]
                lines.append(f"filename_filter={filename_filter}")
            if not visible:
                lines.append("  - 当前过滤条件下无结果")
            for filename, chunk_count in visible:
                lines.append(f"  - {filename} | chunk_count={chunk_count}")
            lines.append("")
        self.set_text(self.schema_text, "\n".join(lines))

    def on_query_file_metadata(self) -> None:
        self.log("点击了“查询文件详情”按钮")
        doc_id = self.meta_doc_id_var.get().strip()
        filename = self.meta_filename_var.get().strip()
        if not doc_id and not filename:
            messagebox.showwarning("提示", "请至少填写 doc_id 或文件名。")
            return

        try:
            self.log("开始查询文件元信息")
            result = inspect_file_metadata(
                self.current_settings(),
                doc_id=doc_id or None,
                filename=filename or None,
            )
            lines = [
                "=== 文件查询条件 ===",
                f"query_doc_id={result.query_doc_id}",
                f"query_filename={result.query_filename}",
                f"chunk_total={result.chunk_total}",
                f"matched_doc_ids={json.dumps(result.matched_doc_ids, ensure_ascii=False)}",
                f"matched_filenames={json.dumps(result.matched_filenames, ensure_ascii=False)}",
                "",
            ]
            if not result.chunks:
                lines.append("没有查询到匹配文件。")
            for idx, chunk in enumerate(result.chunks, start=1):
                lines.extend(
                    [
                        f"[chunk {idx}] collection={chunk.collection_name}",
                        f"doc_id={chunk.doc_id}",
                        f"filename={chunk.filename}",
                        f"source_path={chunk.source_path}",
                        f"record_type={chunk.record_type}",
                        f"file_type={chunk.file_type} extension={chunk.extension} language={chunk.language}",
                        f"chunk_index={chunk.chunk_index} / chunk_count={chunk.chunk_count}",
                        f"char_len={chunk.char_len} size_bytes={chunk.size_bytes} mtime={chunk.mtime}",
                        f"chunk_id={chunk.chunk_id}",
                        f"content_hash={chunk.content_hash}",
                        f"vector_dim={chunk.vector_dim}",
                        f"vector_preview={json.dumps(chunk.vector_preview, ensure_ascii=False)}",
                        f"text={chunk.text}",
                        "",
                    ]
                )
            self.set_text(self.file_meta_text, "\n".join(lines))
            self.log("文件元信息查询完成")
        except Exception as exc:  # noqa: BLE001
            self.log(f"文件元信息查询失败：{exc}")
            self.set_text(self.file_meta_text, f"文件元信息查询失败：{exc}")


def main() -> None:
    """启动调试 UI。"""
    app = DebugUiApp()
    app.run()


if __name__ == "__main__":
    main()
