import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/noto-sans-sc";
import {
  Aperture,
  ArrowDownToLine,
  ArrowRight,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Copy,
  Cpu,
  Film,
  ImagePlus,
  Images,
  LoaderCircle,
  Maximize2,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import "./style.css";

type Mode = "text-to-image" | "text-to-video" | "image-to-video";
type Params = {
  mode: Mode;
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  guidance: number;
  frames: number;
  fps: number;
  input_asset_id: string | null;
};
type Job = {
  id: string;
  model: string;
  params: Params;
  status: string;
  message: string;
  created: number;
  updated: number;
  error: string | null;
  outputs: string[];
  cancel_requested: boolean;
};
type System = {
  models: { id: string; name: string; ready: boolean }[];
  workers: {
    id: string;
    online: boolean;
    message: string;
    ready_models: string[];
    missing?: Record<string, string[]>;
  }[];
  gpus: {
    index: number;
    name: string;
    total_mib: number;
    free_mib: number;
    utilization: number;
  }[];
};
const modes: { id: Mode; label: string; hint: string }[] = [
  { id: "text-to-image", label: "图片", hint: "把想象变成画面" },
  { id: "text-to-video", label: "视频", hint: "让故事开始流动" },
  { id: "image-to-video", label: "图片转视频", hint: "让静止的画面动起来" },
];
const labels: Record<string, string> = {
  queued: "排队中",
  dispatching: "提交中",
  running: "生成中",
  succeeded: "已完成",
  failed: "未完成",
  cancelled: "已取消",
};
const complete = (j: Job) =>
  ["succeeded", "failed", "cancelled"].includes(j.status);
const url = (id: string) => "/api/assets/" + id;
const templates = [
  {
    name: "雨夜街景",
    prompt:
      "A cinematic photograph of a quiet street in Kyoto on a rainy evening, warm lights reflected on wet pavement, a single red umbrella, subtle film grain, natural colors, 35mm lens",
  },
  {
    name: "产品静物",
    prompt:
      "A minimal product photograph of a ceramic perfume bottle on a travertine pedestal, olive branches, soft afternoon sunlight, warm neutral background, precise natural shadows, editorial photography",
  },
  {
    name: "山间旅程",
    prompt:
      "A cinematic wide shot of a lone traveler walking slowly through a misty mountain valley at sunrise, grass moving gently in the wind, slow camera push forward, natural lighting, calm atmosphere",
  },
];
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch("/api" + path, init);
  const data = await r.json();
  if (!r.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : Array.isArray(data.detail)
          ? data.detail.map((x: { msg: string }) => x.msg).join("；")
          : "请求失败，请稍后重试",
    );
  return data;
}
function initial(): Params {
  try {
    const p = JSON.parse(localStorage.getItem("studio-draft") || "null");
    if (p?.mode && modes.some((m) => m.id === p.mode))
      return { ...p, input_asset_id: null };
  } catch {}
  return {
    mode: "text-to-image",
    prompt: "",
    negative_prompt: "",
    width: 1024,
    height: 1024,
    steps: 25,
    seed: -1,
    guidance: 6,
    frames: 49,
    fps: 16,
    input_asset_id: null,
  };
}

function App() {
  const [params, setParams] = useState<Params>(initial);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [system, setSystem] = useState<System | null>(null);
  const [page, setPage] = useState<"create" | "library" | "resources">(
    "create",
  );
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false),
    [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState(""),
    [connection, setConnection] = useState(true);
  const [preview, setPreview] = useState<Job | null>(null),
    [advanced, setAdvanced] = useState(false);
  const [filter, setFilter] = useState("all");
  const fileInput = useRef<HTMLInputElement>(null);
  const idempotency = useRef<string | null>(null);
  const active =
    selected === "new"
      ? undefined
      : jobs.find((j) => j.id === selected) || jobs[0];
  const ready = system?.models.find(
    (m) => m.id === (params.mode === "text-to-image" ? "sdxl" : "wan22"),
  )?.ready;
  const pending = jobs.filter((j) => !complete(j));
  const outputs = jobs.filter(
    (j) => j.status === "succeeded" && j.outputs.length,
  );
  const currentMode = modes.find((m) => m.id === params.mode)!;
  const update = (patch: Partial<Params>) => {
    setParams((p) => ({ ...p, ...patch }));
    idempotency.current = null;
  };
  async function refresh() {
    const [j, s] = await Promise.all([
      api<{ items: Job[] }>("/jobs"),
      api<System>("/system"),
    ]);
    setJobs(j.items);
    setSystem(s);
    setConnection(true);
  }
  useEffect(() => {
    let mounted = true;
    const poll = () =>
      refresh().catch(() => {
        if (mounted) setConnection(false);
      });
    void poll();
    const timer = setInterval(poll, 4000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    try { localStorage.setItem("studio-draft", JSON.stringify(params)); } catch {}
  }, [params]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 6000);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPreview(null);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  function changeMode(mode: Mode) {
    const image = mode === "text-to-image";
    update({
      mode,
      width: image ? 1024 : 1280,
      height: image ? 1024 : 704,
      steps: image ? 25 : 20,
      guidance: image ? 6 : 5,
    });
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    idempotency.current ||= crypto.randomUUID();
    try {
      const job = await api<Job>("/jobs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": idempotency.current,
        },
        body: JSON.stringify({
          ...params,
          input_asset_id:
            params.mode === "image-to-video" ? params.input_asset_id : null,
        }),
      });
      idempotency.current = null;
      setSelected(job.id);
      await refresh();
      setNotice("已加入生成队列");
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function upload(file?: File) {
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) {
      setNotice("图片不能超过 20MB");
      return;
    }
    setUploading(true);
    const body = new FormData();
    body.append("file", file);
    try {
      const asset = await api<{ id: string }>("/uploads", {
        method: "POST",
        body,
      });
      update({ input_asset_id: asset.id });
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setUploading(false);
    }
  }
  async function action(job: Job, kind: "cancel" | "retry") {
    try {
      const next = await api<Job>(`/jobs/${job.id}/${kind}`, {
        method: "POST",
        headers: { "Idempotency-Key": crypto.randomUUID() },
      });
      setSelected(next.id);
      await refresh();
    } catch (e) {
      setNotice((e as Error).message);
    }
  }
  function reuse(job: Job) {
    setParams(job.params);
    idempotency.current = null;
    setPage("create");
    setNotice("已载入这次生成的参数");
  }
  function media(job: Job, full = false) {
    return job.params.mode === "text-to-image" ? (
      <img src={url(job.outputs[0])} alt={job.params.prompt} loading="lazy" />
    ) : (
      <video
        src={url(job.outputs[0])}
        controls={full}
        muted={!full}
        playsInline
        preload="metadata"
      />
    );
  }
  const ratios =
    params.mode === "text-to-image"
      ? [
          ["1:1", 1024, 1024],
          ["16:9", 1344, 768],
          ["9:16", 768, 1344],
          ["4:3", 1152, 864],
        ]
      : [
          ["16:9", 1280, 704],
          ["9:16", 704, 1280],
          ["1:1", 768, 768],
        ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("create");
          }}
        >
          <span className="brand-mark">
            <Aperture size={25} />
          </span>
          <span>
            帧间<span className="brand-en">STUDIO</span>
          </span>
        </a>
        <div className="workspace-name">
          <span className="avatar">Z</span>
          <div>
            我的创作空间<small>本地工作台</small>
          </div>
          <ChevronDown size={14} />
        </div>
        <p className="nav-caption">工作空间</p>
        <nav aria-label="主导航">
          <button
            className={page === "create" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("create")}
          >
            <WandSparkles size={19} />
            创作工作台
            <span className="nav-dot" />
          </button>
          <button
            className={page === "library" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("library")}
          >
            <Images size={19} />
            作品库<span className="nav-count">{outputs.length}</span>
          </button>
          <button
            className={page === "resources" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("resources")}
          >
            <Cpu size={19} />
            生成资源
          </button>
        </nav>
        <div className="sidebar-note">
          <div className="tiny-orbit">
            <Sparkles size={18} />
          </div>
          <strong>给想象，一个画面。</strong>
          <p>从第一张图片，到下一段故事。</p>
        </div>
        <div className="sidebar-bottom">
          <span className={"status-dot " + (connection ? "on" : "")} />
          {connection ? "工作台已连接" : "连接已断开"}
          <span>v0.1</span>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <span className="breadcrumb">我的空间</span>
            <span className="slash">/</span>
            <strong>
              {page === "create"
                ? "创作工作台"
                : page === "library"
                  ? "作品库"
                  : "生成资源"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="local-badge">
              <span className="status-dot on" /> LOCAL
            </span>
            <span className="profile">Z</span>
          </div>
        </header>
        {page === "create" && (
          <div className="page-body">
            <div className="page-heading">
              <div>
                <div className="eyebrow">CREATE SOMETHING NEW</div>
                <h1>
                  把灵感，变成作品<span>。</span>
                </h1>
                <p>描述你的想法，剩下的交给画面。</p>
              </div>
              <button
                className="subtle-button"
                onClick={() => {
                  update({
                    prompt: "",
                    negative_prompt: "",
                    input_asset_id: null,
                  });
                  setSelected("new");
                }}
              >
                <Plus size={17} />
                新的创作
              </button>
            </div>
            <div className="studio-layout">
              <form className="composer panel" onSubmit={submit}>
                <div className="mode-tabs" role="tablist" aria-label="生成模式">
                  {modes.map((m) => (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={params.mode === m.id}
                      className={params.mode === m.id ? "selected" : ""}
                      key={m.id}
                      onClick={() => changeMode(m.id)}
                    >
                      {m.id === "text-to-image" ? (
                        <ImagePlus size={16} />
                      ) : m.id === "text-to-video" ? (
                        <Film size={16} />
                      ) : null}
                      {m.label}
                    </button>
                  ))}
                </div>
                <label className="field-label" htmlFor="prompt">
                  画面描述<span>{params.prompt.length}/6000</span>
                </label>
                <textarea
                  id="prompt"
                  value={params.prompt}
                  onChange={(e) => update({ prompt: e.target.value })}
                  maxLength={6000}
                  placeholder={
                    params.mode === "text-to-image"
                      ? "例如：雨后的街角，一家亮着暖灯的咖啡馆，胶片质感，柔和的自然光……"
                      : "描述主体、动作与镜头运动，例如：镜头缓缓向前，一位旅人走进晨雾笼罩的山谷……"
                  }
                  required
                />
                <div className="prompt-tips">
                  <Sparkles size={13} />
                  <span>试试这些灵感</span>
                  {params.mode === "text-to-image" && <small>英文描述通常更稳定</small>}
                </div>
                <div className="prompt-chips">
                  {templates.map((t) => (
                    <button
                      type="button"
                      key={t.name}
                      onClick={() => update({ prompt: t.prompt })}
                    >
                      {t.name}
                      <ArrowRight size={11} />
                    </button>
                  ))}
                </div>
                {params.mode === "image-to-video" && (
                  <div
                    className="upload-box"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      void upload(e.dataTransfer.files[0]);
                    }}
                  >
                    <input
                      ref={fileInput}
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      hidden
                      onChange={(e) => void upload(e.target.files?.[0])}
                    />
                    {params.input_asset_id ? (
                      <>
                        <img src={url(params.input_asset_id)} alt="参考图片" />
                        <button
                          type="button"
                          className="remove-upload"
                          aria-label="移除参考图片"
                          onClick={() => update({ input_asset_id: null })}
                        >
                          <X size={15} />
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        onClick={() => fileInput.current?.click()}
                        disabled={uploading}
                      >
                        {uploading ? (
                          <LoaderCircle className="spin" size={24} />
                        ) : (
                          <Upload size={24} />
                        )}
                        <strong>
                          {uploading ? "正在上传" : "上传参考图片"}
                        </strong>
                        <small>拖放或点击选择 · PNG / JPG / WebP · 20MB</small>
                      </button>
                    )}
                  </div>
                )}
                <div className="divider" />
                <div className="field-label">生成模型</div>
                <div className="model-card">
                  <span className="model-icon">
                    <Aperture size={19} />
                  </span>
                  <div>
                    <strong>
                      {params.mode === "text-to-image"
                        ? "SDXL 1.0"
                        : "Wan 2.2 · 5B"}
                    </strong>
                    <small>
                      {params.mode === "text-to-image"
                        ? "图像生成"
                        : "视频生成"}
                    </small>
                  </div>
                  <span className={"model-status " + (ready ? "ready" : "")}>
                    {ready ? "已就绪" : "等待就绪"}
                  </span>
                </div>
                <div className="field-label">
                  画面比例
                  <span>
                    {params.width} × {params.height}
                  </span>
                </div>
                <div className="ratio-buttons">
                  {ratios.map(([label, w, h]) => (
                    <button
                      type="button"
                      key={label}
                      className={
                        params.width === w && params.height === h
                          ? "selected"
                          : ""
                      }
                      onClick={() =>
                        update({ width: Number(w), height: Number(h) })
                      }
                    >
                      <span
                        className="ratio-box"
                        style={{
                          width: Number(w) > Number(h) ? 23 : 15,
                          height: Number(w) < Number(h) ? 23 : 15,
                        }}
                      />
                      {label}
                    </button>
                  ))}
                </div>
                {params.mode !== "text-to-image" && (
                  <label className="video-duration">
                    视频长度
                    <select
                      aria-label="视频长度"
                      value={params.frames}
                      onChange={(e) =>
                        update({ frames: Number(e.target.value) })
                      }
                    >
                      <option value={33}>约 2 秒</option>
                      <option value={49}>约 3 秒</option>
                      <option value={81}>约 5 秒</option>
                    </select>
                  </label>
                )}
                <button
                  type="button"
                  className="advanced-toggle"
                  aria-expanded={advanced}
                  onClick={() => setAdvanced(!advanced)}
                >
                  <SlidersHorizontal size={15} />
                  高级设置
                  <ChevronDown size={15} className={advanced ? "rotate" : ""} />
                </button>
                {advanced && (
                  <div className="advanced-fields">
                    <label>
                      采样步数
                      <input
                        type="number"
                        min={1}
                        max={60}
                        value={params.steps}
                        onChange={(e) =>
                          update({ steps: Number(e.target.value) })
                        }
                      />
                    </label>
                    <label>
                      引导强度
                      <input
                        type="number"
                        min={1}
                        max={15}
                        step={0.5}
                        value={params.guidance}
                        onChange={(e) =>
                          update({ guidance: Number(e.target.value) })
                        }
                      />
                    </label>
                    <label className="wide">
                      随机种子 <small>−1 为随机</small>
                      <input
                        type="number"
                        min={-1}
                        max={9007199254740991}
                        value={params.seed}
                        onChange={(e) =>
                          update({ seed: Number(e.target.value) })
                        }
                      />
                    </label>
                    <label className="wide">
                      不希望出现的内容
                      <input
                        value={params.negative_prompt}
                        onChange={(e) =>
                          update({ negative_prompt: e.target.value })
                        }
                        placeholder="模糊、文字水印、过曝……"
                      />
                    </label>
                  </div>
                )}
                {!ready && (
                  <div className="inline-note">
                    <CircleHelp size={15} />
                    <span>可以先提交创意，生成服务就绪后将自动开始。</span>
                  </div>
                )}
                <button
                  className="generate-button"
                  disabled={
                    busy ||
                    uploading ||
                    !params.prompt.trim() ||
                    (params.mode === "image-to-video" && !params.input_asset_id)
                  }
                  type="submit"
                >
                  {busy ? (
                    <LoaderCircle size={18} className="spin" />
                  ) : (
                    <Sparkles size={18} />
                  )}{" "}
                  {busy ? "正在提交" : ready ? "开始生成" : "加入生成队列"}
                  <ArrowRight size={18} />
                </button>
                <p className="form-footnote">
                  作品与生成参数自动保存在你的工作空间
                </p>
              </form>
              <section className="canvas-column">
                <div className="canvas-top">
                  <span>
                    <span className="small-square" />
                    画面预览
                  </span>
                  <span>
                    {active ? labels[active.status] : "等待你的第一份灵感"}
                  </span>
                </div>
                <div
                  className={
                    "preview-canvas " +
                    (active?.outputs.length ? "has-media" : "")
                  }
                >
                  {active?.outputs.length ? (
                    <>
                      {media(active, true)}
                      <button
                        className="expand-preview"
                        aria-label="放大作品"
                        onClick={() => setPreview(active)}
                      >
                        <Maximize2 size={18} />
                      </button>
                    </>
                  ) : active && !complete(active) ? (
                    <div className="empty-state">
                      <div className="loading-orbit">
                        <Aperture size={40} />
                      </div>
                      <span className="eyebrow">
                        {active.status === "queued"
                          ? "IN THE QUEUE"
                          : "BRINGING IT TO LIFE"}
                      </span>
                      <h2>
                        {active.status === "queued"
                          ? "灵感已就位"
                          : "画面正在生成"}
                      </h2>
                      <p>{active.message}</p>
                      <div className="indeterminate" />
                      <button
                        className="subtle-button"
                        disabled={active.cancel_requested}
                        onClick={() => void action(active, "cancel")}
                      >
                        {active.cancel_requested ? "正在取消" : "取消任务"}
                      </button>
                    </div>
                  ) : active?.status === "failed" ? (
                    <div className="empty-state">
                      <CircleHelp size={36} />
                      <h2>这次生成没有完成</h2>
                      <p>{active.error || active.message}</p>
                      <button
                        className="subtle-button"
                        onClick={() => void action(active, "retry")}
                      >
                        <RefreshCw size={15} />
                        重新生成
                      </button>
                    </div>
                  ) : (
                    <div className="empty-state">
                      <div className="art-stack">
                        <div />
                        <div />
                        <div>
                          <Aperture size={34} />
                        </div>
                      </div>
                      <span className="eyebrow">
                        YOUR NEXT FRAME STARTS HERE
                      </span>
                      <h2>
                        {active?.status === "cancelled"
                          ? "随时开始下一次创作"
                          : currentMode.hint}
                      </h2>
                      <p>
                        输入一段描述，或从左侧灵感开始。
                        <br />
                        你的作品将在这里呈现。
                      </p>
                      <div className="empty-tags">
                        <span>描绘主体</span>
                        <i>·</i>
                        <span>设定光线</span>
                        <i>·</i>
                        <span>加入风格</span>
                      </div>
                    </div>
                  )}
                </div>
                {active?.outputs.length ? (
                  <div className="result-toolbar">
                    <span>
                      <Check size={15} />
                      生成完成{" "}
                      <small>
                        {active.params.width} × {active.params.height}
                      </small>
                    </span>
                    <div>
                      <button
                        className="subtle-button"
                        onClick={() => reuse(active)}
                      >
                        <Copy size={15} />
                        复用参数
                      </button>
                      <a
                        className="dark-button"
                        href={url(active.outputs[0]) + "?download=true"}
                      >
                        <ArrowDownToLine size={16} />
                        下载作品
                      </a>
                    </div>
                  </div>
                ) : (
                  <div className="canvas-caption">
                    <span>每一个画面，都有新的可能。</span>
                    <span>帧间 · STUDIO</span>
                  </div>
                )}
                <div className="queue-heading">
                  <h3>
                    最近任务 <span>{jobs.length}</span>
                  </h3>
                  <span>
                    {pending.length
                      ? `${pending.length} 个任务进行中`
                      : "创作记录会自动保留"}
                  </span>
                </div>
                <div className="recent-jobs">
                  {jobs.length ? (
                    jobs.slice(0, 5).map((j) => (
                      <button
                        key={j.id}
                        className={
                          "job-row " + (j.id === active?.id ? "chosen" : "")
                        }
                        onClick={() => setSelected(j.id)}
                      >
                        <span className="job-thumb">
                          {j.outputs.length ? (
                            media(j)
                          ) : j.params.mode === "text-to-image" ? (
                            <ImagePlus size={20} />
                          ) : (
                            <Film size={20} />
                          )}
                        </span>
                        <span className="job-description">
                          <strong>{j.params.prompt}</strong>
                          <small>
                            {j.params.width}×{j.params.height}
                            <i>·</i>
                            {new Date(j.created * 1000).toLocaleTimeString(
                              "zh-CN",
                              { hour: "2-digit", minute: "2-digit" },
                            )}
                          </small>
                        </span>
                        <span className={"job-status " + j.status}>
                          {!complete(j) && (
                            <LoaderCircle size={12} className="spin" />
                          )}
                          {labels[j.status]}
                        </span>
                      </button>
                    ))
                  ) : (
                    <div className="queue-empty">
                      <Clock3 size={17} />
                      还没有生成任务。开始创作后，可在这里查看进度。
                    </div>
                  )}
                </div>
              </section>
            </div>
          </div>
        )}
        {page === "library" && (
          <div className="page-body">
            <div className="page-heading">
              <div>
                <div className="eyebrow">YOUR CREATIVE COLLECTION</div>
                <h1>
                  每一帧，都值得留下<span>。</span>
                </h1>
                <p>查看作品、下载原文件，或沿用参数再次创作。</p>
              </div>
              <button className="dark-button" onClick={() => setPage("create")}>
                <Plus size={16} />
                开始创作
              </button>
            </div>
            <div className="library-filters">
              {[
                ["all", "全部"],
                ["image", "图片"],
                ["video", "视频"],
              ].map(([key, label]) => (
                <button
                  key={key}
                  className={filter === key ? "selected" : ""}
                  onClick={() => setFilter(key)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="gallery">
              {outputs
                .filter(
                  (j) =>
                    filter === "all" ||
                    (filter === "image") ===
                      (j.params.mode === "text-to-image"),
                )
                .map((j) => (
                  <article className="gallery-card" key={j.id}>
                    <button
                      className="gallery-media"
                      aria-label={"查看作品 " + j.params.prompt}
                      onClick={() => setPreview(j)}
                    >
                      {media(j)}
                    </button>
                    <div className="gallery-info">
                      <p>{j.params.prompt}</p>
                      <span>
                        {j.params.width} × {j.params.height}
                      </span>
                      <button
                        aria-label="复用生成参数"
                        onClick={() => reuse(j)}
                      >
                        <Copy size={16} />
                      </button>
                      <a
                        aria-label="下载作品"
                        href={url(j.outputs[0]) + "?download=true"}
                      >
                        <ArrowDownToLine size={16} />
                      </a>
                    </div>
                  </article>
                ))}
            </div>
            {!outputs.length && (
              <div className="library-empty">
                <Images size={40} />
                <h2>你的作品库，即将有故事。</h2>
                <p>完成一次生成，作品就会自动出现在这里。</p>
                <button
                  className="dark-button"
                  onClick={() => setPage("create")}
                >
                  开始第一份创作
                  <ArrowRight size={16} />
                </button>
              </div>
            )}
          </div>
        )}
        {page === "resources" && (
          <div className="page-body">
            <div className="page-heading">
              <div>
                <div className="eyebrow">LOCAL GENERATION ENGINE</div>
                <h1>
                  创作背后的算力<span>。</span>
                </h1>
                <p>模型与设备状态来自本机服务，自动更新。</p>
              </div>
              <button
                className="subtle-button"
                onClick={() =>
                  void refresh().catch(() => setNotice("连接失败"))
                }
              >
                <RefreshCw size={16} />
                刷新状态
              </button>
            </div>
            <div className="resource-models">
              {system?.models.map((m) => (
                <div className="panel resource-model" key={m.id}>
                  <Aperture size={25} />
                  <div>
                    <h3>{m.name}</h3>
                    <p>
                      {m.ready
                        ? "已连接，可以开始生成"
                        : "等待模型和生成服务就绪"}
                    </p>
                  </div>
                  <span className={"model-status " + (m.ready ? "ready" : "")}>
                    {m.ready ? "可用" : "等待就绪"}
                  </span>
                </div>
              ))}
            </div>
            <div className="gpu-grid">
              {system?.gpus.map((g) => (
                <div className="panel gpu-card" key={g.index}>
                  <div>
                    <Cpu size={20} />
                    <strong>GPU {g.index}</strong>
                    <span>{g.name.replace("NVIDIA GeForce ", "")}</span>
                  </div>
                  <p>
                    显存占用
                    <span>
                      {((g.total_mib - g.free_mib) / 1024).toFixed(1)} /{" "}
                      {(g.total_mib / 1024).toFixed(0)} GB
                    </span>
                  </p>
                  <div className="meter">
                    <span
                      style={{
                        width: `${(100 * (g.total_mib - g.free_mib)) / g.total_mib}%`,
                      }}
                    />
                  </div>
                  <p>
                    GPU 利用率<span>{g.utilization}%</span>
                  </p>
                </div>
              ))}
            </div>
            <div className="panel resource-help">
              <h3>生成服务</h3>
              {system?.workers.map((w) => (
                <div key={w.id}>
                  <span className={"status-dot " + (w.online ? "on" : "")} />
                  <strong>{w.id}</strong>
                  <span>{w.message}</span>
                  {w.missing && (
                    <small>
                      缺少：{Object.values(w.missing).flat().join("、")}
                    </small>
                  )}
                </div>
              ))}
              <p>
                尚未就绪时，任务会留在队列中。模型准备完成并启动推理服务后，任务会自动执行。
              </p>
            </div>
          </div>
        )}
        {!connection && (
          <div className="connection-banner" role="alert">
            工作台连接已断开，正在重连。已提交的任务会保留。
          </div>
        )}
      </main>
      {notice && (
        <div className="toast" role="status">
          <span>{notice}</span>
          <button aria-label="关闭提示" onClick={() => setNotice("")}>
            <X size={16} />
          </button>
        </div>
      )}
      {preview && (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label="作品预览"
          onClick={() => setPreview(null)}
        >
          <div className="preview-modal" onClick={(e) => e.stopPropagation()}>
            <button
              className="modal-close"
              aria-label="关闭预览"
              onClick={() => setPreview(null)}
            >
              <X size={22} />
            </button>
            {media(preview, true)}
            <div>
              <p>{preview.params.prompt}</p>
              <a
                className="dark-button"
                href={url(preview.outputs[0]) + "?download=true"}
              >
                <ArrowDownToLine size={16} />
                下载原文件
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
