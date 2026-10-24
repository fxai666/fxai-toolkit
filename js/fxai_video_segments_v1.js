// Copyright (c) 2026 凤希AI/www.fxai.site
// Licensed under MIT License
// 商用需购买商业授权

/**
 * 凤希AI - 视频分段器V1 · 面板
 * ============================
 *
 * 和「音频分段器V2」同一个交互：选文件 → 时间轴打点 → 参数决定怎么切。
 * 区别只有底图：音频画波形，视频画等间隔抽帧拼成的胶片带。
 *
 * 画布分四层：
 *   顶部   当前帧预览 + 当前时间 / 段信息
 *   胶片带 整条视频等间隔抽 40 帧，一眼能看出画面在哪
 *   分段条 按当前参数算出来的段边界，段号标在里面
 *   标记   手动打的关键帧，橙色竖线
 *
 * 分段算法在这里复刻了一份（和后端 fxai_video_utils.build_frame_segments 对齐），
 * 只为画分段条；真正切段仍以后端为准。
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiVideoSegmenterV1";
const DEFAULT_WIDTH = 900;
const DEFAULT_HEIGHT = 560;
const STRIP_TILES = 40;
const H3_FPS = 24;

const CANVAS_H = 244;
const PREVIEW_W = 168;
const PREVIEW_H = 95;
const STRIP_TOP = 100;
const STRIP_H = 90;
const SEG_TOP = 196;
const SEG_H = 44;

function getWidget(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name);
}

function hideWidget(widget) {
    if (!widget || widget.__fxaiHidden) return;
    widget.__fxaiOriginalType = widget.type;
    widget.__fxaiOriginalComputeSize = widget.computeSize;
    widget.__fxaiOriginalSerializeValue = widget.serializeValue;
    const element = widget.inputEl || widget.element || widget.el;
    const targets = [element, element?.parentElement, element?.parentElement?.parentElement].filter(Boolean);
    if (targets.length) {
        widget.__fxaiElements = targets.map((target) => ({ target, cssText: target.style.cssText }));
        for (const { target } of widget.__fxaiElements) {
            target.style.display = "none";
            target.style.visibility = "hidden";
            target.style.height = "0";
            target.style.minHeight = "0";
            target.style.maxHeight = "0";
            target.style.margin = "0";
            target.style.padding = "0";
            target.style.border = "0";
            target.style.overflow = "hidden";
            target.style.pointerEvents = "none";
        }
    }
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
    widget.__fxaiHidden = true;
}

function resizeNode(node) {
    const computed = node.computeSize?.();
    const width = Math.max(DEFAULT_WIDTH, computed?.[0] ?? 0, Array.isArray(node.size) ? node.size[0] : 0);
    const height = Math.max(DEFAULT_HEIGHT, computed?.[1] ?? 0, Array.isArray(node.size) ? node.size[1] : 0);
    node.size = [width, height];
    app.graph.setDirtyCanvas(true, true);
}

function parseKeyframes(value) {
    try {
        const parsed = JSON.parse(String(value || "[]"));
        if (!Array.isArray(parsed)) return [];
        return parsed
            .map((item) => Number.parseFloat(item))
            .filter((item) => Number.isFinite(item) && item >= 0)
            .sort((left, right) => left - right);
    } catch {
        return [];
    }
}

function normalizeKeyframes(keyframes, duration, fps) {
    const step = fps > 0 ? 1 / fps : 0.1;
    const seen = new Set();
    const out = [];
    for (const value of keyframes || []) {
        let seconds = Math.max(0, Number(value) || 0);
        if (duration > 0) seconds = Math.min(seconds, Math.max(0, duration - step / 2));
        const frame = Math.round(seconds / step);
        if (seen.has(frame)) continue;
        seen.add(frame);
        out.push(frame * step);
    }
    out.sort((left, right) => left - right);
    return out;
}

function formatTime(seconds) {
    const clamped = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(clamped / 60);
    const remainder = clamped - minutes * 60;
    return `${minutes}:${remainder.toFixed(3).padStart(6, "0")}`;
}

// ---------- 分段算法（与后端一致） ----------
function alignDownH3(frames) {
    if (frames < 5) return 5;
    return 17 * Math.floor((frames - 5) / 17) + 5;
}

function alignDownLtx(frames) {
    if (frames <= 9) return 9;
    return 8 * Math.floor((frames - 1) / 8) + 1;
}

function segmentFrames(avgDur, fps, model) {
    const name = String(model || "").trim().toLowerCase();
    if (name === "minimax") {
        const h3 = alignDownH3(Math.round(avgDur * H3_FPS));
        return Math.max(1, Math.round((h3 / H3_FPS) * fps));
    }
    if (name === "ltx") return Math.max(1, alignDownLtx(Math.round(avgDur * fps)));
    return Math.max(1, Math.round(avgDur * fps));
}

function buildFrameSegments(totalFrames, fps, keyframes, skipInitial, includeTail, avgSplit, avgDur, model) {
    if (!(totalFrames > 0) || !(fps > 0)) return [];

    const markers = [];
    const seen = new Set();
    for (const sec of keyframes) {
        const frame = Math.round(Math.max(0, sec) * fps);
        if (frame > 0 && frame < totalFrames && !seen.has(frame)) {
            seen.add(frame);
            markers.push(frame);
        }
    }
    markers.sort((left, right) => left - right);

    const points = [0, ...markers, totalFrames];
    let segments = [];
    for (let i = 0; i < points.length - 1; i += 1) {
        segments.push({ start: points[i], count: points[i + 1] - points[i] });
    }
    if (skipInitial && segments.length > 1) segments = segments.slice(1);
    if (!includeTail && segments.length > 1) segments = segments.slice(0, -1);

    const minFrames = Math.max(1, Math.round(0.1 * fps));
    segments = segments.filter((seg) => seg.count >= minFrames);
    if (!segments.length) segments = [{ start: 0, count: totalFrames }];

    if (!(avgSplit && avgDur > 0)) return segments;

    const spanStart = segments[0].start;
    const last = segments[segments.length - 1];
    const spanEnd = last.start + last.count;
    const step = segmentFrames(avgDur, fps, model);

    const out = [];
    let cursor = spanStart;
    while (cursor < spanEnd) {
        const take = Math.min(step, spanEnd - cursor);
        out.push({ start: cursor, count: take });
        cursor += take;
    }

    const threshold = (avgDur > 10 ? 5 : 3) * fps;
    if (out.length >= 2 && out[out.length - 1].count <= threshold) {
        const tail = out.pop();
        out[out.length - 1].count += tail.count;
    }
    return out;
}

// ---------- 上传 ----------
async function uploadFile(file, onProgress) {
    const body = new FormData();
    body.append("image", file);
    return new Promise((resolve, reject) => {
        const req = new XMLHttpRequest();
        req.upload.onprogress = (event) => onProgress?.(event.loaded / event.total);
        req.onload = () => {
            if (req.status !== 200) {
                reject(new Error(`上传失败: ${req.status}`));
                return;
            }
            try {
                resolve(JSON.parse(req.responseText).name);
            } catch {
                reject(new Error("解析响应失败"));
            }
        };
        req.onerror = () => reject(new Error("网络错误"));
        req.open("POST", api.apiURL("/upload/image"), true);
        req.send(body);
    });
}

// ---------- 面板 ----------
function buildEditor(node) {
    if (node.__fxaiVideoEditor) return node.__fxaiVideoEditor;

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "6px";
    container.style.width = "100%";

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.flexWrap = "wrap";
    toolbar.style.gap = "6px";
    toolbar.style.alignItems = "center";

    const pickButton = document.createElement("button");
    pickButton.textContent = "📁 选择视频";
    const uploadButton = document.createElement("button");
    uploadButton.textContent = "上传视频";
    const playButton = document.createElement("button");
    playButton.textContent = "播放";
    const addButton = document.createElement("button");
    addButton.textContent = "添加标记";
    const removeButton = document.createElement("button");
    removeButton.textContent = "删除选中";
    const status = document.createElement("div");
    status.style.fontSize = "12px";
    status.style.opacity = "0.85";
    status.style.flex = "1";
    status.style.minWidth = "220px";
    toolbar.append(pickButton, uploadButton, playButton, addButton, removeButton, status);

    // 换算提示：秒数 → 实际每段帧数（按目标模型对齐），一行说清
    const hint = document.createElement("div");
    hint.style.fontSize = "12px";
    hint.style.lineHeight = "1.5";
    hint.style.padding = "4px 8px";
    hint.style.borderRadius = "6px";
    hint.style.background = "rgba(255,255,255,0.05)";
    hint.style.border = "1px solid rgba(255,255,255,0.10)";
    hint.style.color = "#9fb3c8";
    hint.style.fontFamily = "monospace";
    hint.style.whiteSpace = "pre-wrap";
    hint.textContent = "选好视频后显示「秒 → 实际帧数」的换算结果";

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = `${CANVAS_H}px`;
    canvas.style.border = "1px solid rgba(255,255,255,0.18)";
    canvas.style.borderRadius = "8px";
    canvas.style.background = "#111";
    canvas.style.cursor = "crosshair";
    canvas.style.display = "block";

    const video = document.createElement("video");
    video.preload = "auto";
    video.style.display = "none";
    video.muted = false;

    container.append(toolbar, hint, canvas, video);

    const domWidget = typeof node.addDOMWidget === "function"
        ? node.addDOMWidget("video_segment_editor", "video_segment_editor", container, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => "",
            setValue: () => {},
        })
        : null;
    if (domWidget) domWidget.computeSize = () => [DEFAULT_WIDTH - 30, CANVAS_H + 72];

    const state = {
        node,
        container,
        canvas,
        video,
        status,
        playButton,
        peaks: null,
        strip: null,
        info: null,
        segments: [],
        selectedIndex: 0,
        pointerDown: false,
        error: "",
    };
    node.__fxaiVideoEditor = state;

    const info = () => state.info || {};
    const totalFrames = () => Number(info().frames) || 0;
    const fps = () => Number(info().fps) || 0;
    const duration = () => Number(info().duration) || 0;

    const currentKeyframes = () => normalizeKeyframes(
        parseKeyframes(getWidget(node, "关键帧JSON")?.value), duration(), fps());

    const syncKeyframes = (nextKeyframes, preferredSelectedIndex = null) => {
        const keyframesWidget = getWidget(node, "关键帧JSON");
        const normalized = normalizeKeyframes(nextKeyframes, duration(), fps());
        if (keyframesWidget) {
            keyframesWidget.value = JSON.stringify(normalized.map((value) => Number(value.toFixed(4))));
        }
        state.selectedIndex = Math.max(0, Math.min(
            preferredSelectedIndex ?? state.selectedIndex,
            Math.max(0, normalized.length - 1)));
        render();
    };

    // 秒 → 帧的换算说明。三种模型对齐规则不同，这里把中间步骤也写出来，
    // 免得看到「填 10 秒、结果 226 帧」以为算错了。
    const updateHint = () => {
        const rate = fps();
        const avgDur = Number(getWidget(node, "平均分段时长")?.value) || 0;
        const model = String(getWidget(node, "目标模型")?.value || "标准").trim();
        const avgSplit = Boolean(getWidget(node, "是否平均分段")?.value);

        if (!avgSplit) {
            hint.textContent = "未开启平均分段：只在关键帧处切，段长由标记位置决定";
            hint.style.color = "#8b98a6";
            return;
        }
        if (!(rate > 0) || !(avgDur > 0)) {
            hint.textContent = "选好视频后显示「秒 → 实际帧数」的换算结果";
            hint.style.color = "#8b98a6";
            return;
        }

        const raw = Math.max(1, Math.round(avgDur * rate));
        const step = segmentFrames(avgDur, rate, model);
        const name = model.toLowerCase();
        let detail = "";

        if (name === "minimax") {
            const h3 = alignDownH3(Math.round(avgDur * H3_FPS));
            detail = `按 H3 固定 ${H3_FPS}fps 折算 ${Math.round(avgDur * H3_FPS)} 帧`
                + ` → 17k+5 对齐 ${h3} 帧(H3)`
                + ` → 回源 ${rate.toFixed(3)}fps 得 ${step} 帧/段`;
        } else if (name === "ltx") {
            detail = `${avgDur.toFixed(2)} 秒 × ${rate.toFixed(3)}fps = ${raw} 帧`
                + ` → 8k+1 向下对齐 ${step} 帧/段`;
        } else {
            detail = `${avgDur.toFixed(2)} 秒 × ${rate.toFixed(3)}fps = ${step} 帧/段（四舍五入到帧）`;
        }

        const segs = state.segments.length;
        const tail = segs ? ` · 实际切成 ${segs} 段（末段可能并入前一段）` : "";
        hint.textContent = `【${model}】${detail} = ${(step / rate).toFixed(3)} 秒/段${tail}`;
        hint.style.color = "#9fd6ff";
    };

    const recomputeSegments = () => {
        const total = totalFrames();
        const rate = fps();
        if (!(total > 0) || !(rate > 0)) {
            state.segments = [];
            updateHint();
            return;
        }
        state.segments = buildFrameSegments(
            total,
            rate,
            currentKeyframes(),
            Boolean(getWidget(node, "跳过初始段")?.value),
            Boolean(getWidget(node, "包含尾部段")?.value),
            Boolean(getWidget(node, "是否平均分段")?.value),
            Number(getWidget(node, "平均分段时长")?.value) || 0,
            String(getWidget(node, "目标模型")?.value || "标准"),
        );
        updateHint();
    };

    const segmentAt = (seconds) => {
        const rate = fps();
        if (!(rate > 0) || !state.segments.length) return -1;
        const frame = Math.round(seconds * rate);
        for (let index = 0; index < state.segments.length; index += 1) {
            const seg = state.segments[index];
            if (frame >= seg.start && frame < seg.start + seg.count) return index;
        }
        return -1;
    };

    const render = () => {
        const width = Math.max(720, Math.floor(container.clientWidth || DEFAULT_WIDTH));
        canvas.width = width;
        canvas.height = CANVAS_H;
        const context = canvas.getContext("2d");
        if (!context) return;

        context.fillStyle = "#101317";
        context.fillRect(0, 0, width, CANVAS_H);

        if (state.error) {
            context.fillStyle = "#ff8f8f";
            context.font = "13px sans-serif";
            context.fillText(state.error, 12, 24);
            return;
        }

        const total = totalFrames();
        const rate = fps();
        const dur = duration();
        const time = Number(video.currentTime) || 0;
        const ratio = dur > 0 ? Math.max(0, Math.min(1, time / dur)) : 0;

        // ---- 当前帧预览 ----
        context.fillStyle = "#000";
        context.fillRect(6, 4, PREVIEW_W, PREVIEW_H);
        if (video.readyState >= 2) {
            try {
                context.drawImage(video, 6, 4, PREVIEW_W, PREVIEW_H);
            } catch {
                /* 帧还没就绪，下一帧再画 */
            }
        } else {
            context.fillStyle = "#556";
            context.font = "12px sans-serif";
            context.fillText("无预览", 60, 52);
        }
        context.strokeStyle = "rgba(255,255,255,0.22)";
        context.lineWidth = 1;
        context.strokeRect(6.5, 4.5, PREVIEW_W - 1, PREVIEW_H - 1);

        // ---- 右侧文字 ----
        const textLeft = PREVIEW_W + 22;
        context.fillStyle = "#ffffff";
        context.font = "bold 22px monospace";
        context.fillText(formatTime(time), textLeft, 32);

        context.fillStyle = "#8fb6d8";
        context.font = "12px monospace";
        const frameNo = rate > 0 ? Math.round(time * rate) : 0;
        context.fillText(`帧 ${frameNo} / ${total}`, textLeft, 52);

        const activeIndex = segmentAt(time);
        if (state.segments.length) {
            const seg = state.segments[Math.max(0, activeIndex)];
            const segText = activeIndex >= 0
                ? `第 ${activeIndex + 1}/${state.segments.length} 段 · 帧 ${seg.start}~${seg.start + seg.count - 1}`
                  + `（${seg.count} 帧 / ${(seg.count / rate).toFixed(3)} 秒）`
                : `不在任何段内 · 共 ${state.segments.length} 段`;
            context.fillStyle = "#ffd166";
            context.font = "13px sans-serif";
            context.fillText(segText, textLeft, 76);
        } else {
            context.fillStyle = "#778";
            context.font = "13px sans-serif";
            context.fillText("还没读到视频信息", textLeft, 76);
        }
        context.fillStyle = "#7f8c99";
        context.font = "11px monospace";
        context.fillText(
            `${info().width || 0}x${info().height || 0} · ${rate.toFixed(3)}fps · ${total} 帧`,
            textLeft, 94);

        // ---- 胶片带 ----
        context.fillStyle = "#0b0d10";
        context.fillRect(0, STRIP_TOP, width, STRIP_H);
        if (state.strip && state.strip.complete && state.strip.naturalWidth > 0) {
            context.drawImage(state.strip, 0, STRIP_TOP, width, STRIP_H);
        } else {
            context.fillStyle = "#33404d";
            context.font = "12px sans-serif";
            context.fillText("胶片带加载中…", 12, STRIP_TOP + STRIP_H / 2);
        }
        context.strokeStyle = "rgba(255,255,255,0.18)";
        context.strokeRect(0.5, STRIP_TOP + 0.5, width - 1, STRIP_H - 1);

        // ---- 时间刻度 ----
        context.fillStyle = "#6f8296";
        context.font = "10px monospace";
        for (let tick = 0; tick <= 10; tick += 1) {
            const x = (tick / 10) * width;
            context.fillRect(Math.min(width - 1, x), STRIP_TOP - 5, 1, 5);
            if (tick > 0 && tick < 10) {
                const label = formatTime((tick / 10) * dur);
                context.fillText(label, Math.min(width - 44, x + 3), STRIP_TOP - 8);
            }
        }

        // ---- 分段条 ----
        context.fillStyle = "#0e1318";
        context.fillRect(0, SEG_TOP, width, SEG_H);
        if (total > 0) {
            state.segments.forEach((seg, index) => {
                const x1 = (seg.start / total) * width;
                const x2 = ((seg.start + seg.count) / total) * width;
                context.fillStyle = index % 2 === 0 ? "#2c4358" : "#243a4d";
                context.fillRect(x1, SEG_TOP, Math.max(1, x2 - x1 - 1), SEG_H);
                if (x2 - x1 > 22) {
                    context.fillStyle = index === activeIndex ? "#ffd166" : "#9fd6ff";
                    context.font = "11px sans-serif";
                    context.fillText(String(index + 1), x1 + 3, SEG_TOP + 13);
                    if (x2 - x1 > 58) {
                        context.fillStyle = "#7fa8c8";
                        context.fillText(`${(seg.count / rate).toFixed(2)}s`, x1 + 3, SEG_TOP + 27);
                    }
                }
            });
            if (activeIndex >= 0) {
                const seg = state.segments[activeIndex];
                const x1 = (seg.start / total) * width;
                const x2 = ((seg.start + seg.count) / total) * width;
                context.strokeStyle = "#ffd166";
                context.lineWidth = 2;
                context.strokeRect(x1 + 1, SEG_TOP + 1, Math.max(2, x2 - x1 - 2), SEG_H - 2);
            }
        }

        // ---- 关键帧标记 ----
        const keyframes = currentKeyframes();
        keyframes.forEach((seconds, index) => {
            const x = dur > 0 ? (seconds / dur) * width : 0;
            context.strokeStyle = index === state.selectedIndex ? "#ffd166" : "#ff7f50";
            context.lineWidth = index === state.selectedIndex ? 3 : 2;
            context.beginPath();
            context.moveTo(x, STRIP_TOP - 4);
            context.lineTo(x, SEG_TOP + SEG_H);
            context.stroke();
            context.fillStyle = context.strokeStyle;
            context.font = "11px sans-serif";
            context.fillText(String(index + 1), Math.min(width - 12, x + 4), STRIP_TOP + 12 + (index % 2) * 13);
        });

        // ---- 播放头 ----
        const playX = ratio * width;
        context.strokeStyle = "#ffffff";
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(playX, STRIP_TOP - 6);
        context.lineTo(playX, SEG_TOP + SEG_H);
        context.stroke();

        const step = state.segments.length && total > 0
            ? Math.round(total / state.segments.length)
            : 0;
        status.textContent = `标记 ${keyframes.length} · 分段 ${state.segments.length}`
            + (step && rate > 0 ? ` · 约 ${(step / rate).toFixed(2)} 秒/段` : "")
            + ` · 选中标记 ${state.selectedIndex + 1}`;
    };

    // ---------- 加载 ----------
    const loadVideo = async () => {
        const pathWidget = getWidget(node, "视频文件");
        const raw = String(pathWidget?.value || "").trim();
        state.error = "";
        state.info = null;
        state.segments = [];
        state.strip = null;
        if (!raw) {
            render();
            return;
        }

        try {
            const response = await fetch(api.apiURL(`/fxai/video/segments/info?path=${encodeURIComponent(raw)}`));
            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.error || "读取视频信息失败");

            state.info = payload;
            // 服务端能算出 input 下的相对路径就写回控件，
            // 工作流换机器、走 API 调用都不用关心物理路径
            if (payload.rel && payload.rel !== raw && pathWidget) {
                pathWidget.value = payload.rel;
            }
            video.src = api.apiURL(`/fxai/video/segments/file?path=${encodeURIComponent(payload.path)}`);
            video.load();

            const strip = new Image();
            strip.onload = () => render();
            strip.onerror = () => { state.strip = null; render(); };
            strip.src = api.apiURL(`/fxai/video/segments/strip?path=${encodeURIComponent(payload.path)}&tiles=${STRIP_TILES}`);
            state.strip = strip;

            syncKeyframes(parseKeyframes(getWidget(node, "关键帧JSON")?.value), state.selectedIndex);
            recomputeSegments();
            render();
        } catch (error) {
            state.error = error?.message || String(error);
            render();
        }
    };

    const seekToPosition = (event) => {
        if (!(duration() > 0)) return;
        const rect = canvas.getBoundingClientRect();
        const offsetX = event.clientX - rect.left;
        const positionRatio = Math.max(0, Math.min(1, offsetX / rect.width));
        let target = positionRatio * duration();
        const rate = fps();
        if (rate > 0) target = Math.round(target * rate) / rate;
        video.currentTime = Math.min(target, Math.max(0, duration() - 0.001));

        const keyframes = currentKeyframes();
        if (!keyframes.length) {
            render();
            return;
        }
        let nearest = 0;
        let distance = Number.POSITIVE_INFINITY;
        for (let index = 0; index < keyframes.length; index += 1) {
            const gap = Math.abs(keyframes[index] - video.currentTime);
            if (gap < distance) {
                distance = gap;
                nearest = index;
            }
        }
        if (distance <= Math.max(0.05, duration() / 200)) state.selectedIndex = nearest;
        render();
    };

    // ---------- 交互 ----------
    pickButton.onclick = () => {
        const pathWidget = getWidget(node, "视频文件");
        let startDir = String(pathWidget?.value || "").trim();
        if (/\.(mp4|avi|mov|mkv|flv|wmv|mpeg|mpg|webm)$/i.test(startDir)) {
            startDir = startDir.replace(/[\\/][^\\/]*$/, "");
        }
        const picker = window.FxAiVideoPicker;
        if (typeof picker !== "function") {
            alert("视频选片窗口没加载出来，刷新一下页面再试");
            return;
        }
        picker(startDir).then((selection) => {
            if (!selection) return;
            const target = selection.path || selection.name;
            if (!target) return;
            if (pathWidget) {
                pathWidget.value = target;
                if (pathWidget.callback) pathWidget.callback(target);
            }
            loadVideo();
        });
    };

    uploadButton.onclick = () => {
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);
        fileInput.click();
        fileInput.onchange = async () => {
            if (!fileInput.files.length) {
                fileInput.remove();
                return;
            }
            const file = fileInput.files[0];
            const originalText = uploadButton.textContent;
            uploadButton.disabled = true;
            try {
                const name = await uploadFile(file, (progress) => {
                    uploadButton.textContent = `上传中 ${Math.round(progress * 100)}%`;
                });
                const pathWidget = getWidget(node, "视频文件");
                if (pathWidget) {
                    pathWidget.value = name;
                    if (pathWidget.callback) pathWidget.callback(name);
                }
                await loadVideo();
            } catch (error) {
                alert(`上传失败: ${error.message}`);
            } finally {
                uploadButton.textContent = originalText;
                uploadButton.disabled = false;
                fileInput.remove();
            }
        };
    };

    playButton.addEventListener("click", async () => {
        if (video.paused) {
            try {
                await video.play();
            } catch (error) {
                console.warn("视频播放失败", error);
            }
        } else {
            video.pause();
        }
    });

    addButton.addEventListener("click", () => {
        let current = Number(video.currentTime) || 0;
        const rate = fps();
        if (rate > 0) current = Math.round(current * rate) / rate;
        if (duration() > 0) current = Math.min(current, duration() - 1 / Math.max(1, rate));
        const keyframes = parseKeyframes(getWidget(node, "关键帧JSON")?.value);
        keyframes.push(current);
        const normalized = normalizeKeyframes(keyframes, duration(), fps());
        const step = rate > 0 ? 1 / rate : 0.1;
        const bucket = Math.round(current / step);
        const preferred = normalized.findIndex((value) => Math.round(value / step) === bucket);
        syncKeyframes(normalized, preferred >= 0 ? preferred : normalized.length - 1);
        recomputeSegments();
        render();
    });

    removeButton.addEventListener("click", () => {
        const keyframes = currentKeyframes();
        if (!keyframes.length) return;
        keyframes.splice(state.selectedIndex, 1);
        syncKeyframes(keyframes, Math.max(0, state.selectedIndex - 1));
        recomputeSegments();
        render();
    });

    video.addEventListener("play", () => { playButton.textContent = "暂停"; });
    video.addEventListener("pause", () => { playButton.textContent = "播放"; });
    video.addEventListener("timeupdate", render);
    video.addEventListener("loadedmetadata", () => {
        recomputeSegments();
        render();
    });
    video.addEventListener("seeked", render);

    canvas.addEventListener("pointerdown", (event) => { state.pointerDown = true; seekToPosition(event); });
    canvas.addEventListener("pointermove", (event) => { if (state.pointerDown) seekToPosition(event); });
    window.addEventListener("pointerup", () => { state.pointerDown = false; });

    // 参数一变就重算分段条，不用重新拉视频
    ["跳过初始段", "包含尾部段", "是否平均分段", "平均分段时长", "目标模型"].forEach((name) => {
        const widget = getWidget(node, name);
        if (!widget || widget.__fxaiHooked) return;
        const original = widget.callback;
        widget.callback = function () {
            const result = original?.apply(this, arguments);
            recomputeSegments();
            render();
            return result;
        };
        widget.__fxaiHooked = true;
    });

    const pathWidget = getWidget(node, "视频文件");
    if (pathWidget && !pathWidget.__fxaiHooked) {
        const original = pathWidget.callback;
        pathWidget.callback = function () {
            const result = original?.apply(this, arguments);
            loadVideo();
            return result;
        };
        pathWidget.__fxaiHooked = true;
    }

    hideWidget(getWidget(node, "关键帧JSON"));

    state.render = render;
    state.reload = loadVideo;
    state.recompute = recomputeSegments;

    requestAnimationFrame(() => {
        render();
        loadVideo();
        resizeNode(node);
    });

    return state;
}

function refresh(node) {
    hideWidget(getWidget(node, "关键帧JSON"));
    const editor = buildEditor(node);
    const keyframes = normalizeKeyframes(
        parseKeyframes(getWidget(node, "关键帧JSON")?.value),
        editor.info?.duration || 0,
        editor.info?.fps || 0);
    editor.selectedIndex = Math.max(0, Math.min(editor.selectedIndex, Math.max(0, keyframes.length - 1)));
    if (editor.container.isConnected) editor.reload?.();
}

app.registerExtension({
    name: "FxAiVideoSegmenterV1",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== TARGET_CLASS) return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            refresh(this);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = originalOnConnectionsChange?.apply(this, arguments);
            refresh(this);
            return result;
        };
    },
    async nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) return;
        buildEditor(node);
        refresh(node);
    },
});
