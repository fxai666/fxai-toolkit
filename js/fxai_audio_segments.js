import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiAudioSegmenter";
const DEFAULT_WIDTH = 820;
const DEFAULT_HEIGHT = 430;
const SNAP_STEP_SECONDS = 1.0;

function getWidget(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name);
}

function resizeNode(node) {
    const computedSize = node.computeSize?.();
    const width = Math.max(DEFAULT_WIDTH, computedSize?.[0] ?? 0, Array.isArray(node.size) ? node.size[0] : 0);
    const height = Math.max(DEFAULT_HEIGHT, computedSize?.[1] ?? 0, Array.isArray(node.size) ? node.size[1] : 0);
    node.size = [width, height];
    app.graph.setDirtyCanvas(true, true);
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

function snapTime(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    return Math.round(value / SNAP_STEP_SECONDS) * SNAP_STEP_SECONDS;
}

function normalizeKeyframes(keyframes, duration) {
    const unique = [];
    const seen = new Set();
    for (const value of keyframes || []) {
        let seconds = snapTime(value);
        if (Number.isFinite(duration) && duration > 0) {
            seconds = Math.min(seconds, Math.max(0, duration - 0.001));
        }
        const bucket = Math.round(seconds / SNAP_STEP_SECONDS);
        if (seen.has(bucket)) continue;
        seen.add(bucket);
        unique.push(seconds);
    }
    unique.sort((left, right) => left - right);
    return unique;
}

function formatTime(seconds) {
    const clamped = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(clamped / 60);
    const remainder = clamped - minutes * 60;
    return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function normalizeSegmentIndexWidget(node) {
    const widget = getWidget(node, "分段索引");
    if (!widget) return;
    const raw = widget.value;
    if (raw === "" || raw === null || raw === undefined || Number.isNaN(Number(raw))) {
        widget.value = 0;
    } else {
        widget.value = Math.max(0, Math.floor(Number(raw)));
    }
}

// ---------- 上传功能 ----------
async function uploadFile(file, onProgress) {
    const body = new FormData();
    body.append("image", file);
    return new Promise((resolve, reject) => {
        const req = new XMLHttpRequest();
        req.upload.onprogress = (e) => onProgress?.(e.loaded / e.total);
        req.onload = () => {
            if (req.status === 200) {
                try {
                    const data = JSON.parse(req.responseText);
                    resolve(data.name);
                } catch (e) {
                    reject(new Error("解析响应失败"));
                }
            } else {
                reject(new Error(`上传失败: ${req.status}`));
            }
        };
        req.onerror = () => reject(new Error("网络错误"));
        req.open("POST", api.apiURL("/upload/image"), true);
        req.send(body);
    });
}

// ---------- 波形编辑器构建 ----------
function buildEditor(node) {
    if (node.__fxaiWaveformEditor) return node.__fxaiWaveformEditor;

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "8px";
    container.style.width = "100%";

    // ========== 新增：全局阻止拖拽默认行为 ==========
    function preventDefaultDragDrop() {
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => e.preventDefault());
        document.addEventListener('dragenter', (e) => e.preventDefault());
        document.addEventListener('dragleave', (e) => e.preventDefault());
    }
    preventDefaultDragDrop();

    // ========== 新增：拖拽上传区域 ==========
    const dropArea = document.createElement("div");
    dropArea.style.padding = "15px 12px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.style.fontSize = "12px";
    dropArea.textContent = "📥 拖拽音频到这里上传";
    container.appendChild(dropArea);

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.flexWrap = "wrap";
    toolbar.style.gap = "8px";
    toolbar.style.alignItems = "center";    

    // 原有上传按钮
    const uploadButton = document.createElement("button");
    uploadButton.textContent = "上传音频";
    uploadButton.style.padding = "2px 6px";
    uploadButton.style.cursor = "pointer";
    uploadButton.style.fontSize = "12px";
    toolbar.appendChild(uploadButton);

    // 原有播放/添加/删除按钮
    const playButton = document.createElement("button");
    playButton.textContent = "播放";
    const addButton = document.createElement("button");
    addButton.textContent = "添加标记";
    const removeButton = document.createElement("button");
    removeButton.textContent = "删除选中";
    const status = document.createElement("div");
    status.style.fontSize = "12px";
    status.style.opacity = "0.85";
    toolbar.append(playButton, addButton, removeButton, status);

    // ========== 新增：拖拽上传逻辑 ==========
    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.style.borderColor = "#fff";
    });
    dropArea.addEventListener("dragleave", () => {
        dropArea.style.borderColor = "#777";
    });
    dropArea.addEventListener("drop", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.style.borderColor = "#777";
        const files = Array.from(e.dataTransfer.files);
        if (!files.length || !files[0].type.startsWith("audio/")) {
            alert("请上传音频文件！");
            return;
        }

        const file = files[0];
        const originalText = uploadButton.textContent;
        uploadButton.textContent = "上传中...";
        uploadButton.disabled = true;
        try {
            const filename = await uploadFile(file, (p) => {
                uploadButton.textContent = `上传中 ${Math.round(p * 100)}%`;
            });
            const pathWidget = getWidget(node, "音频文件");
            if (pathWidget) {
                if (!pathWidget.options.values.includes(filename)) {
                    pathWidget.options.values.push(filename);
                    pathWidget.options.values.sort();
                }
                pathWidget.value = filename;
                if (pathWidget.callback) {
                    pathWidget.callback(filename);
                }
            } else {
                console.warn("未找到音频文件控件");
            }
        } catch (err) {
            console.error("上传失败:", err);
            alert("上传失败: " + err.message);
        } finally {
            uploadButton.textContent = originalText;
            uploadButton.disabled = false;
        }
    });

    // 原有Canvas和Audio部分保持不变
    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "180px";
    canvas.style.border = "1px solid rgba(255,255,255,0.18)";
    canvas.style.borderRadius = "8px";
    canvas.style.background = "#111";
    canvas.style.cursor = "pointer";

    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.style.display = "none";

    container.append(toolbar, canvas, audio);

    const domWidget = typeof node.addDOMWidget === "function"
        ? node.addDOMWidget("waveform_editor", "waveform_editor", container, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => "",
            setValue: () => {},
        })
        : null;
    if (domWidget) {
        domWidget.computeSize = () => [DEFAULT_WIDTH - 30, 280];
    }

    const state = {
        node,
        container,
        canvas,
        audio,
        status,
        playButton,
        addButton,
        removeButton,
            peaks: [],
            duration: 0,
            selectedIndex: 0,
            pointerDown: false,
            error: "",
        };

    node.__fxaiWaveformEditor = state;

        // 上传逻辑
    uploadButton.onclick = () => {
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "audio/*";
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
            uploadButton.textContent = "上传中...";
            uploadButton.disabled = true;
            try {
                const filename = await uploadFile(file, (p) => {
                    uploadButton.textContent = `上传中 ${Math.round(p * 100)}%`;
                });
                const pathWidget = getWidget(node, "音频文件");
                if (pathWidget) {
                    if (!pathWidget.options.values.includes(filename)) {
                        pathWidget.options.values.push(filename);
                        pathWidget.options.values.sort();
                    }
                    pathWidget.value = filename;
                    if (pathWidget.callback) {
                        pathWidget.callback(filename);
                    }
                } else {
                    console.warn("未找到音频文件控件");
                }
            } catch (err) {
                console.error("上传失败:", err);
                alert("上传失败: " + err.message);
            } finally {
                uploadButton.textContent = originalText;
                uploadButton.disabled = false;
                fileInput.remove();
            }
        };
    };

    const render = () => {
        const width = Math.max(760, Math.floor(container.clientWidth || DEFAULT_WIDTH));
        const height = 180;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        if (!context) return;

        context.clearRect(0, 0, width, height);
        context.fillStyle = "#101317";
        context.fillRect(0, 0, width, height);

        if (state.error) {
            context.fillStyle = "#ff8f8f";
            context.font = "14px sans-serif";
            context.fillText(state.error, 12, 24);
            return;
        }

        const midY = Math.floor(height / 2);
        context.strokeStyle = "rgba(255,255,255,0.12)";
        context.beginPath();
        context.moveTo(0, midY);
        context.lineTo(width, midY);
        context.stroke();

        if (state.peaks.length) {
            const barWidth = width / state.peaks.length;
            context.fillStyle = "#71d0ff";
            for (let index = 0; index < state.peaks.length; index += 1) {
                const peak = Math.max(0, Math.min(1, state.peaks[index] || 0));
                const barHeight = Math.max(1, peak * (height * 0.42));
                const x = index * barWidth;
                context.fillRect(x, midY - barHeight, Math.max(1, barWidth - 1), barHeight * 2);
            }
        }

        const keyframesWidget = getWidget(node, "关键帧JSON");
        const keyframes = normalizeKeyframes(parseKeyframes(keyframesWidget?.value), state.duration);
        keyframes.forEach((time, index) => {
            const ratio = state.duration > 0 ? time / state.duration : 0;
            const x = Math.max(0, Math.min(width, ratio * width));
            context.strokeStyle = index === state.selectedIndex ? "#ffd166" : "#ff7f50";
            context.lineWidth = index === state.selectedIndex ? 3 : 2;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
            context.fillStyle = context.strokeStyle;
            context.font = "12px sans-serif";
            context.fillText(String(index + 1), Math.min(width - 14, x + 4), 14 + (index % 2) * 14);
        });

        if (state.duration > 0) {
            const ratio = Math.max(0, Math.min(1, (audio.currentTime || 0) / state.duration));
            const x = ratio * width;
            context.strokeStyle = "#ffffff";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
        }

        const skipWidget = getWidget(node, "跳过初始段");
        const tailWidget = getWidget(node, "包含尾部段");
        status.textContent = `时间 ${formatTime(audio.currentTime || 0)} / ${formatTime(state.duration)} | 标记 ${keyframes.length} | 选中 ${state.selectedIndex + 1} | 跳过首段 ${Boolean(skipWidget?.value)} | 包含尾段 ${Boolean(tailWidget?.value)}`;
    };

    const syncKeyframes = (nextKeyframes, preferredSelectedIndex = null) => {
        const keyframesWidget = getWidget(node, "关键帧JSON");
        const normalized = normalizeKeyframes(nextKeyframes, state.duration);
        const serialized = JSON.stringify(normalized.map((value) => Number(value.toFixed(3))));
        if (keyframesWidget) keyframesWidget.value = serialized;
        state.selectedIndex = Math.max(0, Math.min(preferredSelectedIndex ?? state.selectedIndex, Math.max(0, normalized.length - 1)));
        render();
    };

    const loadWaveform = async () => {
        const audioFileWidget = getWidget(node, "音频文件");
        const audioFile = String(audioFileWidget?.value || "").trim();
        state.error = "";
        if (!audioFile) {
            state.peaks = [];
            state.duration = 0;
            render();
            return;
        }

        try {
            const waveformUrl = api.apiURL(`/fxai/audio-waveform?audio_file=${encodeURIComponent(audioFile)}&bins=1400`);
            const response = await fetch(waveformUrl);
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload?.error || "加载波形失败");
            }

            state.peaks = Array.isArray(payload.peaks) ? payload.peaks : [];
            state.duration = Number(payload.duration) || 0;
            const audioUrl = api.apiURL(`/fxai/audio-file?audio_file=${encodeURIComponent(audioFile)}`);
            audio.src = audioUrl;
            audio.load();
            const currentKeyframes = parseKeyframes(getWidget(node, "关键帧JSON")?.value);
            syncKeyframes(currentKeyframes, state.selectedIndex);
        } catch (error) {
            state.error = error?.message || String(error);
            state.peaks = [];
            state.duration = 0;
            render();
        }
    };

    const seekToPosition = (event) => {
        if (!(state.duration > 0)) return;
        const rect = canvas.getBoundingClientRect();
        const offsetX = event.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, offsetX / rect.width));
        let snapped = snapTime(ratio * state.duration);
        if (state.duration > 0) {
            snapped = Math.min(snapped, Math.max(0, state.duration - 0.001));
        }
        audio.currentTime = snapped;

        const keyframes = normalizeKeyframes(parseKeyframes(getWidget(node, "关键帧JSON")?.value), state.duration);
        if (!keyframes.length) {
            render();
            return;
        }
        let nearestIndex = 0;
        let nearestDistance = Number.POSITIVE_INFINITY;
        for (let index = 0; index < keyframes.length; index += 1) {
            const distance = Math.abs(keyframes[index] - audio.currentTime);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestIndex = index;
            }
        }
        if (nearestDistance <= Math.max(0.2, state.duration / 100)) {
            state.selectedIndex = nearestIndex;
        }
        render();
    };

    playButton.addEventListener("click", async () => {
        if (audio.paused) {
            await audio.play();
        } else {
            audio.pause();
        }
    });

    addButton.addEventListener("click", () => {
        let current = snapTime(audio.currentTime || 0);
        if (state.duration > 0) {
            current = Math.min(current, Math.max(0, state.duration - 0.001));
        }
        audio.currentTime = current;
        const keyframes = parseKeyframes(getWidget(node, "关键帧JSON")?.value);
        keyframes.push(current);
        const normalized = normalizeKeyframes(keyframes, state.duration);
        const bucket = Math.round(current / SNAP_STEP_SECONDS);
        const preferredIndex = normalized.findIndex((value) => Math.round(value / SNAP_STEP_SECONDS) === bucket);
        syncKeyframes(normalized, preferredIndex >= 0 ? preferredIndex : normalized.length - 1);
    });

    removeButton.addEventListener("click", () => {
        const keyframes = normalizeKeyframes(parseKeyframes(getWidget(node, "关键帧JSON")?.value), state.duration);
        if (!keyframes.length) return;
        keyframes.splice(state.selectedIndex, 1);
        syncKeyframes(keyframes, Math.max(0, state.selectedIndex - 1));
    });

    audio.addEventListener("play", () => { playButton.textContent = "暂停"; render(); });
    audio.addEventListener("pause", () => { playButton.textContent = "播放"; render(); });
    audio.addEventListener("timeupdate", render);
    audio.addEventListener("loadedmetadata", render);

    canvas.addEventListener("pointerdown", (event) => { state.pointerDown = true; seekToPosition(event); });
    canvas.addEventListener("pointermove", (event) => { if (state.pointerDown) seekToPosition(event); });
    window.addEventListener("pointerup", () => { state.pointerDown = false; });

    const audioFileWidget = getWidget(node, "音频文件");
    if (audioFileWidget && !audioFileWidget.__fxaiHooked) {
        const originalCallback = audioFileWidget.callback;
        audioFileWidget.callback = function (value) {
            const result = originalCallback?.apply(this, arguments);
            loadWaveform();
            return result;
        };
        audioFileWidget.__fxaiHooked = true;
    }

    const keyframesWidget = getWidget(node, "关键帧JSON");
    const renderIdWidget = getWidget(node, "渲染ID");
    hideWidget(keyframesWidget);
    hideWidget(renderIdWidget);

    requestAnimationFrame(() => {
        render();
        loadWaveform();
        resizeNode(node);
    });

    return state;
}

function syncFromStoredState(node) {
    normalizeSegmentIndexWidget(node);
    const keyframesWidget = getWidget(node, "关键帧JSON");
    const renderIdWidget = getWidget(node, "渲染ID");
    hideWidget(keyframesWidget);
    hideWidget(renderIdWidget);

    const editor = buildEditor(node);
    const keyframes = normalizeKeyframes(parseKeyframes(keyframesWidget?.value), editor.duration);
    editor.selectedIndex = Math.max(0, Math.min(editor.selectedIndex, Math.max(0, keyframes.length - 1)));
    if (editor.container.isConnected) {
        const skipWidget = getWidget(node, "跳过初始段");
        const tailWidget = getWidget(node, "包含尾部段");
        editor.status.textContent = `时间 ${formatTime(editor.audio.currentTime || 0)} / ${formatTime(editor.duration)} | 标记 ${keyframes.length} | 选中 ${editor.selectedIndex + 1} | 跳过首段 ${Boolean(skipWidget?.value)} | 包含尾段 ${Boolean(tailWidget?.value)}`;
    }
}

app.registerExtension({
    name: "FxAiAudioSegmenter",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== TARGET_CLASS) return;

const originalOnConfigure = nodeType.prototype.onConfigure;
nodeType.prototype.onConfigure = function () {
    const result = originalOnConfigure?.apply(this, arguments);
    syncFromStoredState(this);
    return result;
};

const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
nodeType.prototype.onConnectionsChange = function () {
    const result = originalOnConnectionsChange?.apply(this, arguments);
    syncFromStoredState(this);
    return result;
};
},
async nodeCreated(node) {
    if (node.comfyClass !== TARGET_CLASS) return;
    buildEditor(node);
    syncFromStoredState(node);
},
});