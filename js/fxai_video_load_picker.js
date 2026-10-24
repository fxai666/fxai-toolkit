// Copyright (c) 2026 凤希AI/www.fxai.site
// Licensed under MIT License
// 商用需购买商业授权

/**
 * 凤希AI - 视频加载 · 可视化选片
 * ==============================
 *
 * 给 FxAiVideoLoad 加一个和「角色资源图片选择器」同款的界面：
 * 节点上直接显示当前视频的首帧缩略图，点「选择视频」弹出网格选片窗口。
 *
 * 选完回填的是节点本来就有的两个参数：
 *     视频文件夹路径 = 该视频所在目录（绝对路径）
 *     视频索引       = 该视频在这个目录里的排序号
 * 所以后端逻辑、老工作流、序列化格式全都不用动。
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

(function () {

    const NODE_TYPE = "FxAiVideoLoad";
    const W_DIR = "视频文件夹路径";
    const W_INDEX = "视频索引";

    // 「视频文件夹路径」现在两种用法：填目录（走索引）或填视频文件（索引失效）。
    // 前端靠扩展名区分，够用且不会误判——目录名几乎不会以 .mp4 结尾。
    const VIDEO_EXT_RE = /\.(mp4|avi|mov|mkv|flv|wmv|mpeg|mpg|webm)$/i;
    function isVideoFile(p) {
        return VIDEO_EXT_RE.test(String(p || "").trim());
    }
    function baseName(p) {
        const s = String(p || "").replace(/\\/g, "/");
        return s.substring(s.lastIndexOf("/") + 1);
    }

    // ---------------------------------------------------------------- 工具
    function fmtSize(n) {
        if (!n) return "";
        if (n >= 1024 * 1024 * 1024) return (n / 1024 / 1024 / 1024).toFixed(2) + " GB";
        if (n >= 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MB";
        if (n >= 1024) return (n / 1024).toFixed(0) + " KB";
        return n + " B";
    }

    function fmtTime(sec) {
        if (sec === undefined || sec === null || isNaN(sec)) return "";
        sec = Math.max(0, Math.round(sec));
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return m + ":" + String(s).padStart(2, "0");
    }

    function apiUrl(path) {
        return api.apiURL(path);
    }

    function thumbUrl(video) {
        return apiUrl("/fxai/video/thumb?path=" + encodeURIComponent(video.path) +
                      "&t=" + (video.mtime || 0));
    }

    // 同一个目录的列表缓存，避免来回切标签重复请求
    const browseCache = new Map();
    function browse(dir, force) {
        const key = dir || "";
        if (!force && browseCache.has(key)) return Promise.resolve(browseCache.get(key));
        return fetch(apiUrl("/fxai/video/browse?dir=" + encodeURIComponent(key)))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data && !data.error) browseCache.set(key, data);
                return data;
            })
            .catch(function () { return { error: "请求失败" }; });
    }

    // ---------------------------------------------------------------- 样式
    function injectStyle() {
        if (document.getElementById("fxai-vload-style")) return;
        const style = document.createElement("style");
        style.id = "fxai-vload-style";
        style.textContent = `
            .fxai-vload-wrap {
                display: flex; flex-direction: column; gap: 6px;
                width: 100%; box-sizing: border-box; padding: 4px 2px 2px;
                color: #ddd; font-family: system-ui, "Microsoft YaHei", sans-serif;
            }
            .fxai-vload-row { display: flex; gap: 8px; align-items: center; }
            .fxai-vload-thumb {
                width: 96px; height: 54px; flex-shrink: 0; border-radius: 5px;
                background: #111; overflow: hidden; position: relative;
                display: flex; align-items: center; justify-content: center;
                color: #666; font-size: 11px; border: 1px solid #333;
            }
            .fxai-vload-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
            .fxai-vload-meta { flex: 1; min-width: 0; }
            .fxai-vload-name {
                font-size: 12px; color: #eee; line-height: 1.35;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            }
            .fxai-vload-sub { font-size: 11px; color: #888; line-height: 1.4; }
            .fxai-vload-btn {
                width: 100%; height: 28px; border: none; border-radius: 4px;
                background: #4a8fff; color: #fff; cursor: pointer; font-size: 13px;
            }
            .fxai-vload-btn:hover { background: #5b9bff; }

            /* ---------- 弹窗 ---------- */
            .fxai-vp-mask {
                position: fixed; inset: 0; z-index: 99999;
                background: rgba(0,0,0,0.85);
                display: flex; align-items: center; justify-content: center;
            }
            .fxai-vp-modal {
                width: 980px; max-width: 95vw; height: 760px; max-height: 92vh;
                background: #222; border-radius: 10px; padding: 12px;
                box-sizing: border-box; display: flex; flex-direction: column;
                gap: 10px; color: #ddd;
                font-family: system-ui, "Microsoft YaHei", sans-serif;
            }
            .fxai-vp-title { font-size: 17px; font-weight: bold; color: #fff; }
            .fxai-vp-path {
                font-size: 12px; color: #9ab; background: #1a1a1a; border-radius: 4px;
                padding: 5px 8px; word-break: break-all; font-family: Consolas, monospace;
            }
            .fxai-vp-bar { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
            .fxai-vp-chip {
                padding: 3px 9px; border: none; border-radius: 4px; cursor: pointer;
                background: #333; color: #ccc; font-size: 12px;
            }
            .fxai-vp-chip:hover { background: #3d3d3d; }
            .fxai-vp-chip.on { background: #4a8fff; color: #fff; }
            .fxai-vp-grid {
                flex: 1; overflow-y: auto; display: flex; flex-wrap: wrap;
                gap: 8px; align-content: flex-start; padding: 2px;
                background: #1b1b1b; border-radius: 6px; padding: 8px;
            }
            .fxai-vp-card {
                width: 180px; border-radius: 6px; overflow: hidden; cursor: pointer;
                border: 3px solid transparent; background: #111; position: relative;
                flex-shrink: 0;
            }
            .fxai-vp-card.on { border-color: #4a8fff; }
            .fxai-vp-card .thumb {
                width: 100%; height: 100px; object-fit: cover; display: block;
                background: #000;
            }
            .fxai-vp-card .thumb-ph {
                width: 100%; height: 100px; display: flex; align-items: center;
                justify-content: center; color: #555; font-size: 12px; background: #0d0d0d;
            }
            .fxai-vp-card .cap { padding: 5px 6px 6px; }
            .fxai-vp-card .nm {
                font-size: 12px; color: #ddd; line-height: 1.3;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            }
            .fxai-vp-card .sub { font-size: 11px; color: #777; margin-top: 2px; }
            .fxai-vp-card .idx {
                position: absolute; top: 0; left: 0; background: #4a8fff; color: #fff;
                font-size: 11px; font-weight: bold; padding: 1px 6px; border-radius: 0 0 5px 0;
            }
            .fxai-vp-empty {
                width: 100%; text-align: center; color: #777; font-size: 13px;
                padding: 60px 0; line-height: 1.9;
            }
            .fxai-vp-foot {
                display: flex; justify-content: space-between; align-items: center; gap: 10px;
            }
            .fxai-vp-foot .left { font-size: 12px; color: #888; }
            .fxai-vp-foot .right { display: flex; gap: 10px; }
            .fxai-vp-foot button {
                padding: 7px 16px; border: none; border-radius: 4px; cursor: pointer;
                font-size: 13px;
            }
            .fxai-vp-cancel { background: #555; color: #eee; }
            .fxai-vp-ok { background: #4a8fff; color: #fff; }
            .fxai-vp-ok:disabled { background: #34527f; color: #9bb; cursor: not-allowed; }
        `;
        document.head.appendChild(style);
    }

    // ---------------------------------------------------------------- 弹窗
    /**
     * @param {string} startDir 打开时定位的目录（一般是节点当前填的目录）
     * @returns {Promise<{dir:string,index:number,name:string}|undefined>}
     */
    function openVideoPicker(startDir) {
        injectStyle();
        return new Promise(function (resolve) {

            let curDir = startDir || "";
            let picked = null;          // 当前选中的 {dir,index,name}
            let curData = null;

            const mask = document.createElement("div");
            mask.className = "fxai-vp-mask";

            const modal = document.createElement("div");
            modal.className = "fxai-vp-modal";
            mask.appendChild(modal);

            const title = document.createElement("div");
            title.className = "fxai-vp-title";
            title.textContent = "🎬 选择视频";
            modal.appendChild(title);

            const pathEl = document.createElement("div");
            pathEl.className = "fxai-vp-path";
            modal.appendChild(pathEl);

            const rootBar = document.createElement("div");
            rootBar.className = "fxai-vp-bar";
            modal.appendChild(rootBar);

            const dirBar = document.createElement("div");
            dirBar.className = "fxai-vp-bar";
            modal.appendChild(dirBar);

            const grid = document.createElement("div");
            grid.className = "fxai-vp-grid";
            modal.appendChild(grid);

            const foot = document.createElement("div");
            foot.className = "fxai-vp-foot";
            const footLeft = document.createElement("div");
            footLeft.className = "left";
            const footRight = document.createElement("div");
            footRight.className = "right";
            const btnCancel = document.createElement("button");
            btnCancel.className = "fxai-vp-cancel";
            btnCancel.textContent = "取消";
            const btnIdx = document.createElement("button");
            btnIdx.className = "fxai-vp-cancel";
            btnIdx.textContent = "按目录+索引";
            btnIdx.title = "把目录和索引填进去（老方式，适合按索引批量换视频）";
            btnIdx.disabled = true;
            const btnOk = document.createElement("button");
            btnOk.className = "fxai-vp-ok";
            btnOk.textContent = "✅ 用完整路径";
            btnOk.title = "直接把视频文件的完整路径填进去，最稳";
            btnOk.disabled = true;
            footRight.append(btnCancel, btnIdx, btnOk);
            foot.append(footLeft, footRight);
            modal.appendChild(foot);

            function close(result) {
                if (mask.parentNode) mask.parentNode.removeChild(mask);
                resolve(result);
            }

            btnCancel.onclick = function () { close(); };
            btnOk.onclick = function () {
                if (picked) close(Object.assign({ mode: "path" }, picked));
            };
            btnIdx.onclick = function () {
                if (picked) close(Object.assign({ mode: "index" }, picked));
            };
            mask.onclick = function (e) { if (e.target === mask) close(); };
            document.addEventListener("keydown", function esc(e) {
                if (e.key === "Escape") { document.removeEventListener("keydown", esc); close(); }
            });

            // ---- 顶部快捷入口 ----
            function renderRoots(roots) {
                rootBar.innerHTML = "";
                (roots || []).forEach(function (r) {
                    const b = document.createElement("button");
                    b.className = "fxai-vp-chip";
                    b.textContent = r.label;
                    b.title = r.path;
                    b.onclick = function () { load(r.path); };
                    rootBar.appendChild(b);
                });
            }

            // ---- 子目录 ----
            function renderDirs(data) {
                dirBar.innerHTML = "";
                if (data.parent) {
                    const up = document.createElement("button");
                    up.className = "fxai-vp-chip";
                    up.textContent = "⬆ 上一级";
                    up.onclick = function () { load(data.parent); };
                    dirBar.appendChild(up);
                }
                (data.dirs || []).forEach(function (d) {
                    const b = document.createElement("button");
                    b.className = "fxai-vp-chip";
                    b.textContent = "📁 " + d.name + (d.videos ? " (" + d.videos + ")" : "");
                    b.onclick = function () { load(d.path); };
                    dirBar.appendChild(b);
                });
                if (!data.parent && (!data.dirs || !data.dirs.length)) {
                    dirBar.style.display = "none";
                } else {
                    dirBar.style.display = "flex";
                }
            }

            // ---- 视频网格 ----
            function renderGrid(data) {
                grid.innerHTML = "";
                const videos = data.videos || [];
                if (!videos.length) {
                    const empty = document.createElement("div");
                    empty.className = "fxai-vp-empty";
                    empty.innerHTML = "这个目录下没有视频文件<br>" +
                        "<span style='font-size:12px'>点上面的目录进下一层，或者把视频放进这个目录再点刷新</span>";
                    grid.appendChild(empty);
                    return;
                }
                videos.forEach(function (v) {
                    const card = document.createElement("div");
                    card.className = "fxai-vp-card";
                    card.title = v.path;

                    const img = document.createElement("img");
                    img.className = "thumb";
                    img.loading = "lazy";
                    img.src = thumbUrl(v);
                    img.onerror = function () {
                        const ph = document.createElement("div");
                        ph.className = "thumb-ph";
                        ph.textContent = "无预览";
                        if (img.parentNode) img.parentNode.replaceChild(ph, img);
                    };
                    card.appendChild(img);

                    const idxTag = document.createElement("div");
                    idxTag.className = "idx";
                    idxTag.textContent = v.index;
                    card.appendChild(idxTag);

                    const cap = document.createElement("div");
                    cap.className = "cap";
                    const nm = document.createElement("div");
                    nm.className = "nm";
                    nm.textContent = v.name;
                    const sub = document.createElement("div");
                    sub.className = "sub";
                    sub.textContent = fmtSize(v.size);
                    cap.append(nm, sub);
                    card.appendChild(cap);

                    card.onclick = function () {
                        picked = { dir: data.dir, index: v.index, name: v.name, path: v.path };
                        grid.querySelectorAll(".fxai-vp-card").forEach(function (el) {
                            el.classList.remove("on");
                        });
                        card.classList.add("on");
                        btnOk.disabled = false;
                        btnIdx.disabled = false;
                        footLeft.textContent = "已选：" + v.name + "（索引 " + v.index + "）";
                    };
                    card.ondblclick = function () {
                        picked = { dir: data.dir, index: v.index, name: v.name, path: v.path };
                        close(Object.assign({ mode: "path" }, picked));
                    };
                    grid.appendChild(card);
                });
            }

            function load(dir, force) {
                grid.innerHTML = "<div class='fxai-vp-empty'>读取中…</div>";
                browse(dir, force).then(function (data) {
                    if (!data || data.error) {
                        grid.innerHTML = "<div class='fxai-vp-empty'>" +
                            (data && data.error ? data.error : "读取失败") + "</div>";
                        return;
                    }
                    curData = data;
                    curDir = data.dir;
                    pathEl.textContent = data.dir;
                    renderRoots(data.roots);
                    renderDirs(data);
                    renderGrid(data);
                    footLeft.textContent = (data.videos || []).length
                        ? "共 " + data.videos.length + " 个视频 · 点一下选中，双击直接用完整路径"
                        : "";
                });
            }

            // 加一个刷新按钮到左侧
            const btnRefresh = document.createElement("button");
            btnRefresh.className = "fxai-vp-chip";
            btnRefresh.textContent = "🔄 刷新";
            btnRefresh.onclick = function () { load(curDir, true); };
            rootBar.appendChild(btnRefresh);

            document.body.appendChild(mask);
            load(startDir);
        });
    }

    // ---------------------------------------------------------------- 节点 UI
    // 视频分段器V1 的面板也要弹同一个选片窗口，挂到 window 上给它用
    window.FxAiVideoPicker = openVideoPicker;

    function getWidget(node, name) {
        return (node.widgets || []).find(function (w) { return w.name === name; });
    }

    function applySelection(node, sel) {
        const wDir = getWidget(node, W_DIR);
        const wIdx = getWidget(node, W_INDEX);
        if (!wDir || !wIdx) return;

        if (sel.mode === "index") {
            // 目录模式：目录 + 序号
            wDir.value = sel.dir;
            wIdx.value = sel.index;
        } else {
            // 路径模式：直接写视频文件完整路径，视频索引 = -1（表示"按路径取"）
            wDir.value = sel.path;
            wIdx.value = -1;
        }

        // 通知前端（有些版本靠 callback 触发节点重算尺寸/标脏）
        try { if (wDir.callback) wDir.callback(wDir.value); } catch (e) {}
        try { if (wIdx.callback) wIdx.callback(wIdx.value); } catch (e) {}

        app.graph.setDirtyCanvas(true, true);
        renderNode(node);
    }

    function renderNode(node) {
        const ui = node.__fxaiVideoUi;
        if (!ui) return;

        const wDir = getWidget(node, W_DIR);
        const wIdx = getWidget(node, W_INDEX);
        const dir = (wDir && wDir.value ? String(wDir.value) : "").trim();
        const index = wIdx ? parseInt(wIdx.value, 10) || 0 : 0;

        if (!dir) {
            ui.img.removeAttribute("src");
            ui.img.style.display = "none";
            ui.ph.style.display = "flex";
            ui.ph.textContent = "未选择";
            ui.name.textContent = "未选择视频";
            ui.sub.textContent = "点下面按钮从文件夹里挑一个";
            return;
        }

        // 填的是视频文件 → 直接预览这个文件，不再走目录/索引
        if (isVideoFile(dir)) {
            ui.ph.style.display = "none";
            ui.img.style.display = "block";
            ui.img.src = apiUrl("/fxai/video/thumb?path=" + encodeURIComponent(dir));
            ui.img.onerror = function () {
                ui.img.style.display = "none";
                ui.ph.style.display = "flex";
                ui.ph.textContent = "无预览";
            };
            ui.name.textContent = baseName(dir);
            ui.sub.textContent = "按路径读取 · 索引不参与";
            return;
        }

        ui.name.textContent = "读取中…";
        ui.sub.textContent = dir;

        browse(dir).then(function (data) {
            if (!data || data.error) {
                ui.name.textContent = "目录读取失败";
                ui.sub.textContent = data && data.error ? data.error : dir;
                return;
            }
            const videos = data.videos || [];
            // 索引 -1 = 没指定序号，节点会取第一个；预览也跟着显示第一个
            const wantIdx = index >= 0 ? index : 0;
            const v = videos.find(function (x) { return x.index === wantIdx; });
            if (!v) {
                ui.img.style.display = "none";
                ui.ph.style.display = "flex";
                ui.ph.textContent = "索引越界";
                ui.name.textContent = "索引 " + index + " 超出范围";
                ui.sub.textContent = "该目录共 " + videos.length + " 个视频";
                return;
            }
            ui.ph.style.display = "none";
            ui.img.style.display = "block";
            ui.img.src = thumbUrl(v);
            ui.name.textContent = v.name;
            ui.sub.textContent = "序号 " + v.index + " · " + fmtSize(v.size) +
                (index < 0 ? " ·（索引 -1，默认取第一个）" : "");
        });
    }

    function buildUi(node) {
        if (node.__fxaiVideoUi) return;      // onNodeCreated 和 onConfigure 都可能触发，只建一次
        injectStyle();

        const wrap = document.createElement("div");
        wrap.className = "fxai-vload-wrap";

        const row = document.createElement("div");
        row.className = "fxai-vload-row";

        const thumbBox = document.createElement("div");
        thumbBox.className = "fxai-vload-thumb";
        const ph = document.createElement("span");
        ph.textContent = "未选择";
        const img = document.createElement("img");
        img.style.display = "none";
        thumbBox.append(ph, img);

        const meta = document.createElement("div");
        meta.className = "fxai-vload-meta";
        const name = document.createElement("div");
        name.className = "fxai-vload-name";
        name.textContent = "未选择视频";
        const sub = document.createElement("div");
        sub.className = "fxai-vload-sub";
        meta.append(name, sub);

        row.append(thumbBox, meta);

        const btn = document.createElement("button");
        btn.className = "fxai-vload-btn";
        btn.textContent = "📁 选择视频";

        wrap.append(row, btn);

        const domWidget = node.addDOMWidget("fxai_video_picker", "custom", wrap);
        domWidget.serialize = false;
        domWidget.computeSize = function (width) {
            return [width || node.size[0] || 300, 132];
        };

        node.__fxaiVideoUi = { img: img, ph: ph, name: name, sub: sub };

        btn.onclick = function () {
            const wDir = getWidget(node, W_DIR);
            let startDir = wDir && wDir.value ? String(wDir.value).trim() : "";
            // 当前填的是文件 → 弹窗定位到它所在的那个目录
            if (isVideoFile(startDir)) {
                startDir = startDir.replace(/[\\/][^\\/]*$/, "");
            }
            openVideoPicker(startDir).then(function (sel) {
                if (sel) applySelection(node, sel);
            });
        };

        // 手动改索引 / 改目录也要跟着刷新
        [W_DIR, W_INDEX].forEach(function (wn) {
            const w = getWidget(node, wn);
            if (!w) return;
            const orig = w.callback;
            w.callback = function () {
                const r = orig ? orig.apply(this, arguments) : undefined;
                renderNode(node);
                return r;
            };
        });

        renderNode(node);
    }

    app.registerExtension({
        name: "FxAi.VideoLoadPicker",
        beforeRegisterNodeDef: function (nodeType, nodeData) {
            if (nodeData.name !== NODE_TYPE) return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;
                // 等 widget 都建好（有些版本 onNodeCreated 里 widgets 还没齐）
                setTimeout(function () { buildUi(node); }, 0);
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                const node = this;
                setTimeout(function () {
                    if (!node.__fxaiVideoUi) buildUi(node);
                    else renderNode(node);
                }, 50);
                return r;
            };
        }
    });

})();
