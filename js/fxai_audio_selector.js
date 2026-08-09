import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { AUDIO_CATEGORY_CONFIG } from "./fxai_category_config.js";
// ==============================================
// 工具：获取文件列表（复用音频管理器接口）
// ==============================================
function fetchFileList(subdir) {
    return new Promise(function (resolve) {
        var url = api.apiURL("/fxai/audio/list?subdir=" + encodeURIComponent(subdir));
        fetch(url)
            .then(function (resp) {
                return resp.ok ? resp.json() : { files: [] };
            })
            .then(function (data) {
                resolve(data.files || []);
            })
            .catch(function () {
                resolve([]);
            });
    });
}

// 上传音频到指定分类目录（并行多文件）
function uploadFiles(files, subdir) {
    var uploadPromises = [];
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var formData = new FormData();
        formData.append("audio", file, file.name);
        formData.append("subdir", subdir);
        uploadPromises.push(fetch(api.apiURL("/fxai/audio/upload"), {
            method: "POST",
            body: formData
        }).then(function (response) {
            if (!response.ok) {
                throw new Error("上传失败: " + response.status);
            }
        }));
    }
    return Promise.all(uploadPromises);
}

// ==============================================
// 全局弹窗 - 音频单选选择器
// ==============================================
window.FxAiAudioSelector = function (initSelectPath) {
    return new Promise(function (resolve) {
        // 单选：只存单个路径，初始值
        let selectedPath = initSelectPath?.trim() || "";

        // 遮罩
        var mask = document.createElement("div");
        mask.style.cssText = `
            position: fixed; inset: 0; z-index: 99999;
            background: rgba(0,0,0,0.85);
            display: flex; align-items: center; justify-content: center;
        `;
        document.body.appendChild(mask);
        // 弹窗
        var modal = document.createElement("div");
        modal.style.cssText = `
            width: 1100px; max-width: 95vw;
            height: 750px; max-height: 90vh;
            background: #222; border-radius: 10px;
            padding: 10px; box-sizing: border-box;
            display: flex; flex-direction: column; gap: 16px;
        `;
        mask.appendChild(modal);
        // 标题
        var title = document.createElement("div");
        title.textContent = "🎵 选择音频";
        title.style.cssText = "font-size: 18px; color: #fff; font-weight: bold;";
        modal.appendChild(title);

        let currentSubdir = Object.values(AUDIO_CATEGORY_CONFIG)[0];
        // 标签栏
        var tabBar = document.createElement("div");
        tabBar.style.cssText = "display: flex; gap: 5px; flex-wrap: wrap; margin-left:5px";
        modal.appendChild(tabBar);

        function setActiveTab(tab) {
            tabBar.querySelectorAll("button").forEach(function (t) {
                t.style.background = "#333"; t.style.color = "#ccc";
            });
            tab.style.background = "#4a8fff"; tab.style.color = "#fff";
        }
        Object.entries(AUDIO_CATEGORY_CONFIG).forEach(function (entry) {
            var label = entry[0];
            var dir = entry[1];
            var btn = document.createElement("button");
            btn.textContent = label;
            btn.style.cssText = `
                padding: 3px 6px; border: none; border-radius: 4px;
                background: #333; color: #ccc; cursor: pointer;
            `;
            btn.onclick = function () {
                currentSubdir = dir;
                setActiveTab(btn);
                renderList();
            };
            tabBar.appendChild(btn);
            if (dir === currentSubdir) setActiveTab(btn);
        });

        // 音频展示区：弹性自动换行布局
        var listContainer = document.createElement("div");
        listContainer.style.cssText = `
            flex: 1; overflow-y: auto;
            display: flex; flex-wrap: wrap; gap: 8px;
            padding:4px; align-content:flex-start;
        `;
        modal.appendChild(listContainer);

        // 底部按钮栏
        var bottomBar = document.createElement("div");
        bottomBar.style.cssText = "display: flex; justify-content: space-between; align-items: center; gap: 10px;";
        modal.appendChild(bottomBar);

        var leftBar = document.createElement("div");
        leftBar.style.cssText = "display: flex; gap: 8px; align-items: center;";
        bottomBar.appendChild(leftBar);

        var rightBar = document.createElement("div");
        rightBar.style.cssText = "display: flex; gap: 10px; align-items: center;";
        bottomBar.appendChild(rightBar);

        var btnUpload = document.createElement("button");
        btnUpload.textContent = "📤 上传音频";
        btnUpload.style.cssText = "padding: 6px 12px; border: none; border-radius: 4px; background: #2a9d3f; color: #fff; cursor: pointer;";
        btnUpload.title = "上传音频到当前分类目录，上传成功后列表自动刷新";
        leftBar.appendChild(btnUpload);

        var btnCancel = document.createElement("button");
        btnCancel.textContent = "取消";
        var btnConfirm = document.createElement("button");
        btnConfirm.textContent = "✅ 确认选择";
        btnConfirm.style.background = "#4a8fff"; btnConfirm.style.color = "#fff";
        rightBar.append(btnCancel, btnConfirm);

        btnUpload.onclick = function () {
            var input = document.createElement("input");
            input.type = "file";
            input.multiple = true;
            input.accept = "audio/*";
            input.onchange = function () {
                if (!input.files.length) return;
                var files = Array.prototype.slice.call(input.files);
                var originalText = btnUpload.textContent;
                btnUpload.textContent = "上传中...";
                btnUpload.disabled = true;
                uploadFiles(files, currentSubdir)
                    .then(function () {
                        renderList();
                        btnUpload.textContent = "✅ 上传成功";
                        setTimeout(function () {
                            btnUpload.textContent = originalText;
                            btnUpload.disabled = false;
                        }, 1500);
                    })
                    .catch(function (err) {
                        alert("上传失败: " + err.message);
                        btnUpload.textContent = originalText;
                        btnUpload.disabled = false;
                    });
            };
            input.click();
        };

        function close() {
            document.body.removeChild(mask);
        }
        btnCancel.onclick = function () { resolve(); close(); };
        btnConfirm.onclick = function () {
            resolve(selectedPath);
            close();
        };
        mask.onclick = function (e) {
            if (e.target === mask) close();
        };

        // 刷新全部音频项选中边框（单选）
        function refreshAllItemBorder() {
            var allItems = listContainer.querySelectorAll(".audio-asset-item");
            allItems.forEach(function (el) {
                var path = el.dataset.path;
                if (selectedPath === path) {
                    el.style.border = "3px solid #4a8fff";
                } else {
                    el.style.border = "3px solid transparent";
                }
            });
        }

        // 渲染音频列表（复用音频管理器预览逻辑）
        function renderList() {
            listContainer.innerHTML = "";
            fetchFileList(currentSubdir).then(function (files) {
                files.forEach(function (filename) {
                    // 音频完整路径：分类目录/文件名
                    var realPath = currentSubdir + "/" + filename;
                    var isSelected = selectedPath === realPath;

                    // 音频预览地址，和音频管理器完全统一
                    var audioSrc = api.apiURL("/fxai/audio/preview?subdir="
                        + encodeURIComponent(currentSubdir)
                        + "&filename=" + encodeURIComponent(filename));

                    var item = document.createElement("div");
                    item.className = "audio-asset-item";
                    item.dataset.path = realPath;
                    item.style.cssText = `
                        width:260px;height:90px;position:relative; border-radius:6px;
                        overflow:hidden; cursor:pointer; flex-shrink:0;
                        border:3px solid ${isSelected ? "#4a8fff" : "transparent"};
                        background:#1a1a1a; padding:6px;
                        display:flex; flex-direction:column; align-items:center; justify-content:center;
                    `;

                    // 音频播放控件，和音频管理器UI保持一致
                    var audio = document.createElement("audio");
                    audio.controls = true;
                    audio.style.width = "100%";
                    audio.style.marginBottom = "4px";
                    audio.src = audioSrc;

                    // 文件名文本
                    var nameSpan = document.createElement("div");
                    nameSpan.textContent = filename;
                    nameSpan.style.cssText = `
                        color:#fff; font-size:11px; text-align:center;
                        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                        width:100%;
                    `;

                    item.appendChild(audio);
                    item.appendChild(nameSpan);
                    listContainer.appendChild(item);

                    // 点击切换单选：只能选中一个
                    item.onclick = function () {
                        if (selectedPath === realPath) {
                            // 再次点击取消选中
                            selectedPath = "";
                        } else {
                            // 选中当前，覆盖原有选择
                            selectedPath = realPath;
                        }
                        refreshAllItemBorder();
                    };
                });
            });
        }

        // 初始渲染
        renderList();
    });
};
// 注册扩展
app.registerExtension({
    name: "FxAiAudioSelector"
});