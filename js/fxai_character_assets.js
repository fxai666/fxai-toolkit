import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { CATEGORY_CONFIG } from "./fxai_category_config.js";

var sortable = null;
var updatedListBindFlag = new WeakMap();

function fetchFileList(subdir) {
    return new Promise(function(resolve, reject) {
        var url = api.apiURL("/fxai/image/v2/list?subdir=" + encodeURIComponent(subdir));
        fetch(url)
        .then(function(resp) {
            if (!resp.ok) {
                return resolve([]);
            }
            return resp.json();
        })
        .then(function(data) {
            resolve(data.files);
        })
        .catch(function() {
            resolve([]);
        });
    });
}

function uploadFiles(files, subdir) {
    var uploadPromises = [];
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var formData = new FormData();
        formData.append("image", file, file.name);
        formData.append("subdir", subdir);

        var promise = fetch(api.apiURL("/fxai/image/v2/upload"), {
            method: "POST",
            body: formData
        }).then(function(response) {
            if (!response.ok) {
                throw new Error("上传失败: " + response.status);
            }
        });
        uploadPromises.push(promise);
    }
    return Promise.all(uploadPromises);
}

function applyChanges(subdir, orderedFilenames) {
    return new Promise(function(resolve, reject) {
        fetch(api.apiURL("/fxai/image/v2/apply"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ subdir: subdir, ordered_filenames: orderedFilenames })
        })
        .then(function(resp) {
            if (!resp.ok) {
                throw new Error("应用更改失败");
            }
            return resp.json();
        })
        .then(function(data) {
            resolve(data.files);
        })
        .catch(function(err) {
            reject(err);
        });
    });
}

// 自绘确认对话框（不可逆操作专用）：返回 Promise<boolean>，点确定才 resolve(true)
function showConfirmDialog(options) {
    var opts = options || {};
    return new Promise(function(resolve) {
        var mask = document.createElement("div");
        mask.style.cssText = "position:fixed;top:0;left:0;right:0;bottom:0;" +
            "background:rgba(0,0,0,0.65);z-index:100000;" +
            "display:flex;align-items:center;justify-content:center;";

        var card = document.createElement("div");
        card.style.cssText = "width:480px;max-width:calc(100vw - 40px);box-sizing:border-box;" +
            "background:#1f2227;border:1px solid #4a8fff;border-radius:10px;" +
            "box-shadow:0 14px 44px rgba(0,0,0,0.65);padding:20px 22px;" +
            "color:#e8e8e8;font-size:13px;line-height:1.75;";

        var titleEl = document.createElement("div");
        titleEl.textContent = opts.title || "请确认";
        titleEl.style.cssText = "font-size:16px;font-weight:700;margin-bottom:12px;color:#ffb64d;";

        var bodyEl = document.createElement("div");
        bodyEl.textContent = opts.message || "";
        bodyEl.style.cssText = "white-space:pre-line;margin-bottom:18px;color:#d6d6d6;";

        var btnRow = document.createElement("div");
        btnRow.style.cssText = "display:flex;justify-content:flex-end;gap:10px;";

        var cancelBtn = document.createElement("button");
        cancelBtn.textContent = opts.cancelText || "取消";
        cancelBtn.style.cssText = "padding:6px 14px;border-radius:6px;border:1px solid #6b6b6b;" +
            "background:#3a3f47;color:#e0e0e0;cursor:pointer;font-size:13px;";

        var okBtn = document.createElement("button");
        okBtn.textContent = opts.confirmText || "确定";
        okBtn.style.cssText = "padding:6px 14px;border-radius:6px;border:1px solid #d9534f;" +
            "background:#c0392b;color:#fff;cursor:pointer;font-size:13px;font-weight:600;";

        function close(result) {
            document.removeEventListener("keydown", onKey, true);
            if (mask.parentNode) mask.parentNode.removeChild(mask);
            resolve(result);
        }
        function onKey(e) {
            if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                close(false);
            }
        }

        cancelBtn.onclick = function() { close(false); };
        okBtn.onclick = function() { close(true); };
        document.addEventListener("keydown", onKey, true);

        btnRow.appendChild(cancelBtn);
        btnRow.appendChild(okBtn);
        card.appendChild(titleEl);
        card.appendChild(bodyEl);
        card.appendChild(btnRow);
        mask.appendChild(card);
        document.body.appendChild(mask);
        okBtn.focus();
    });
}

function deleteImage(subdir, filename) {
    return new Promise(function(resolve, reject) {
        var url = api.apiURL("/fxai/image/v2/delete?subdir=" + encodeURIComponent(subdir) + "&filename=" + encodeURIComponent(filename));
        fetch(url, {
            method: "DELETE"
        })
        .then(function(resp) {
            if (!resp.ok) {
                return resp.json()
                .catch(function() {
                    return { error: "删除失败" };
                })
                .then(function(errData) {
                    throw new Error(errData.error || "删除失败: " + resp.status);
                });
            }
            return resp.json();
        })
        .then(function(data) {
            resolve(data);
        })
        .catch(function(err) {
            reject(err);
        });
    });
}

function preventDefaultDragDrop() {
    document.addEventListener('dragover', function(e) { e.preventDefault(); });
    document.addEventListener('drop', function(e) { e.preventDefault(); });
    document.addEventListener('dragenter', function(e) { e.preventDefault(); });
    document.addEventListener('dragleave', function(e) { e.preventDefault(); });
}

function addUI(node) {
    if (node._uiAdded) return;
    node._uiAdded = true;
    preventDefaultDragDrop();

    // 当前选中的目录（默认第一个）
    let currentSubdir = Object.values(CATEGORY_CONFIG)[0];

    var container = document.createElement("div");
    container.style.padding = "8px";
    container.style.border = "1px solid #555";
    container.style.borderRadius = "4px";
    container.style.boxSizing = "border-box";

    var domWidget = node.addDOMWidget("image_ui", "image_ui", container);
    domWidget.computeSize = function() {
        return [790, 560];
    };

    // ==============================================
    // 🔥 标签栏（自动从配置生成）
    // ==============================================
    const tabBar = document.createElement("div");
    tabBar.style.display = "flex";
    tabBar.style.gap = "6px";
    tabBar.style.marginBottom = "10px";
    tabBar.style.flexWrap = "wrap";
    container.appendChild(tabBar);

    function setActiveTab(tabEl) {
        tabBar.querySelectorAll(".tab-btn").forEach(t => {
            t.style.background = "#333";
            t.style.color = "#ccc";
        });
        tabEl.style.background = "#4a8fff";
        tabEl.style.color = "#fff";
    }

    // 创建所有标签
    Object.entries(CATEGORY_CONFIG).forEach(([label, dirName]) => {
        const tab = document.createElement("button");
        tab.className = "tab-btn";
        tab.textContent = label;
        tab.style.padding = "3px 5px";
        tab.style.borderRadius = "4px";
        tab.style.border = "none";
        tab.style.cursor = "pointer";
        tab.onclick = () => {
            currentSubdir = dirName;
            setActiveTab(tab);
            updateList();
        };
        tabBar.appendChild(tab);
        if (dirName === currentSubdir) setActiveTab(tab);
    });

    // 拖拽上传区域
    var dropArea = document.createElement("div");
    dropArea.style.padding = "12px";
    dropArea.style.marginBottom = "8px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.textContent = "📥 拖拽图片到这里上传（支持多图，按上传时间自动命名）";
    container.appendChild(dropArea);

    // 按钮区域
    var btnDiv = document.createElement("div");
    btnDiv.style.display = "flex";
    btnDiv.style.gap = "8px";
    btnDiv.style.marginBottom = "8px";
    container.appendChild(btnDiv);

    var uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📤 选择图片上传";
    uploadBtn.title = "上传后自动以上传时刻命名（yyyyMMddHHmmss，如 20261123063012.png）";
    var refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    var applyBtn = document.createElement("button");
    applyBtn.textContent = "🔢 按顺序重命名";
    applyBtn.title = "按当前排列顺序重命名本目录文件（不可逆）";
    btnDiv.appendChild(uploadBtn);
    btnDiv.appendChild(refreshBtn);
    btnDiv.appendChild(applyBtn);

    // 图片列表
    var listDiv = document.createElement("div");
    listDiv.style.display = "flex";
    listDiv.style.flexWrap = "wrap";
    listDiv.style.maxHeight = "380px";
    listDiv.style.overflowY = "auto";
    listDiv.style.padding = "4px";
    listDiv.style.border = "1px solid #666";
    container.appendChild(listDiv);

    // 拖拽上传
    dropArea.addEventListener("dragover", function(e) {
        e.preventDefault();
        dropArea.style.borderColor = "#fff";
    });
    dropArea.addEventListener("dragleave", function() {
        dropArea.style.borderColor = "#777";
    });
    dropArea.addEventListener("drop", function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.style.borderColor = "#777";
        var files = Array.prototype.slice.call(e.dataTransfer.files);
        if (!files.length) return;

        var originalText = uploadBtn.textContent;
        uploadBtn.textContent = "上传中...";
        uploadBtn.disabled = true;

        uploadFiles(files, currentSubdir)
        .then(() => updateList())
        .catch(err => alert("上传失败: " + err.message))
        .finally(() => {
            uploadBtn.textContent = originalText;
            uploadBtn.disabled = false;
        });
    });

    // 刷新列表
    function updateList() {
        return new Promise(function(resolve) {
            while (listDiv.firstChild) listDiv.removeChild(listDiv.firstChild);
            if (sortable) {
                sortable.destroy();
                sortable = null;
            }

            fetchFileList(currentSubdir).then(function(files) {
                files.forEach(file => {
                    var item = document.createElement("div");
                    item.className = "image-item";
                    item.dataset.filename = file;
                    item.style.position = "relative";
                    item.style.width = "120px";
                    item.style.height = "120px";
                    item.style.margin = "4px";
                    item.style.cursor = "grab";
                    item.style.backgroundColor = "#222";
                    item.style.borderRadius = "6px";
                    item.style.overflow = "hidden";

                    var img = document.createElement("img");
                    var imgUrl = api.apiURL("/fxai/image/v2/preview?subdir=" + encodeURIComponent(currentSubdir) + "&filename=" + encodeURIComponent(file) + "&t=" + Date.now());
                    img.src = imgUrl;
                    img.style.width = "100%";
                    img.style.height = "100%";
                    img.style.objectFit = "cover";
                    img.style.display = "block";
                    img.onclick = () => window.open(img.src, "_blank");

                    var nameSpan = document.createElement("div");
                    nameSpan.textContent = file;
                    nameSpan.style.position = "absolute";
                    nameSpan.style.bottom = "0";
                    nameSpan.style.left = "0";
                    nameSpan.style.right = "0";
                    nameSpan.style.backgroundColor = "rgba(0,0,0,0.6)";
                    nameSpan.style.color = "white";
                    nameSpan.style.fontSize = "10px";
                    nameSpan.style.textAlign = "center";
                    nameSpan.style.padding = "2px";
                    nameSpan.style.whiteSpace = "nowrap";
                    nameSpan.style.overflow = "hidden";
                    nameSpan.style.textOverflow = "ellipsis";

                    var delBtn = document.createElement("button");
                    delBtn.textContent = "✖";
                    delBtn.style.position = "absolute";
                    delBtn.style.top = "2px";
                    delBtn.style.right = "2px";
                    delBtn.style.backgroundColor = "rgba(0,0,0,0.6)";
                    delBtn.style.color = "white";
                    delBtn.style.border = "none";
                    delBtn.style.borderRadius = "50%";
                    delBtn.style.width = "20px";
                    delBtn.style.height = "20px";
                    delBtn.style.cursor = "pointer";

                    (function(currentItem, currentFile){
                        delBtn.onclick = function(e) {
                            e.stopPropagation();
                            deleteImage(currentSubdir, currentFile)
                            .then(data => {
                                if (data.success) currentItem.remove();
                                else alert("删除失败：" + (data.error || "未知错误"));
                            })
                            .catch(err => alert("删除失败：" + err.message));
                        };
                    })(item, file);

                    item.appendChild(img);
                    item.appendChild(nameSpan);
                    item.appendChild(delBtn);
                    listDiv.appendChild(item);
                });

                if (window.Sortable) {
                    sortable = new Sortable(listDiv, { animation:150, handle:".image-item" });
                } else {
                    var script = document.createElement("script");
                    script.src = "./Sortable.min.js";
                    script.onload = () => {
                        sortable = new Sortable(listDiv, { animation:150, handle:".image-item" });
                    };
                    document.head.appendChild(script);
                }
                resolve();
            }).catch(err => { console.error(err); resolve(); });
        });
    }

    // 选择文件上传
    uploadBtn.onclick = function() {
        var input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = "image/*";
        input.onchange = function() {
            if (!input.files.length) return;
            var files = Array.prototype.slice.call(input.files);
            var originalText = uploadBtn.textContent;
            uploadBtn.textContent = "上传中...";
            uploadBtn.disabled = true;

            uploadFiles(files, currentSubdir)
            .then(() => updateList())
            .catch(err => alert("上传失败: " + err.message))
            .finally(() => {
                uploadBtn.textContent = originalText;
                uploadBtn.disabled = false;
            });
        };
        input.click();
    };

    refreshBtn.onclick = updateList;

    // 按顺序重命名（不可逆，操作前必须二次确认）
    applyBtn.onclick = function() {
        var items = listDiv.querySelectorAll(".image-item");
        if (!items.length) {
            alert("当前目录没有文件，无需重命名。");
            return;
        }

        var targetSubdir = currentSubdir;
        var ordered = Array.from(items).map(i => i.dataset.filename);

        var dirLabel = targetSubdir;
        Object.entries(CATEGORY_CONFIG).forEach(([label, dir]) => {
            if (dir === targetSubdir) dirLabel = label;
        });

        showConfirmDialog({
            title: "⚠️ 确认按顺序重命名？",
            message: "此操作将对当前目录「" + dirLabel + "」（" + targetSubdir + "）的 " + ordered.length + " 个文件，\n" +
                     "按当前排列顺序重新命名（依次编号 000、001、002…）。\n\n" +
                     "是否确认操作？\n\n" +
                     "此操作不可逆，原始文件名将不再保留。\n" +
                     "如果要保持当前目录下的原始文件名，请不要点「确定重命名」。",
            confirmText: "确定重命名",
            cancelText: "取消，保留原名"
        }).then(function(ok) {
            if (!ok) return;

            var originalText = applyBtn.textContent;
            applyBtn.textContent = "重命名中...";
            applyBtn.disabled = true;

            applyChanges(targetSubdir, ordered)
            .then(() => updateList())
            .catch(err => alert("重命名失败，原文件名未改变：" + err.message))
            .finally(() => {
                applyBtn.textContent = originalText;
                applyBtn.disabled = false;
            });
        });
    };

    setTimeout(() => updateList(), 0);
}

app.registerExtension({
    name: "FxAiCharacterAssets",
    nodeCreated:function(node) {
        setTimeout(() => {
            if (node.comfyClass === "FxAiCharacterAssets") addUI(node);
        }, 200);
    }
});