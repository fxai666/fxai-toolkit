import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

var sortable = null;
var updatedListBindFlag = new WeakMap();

// ==============================================
// 🔥 核心配置：这里定义你的标签和对应的文件夹名称
// ==============================================
const CATEGORY_CONFIG = {
    "角色": "avatar",
    "套装": "clothes",
    "首饰": "jewelry",
    "上衣": "tops",
    "胸罩": "bra",
    "裤子": "pants",
    "裙子": "skirts",
    "裙子": "skirts",
    "内裤": "underpants",
    "鞋袜": "shoessocks",
    "姿势": "pose",
    "装备": "equipment",
    "场景": "scene",
    "家具": "furniture",
    "宠物": "pet",
    "座驾": "vehicle",
    "产品": "products",
    "素材": "sucai",
    "其他": "other"
};

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
    container.style.minWidth = "300px";
    container.style.boxSizing = "border-box";

    var domWidget = node.addDOMWidget("image_ui", "image_ui", container);
    domWidget.computeSize = function() {
        return [790, 530];
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
    dropArea.textContent = "📥 拖拽图片到这里上传（支持多图）";
    container.appendChild(dropArea);

    // 按钮区域
    var btnDiv = document.createElement("div");
    btnDiv.style.display = "flex";
    btnDiv.style.gap = "8px";
    btnDiv.style.marginBottom = "8px";
    container.appendChild(btnDiv);

    var uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📤 选择图片上传";
    var refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    var applyBtn = document.createElement("button");
    applyBtn.textContent = "✅ 确认操作";
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

    // 应用排序
    applyBtn.onclick = function() {
        var items = listDiv.querySelectorAll(".image-item");
        var ordered = Array.from(items).map(i => i.dataset.filename);
        applyChanges(currentSubdir, ordered)
        .then(() => updateList())
        .catch(err => alert("应用失败: " + err.message));
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