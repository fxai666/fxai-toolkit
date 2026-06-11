import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

var TARGET_CLASS = "FxAiImageManagerV2";
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

// 修复批量上传：改用并行上传（也可保持串行但简化逻辑），移除进度回调
function uploadFiles(files, subdir) {
    // 并行上传所有文件（效率更高，也避免串行循环的Promise陷阱）
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

    // 等待所有文件上传完成
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

    var subdirWidget = null;
    for (var i = 0; i < node.widgets.length; i++) {
        if (node.widgets[i].name === "目录") {
            subdirWidget = node.widgets[i];
            break;
        }
    }

    if (!subdirWidget) {
        console.error("未找到子目录控件");
        return;
    }

    var container = document.createElement("div");
    container.style.padding = "8px";
    container.style.border = "1px solid #555";
    container.style.borderRadius = "4px";
    container.style.minWidth = "300px";

    var domWidget = node.addDOMWidget("image_ui", "image_ui", container);
    domWidget.computeSize = function() {
        return [790, 530];
    };

    var dropArea = document.createElement("div");
    dropArea.style.padding = "12px";
    dropArea.style.marginBottom = "8px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.textContent = "📥 拖拽图片到这里上传（支持多图）";
    container.appendChild(dropArea);

    // 按钮区域 提前定义，修复之前 childNodes 硬编码取值BUG
    var btnDiv = document.createElement("div");
    btnDiv.style.display = "flex";
    btnDiv.style.gap = "8px";
    btnDiv.style.marginBottom = "8px";
    container.appendChild(btnDiv);
    
    var selectBtn = document.createElement("button");
    selectBtn.textContent = "📁 选择目录";
    var uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📤 选择图片上传";
    var refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    var applyBtn = document.createElement("button");
    applyBtn.textContent = "✅ 确认操作";
    btnDiv.appendChild(selectBtn);
    btnDiv.appendChild(uploadBtn);
    btnDiv.appendChild(refreshBtn);
    btnDiv.appendChild(applyBtn);

    var listDiv = document.createElement("div");
    listDiv.style.display = "flex";
    listDiv.style.flexWrap = "wrap";
    listDiv.style.maxHeight = "400px";
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

        // 移除进度回调，简化上传逻辑
        uploadFiles(files, subdirWidget.value)
        .then(function() {
            return updateList();
        })
        .catch(function(err) {
            alert("上传失败: " + err.message);
        })
        .finally(function() {
            uploadBtn.textContent = originalText;
            uploadBtn.disabled = false;
        });
    });

    // 刷新列表
    function updateList() {
        return new Promise(function(resolve) {
            while (listDiv.firstChild) {
                listDiv.removeChild(listDiv.firstChild);
            }
            if (sortable) {
                sortable.destroy();
                sortable = null;
            }

            fetchFileList(subdirWidget.value).then(function(files) {
                for (var j = 0; j < files.length; j++) {
                    var file = files[j];
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
                    var imgUrl = api.apiURL("/fxai/image/v2/preview?subdir=" + encodeURIComponent(subdirWidget.value) + "&filename=" + encodeURIComponent(file) + "&t=" + Date.now());
                    img.src = imgUrl;
                    img.style.width = "100%";
                    img.style.height = "100%";
                    img.style.objectFit = "cover";
                    img.style.display = "block";
                    img.onclick = function() {
                        window.fxaiOpenImage(this.src);
                    };

                    var nameSpan = document.createElement("div");
                    nameSpan.textContent = file;
                    nameSpan.style.position = "absolute";
                    nameSpan.style.bottom = "0";
                    nameSpan.style.left = "0";
                    nameSpan.style.right = "0";
                    nameSpan.backgroundColor = "rgba(0,0,0,0.6)";
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

                    // ====================== 修复 BUG 核心代码 ======================
                    (function(currentItem, currentFile){
                        delBtn.onclick = function(e) {
                            e.stopPropagation();
                            deleteImage(subdirWidget.value, currentFile)
                            .then(function(data) {
                                if (data.success) {
                                    // 直接删除当前DOM，不刷新整个列表
                                    currentItem.remove();
                                } else {
                                    alert("删除失败：" + (data.error || "未知错误"));
                                }
                            })
                            .catch(function(err) {
                                alert("删除失败：" + err.message);
                            });
                        };
                    })(item, file);
                    // =================================================================

                    item.appendChild(img);
                    item.appendChild(nameSpan);
                    item.appendChild(delBtn);
                    listDiv.appendChild(item);
                }

                if (window.Sortable) {
                    sortable = new Sortable(listDiv, {
                        animation: 150,
                        handle: ".image-item",
                        ghostClass: "sortable-ghost"
                    });
                } else {
                    var script = document.createElement("script");
                    script.src = "./Sortable.min.js";
                    script.onload = function() {
                        sortable = new Sortable(listDiv, {
                            animation: 150,
                            handle: ".image-item",
                            ghostClass: "sortable-ghost"
                        });
                    };
                    document.head.appendChild(script);
                }
                resolve();
            })
            .catch(function(err) {
                console.error("更新列表失败:", err);
                resolve();
            });
        });
    }

    selectBtn.onclick=function(){
        FxAiFolderSelector("image").then(result => {
            if(result!==undefined)
            {
                subdirWidget.value = result;
                updateList();
            }
        });
    }

    // 选择文件上传按钮
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

            // 移除进度回调，简化上传逻辑
            uploadFiles(files, subdirWidget.value)
            .then(function() {
                return updateList();
            })
            .catch(function(err) {
                alert("上传失败: " + err.message);
            })
            .finally(function() {
                uploadBtn.textContent = originalText;
                uploadBtn.disabled = false;
            });
        };
        input.click();
    };

    // 刷新按钮
    refreshBtn.onclick = function() {
        updateList();
    };

    // 应用排序按钮
    applyBtn.onclick = function() {
        var items = listDiv.querySelectorAll(".image-item");
        var ordered = [];
        for (var k = 0; k < items.length; k++) {
            ordered.push(items[k].dataset.filename);
        }
        applyChanges(subdirWidget.value, ordered)
        .then(function() {
            return updateList();
        })
        .catch(function(err) {
            alert("应用失败: " + err.message);
        });
    };

    // 目录切换监听
    if (!updatedListBindFlag.has(subdirWidget)) {
        var origCallback = subdirWidget.callback;
        subdirWidget.callback = function(v) {
            if (origCallback) {
                origCallback.call(this, v);
            }
            updateList();
        };
        updatedListBindFlag.set(subdirWidget, true);
    }

    setTimeout(function() {
        updateList();
    }, 0);
}

app.registerExtension({
    name: "FxAiImageManagerV2",
    nodeCreated:function(node) {
        setTimeout(function() {
            if (node.comfyClass === TARGET_CLASS) {
                addUI(node);
            }
        }, 200);
    }
});