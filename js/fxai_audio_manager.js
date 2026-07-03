import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

var TARGET_CLASS = "FxAiAudioManager";
var sortable = null;

// 获取文件列表
function fetchFileList(subdir, callback) {
    var url = api.apiURL("/fxai/audio/list?subdir=" + encodeURIComponent(subdir));
    fetch(url)
        .then(function(resp) {
            if (!resp.ok) {
                callback(null, []);
                return;
            }
            return resp.json();
        })
        .then(function(data) {
            callback(null, data.files);
        })
        .catch(function() {
            callback(null, []);
        });
}

// 音频删除接口
function deleteAudio(subdir, filename, callback) {
    fetch(api.apiURL("/fxai/audio/delete?subdir=" + encodeURIComponent(subdir) + "&filename=" + encodeURIComponent(filename)))
        .then(function(resp) {
            if (!resp.ok) throw new Error("删除失败");
            return resp.json();
        })
        .then(function(data) {
            callback(null, data);
        })
        .catch(function(err) {
            callback(err);
        });
}

// 音频逐文件上传（保留原串行逻辑）
function uploadFiles(files, subdir, onProgress, callback) {
    var i = 0;
    function doUpload() {
        if (i >= files.length) {
            callback(null);
            return;
        }
        var file = files[i];
        var formData = new FormData();
        formData.append("audio", file, file.name);
        formData.append("subdir", subdir);

        fetch(api.apiURL("/fxai/audio/upload"), {
            method: "POST",
            body: formData
        })
        .then(function(response) {
            if (!response.ok) {
                throw new Error("上传失败: " + response.status);
            }
            if (onProgress) {
                onProgress(i, 1);
            }
            i++;
            doUpload();
        })
        .catch(function(err) {
            callback(new Error("文件 " + file.name + " 上传失败: " + err.message));
        });
    }
    doUpload();
}

// 应用排序
function applyChanges(subdir, orderedFilenames, callback) {
    fetch(api.apiURL("/fxai/audio/apply"), {
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
        callback(null, data.files);
    })
    .catch(function(err) {
        callback(err);
    });
}

// 全局阻止拖拽默认行为
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

    // 查找 目录 widget（等价原 find）
    var subdirWidget = null;
    for (var wIdx = 0; wIdx < node.widgets.length; wIdx++) {
        if (node.widgets[wIdx].name === "目录") {
            subdirWidget = node.widgets[wIdx];
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
    var domWidget = node.addDOMWidget("audio_ui", "audio_ui", container);
    domWidget.computeSize = function() {
        return [500, 410];
    };

    // 拖拽上传区域
    var dropArea = document.createElement("div");
    dropArea.style.padding = "12px";
    dropArea.style.marginBottom = "8px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.textContent = "📥 拖拽音频到这里上传（支持多文件）";
    container.appendChild(dropArea);

    // 拖拽事件
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

        uploadFiles(files, subdirWidget.value, function(idx, prog) {
            uploadBtn.textContent = "上传中 " + (idx + 1) + "/" + files.length + " " + Math.round(prog * 100) + "%";
        }, function(err) {
            if (err) {
                alert("上传失败: " + err.message);
            }
            // 无论成败都恢复按钮状态（和原逻辑一致）
            uploadBtn.textContent = originalText;
            uploadBtn.disabled = false;
            // 上传成功才刷新列表（还原原版逻辑）
            if (!err) {
                updateList();
            }
        });
    });

    // 按钮容器
    var btnDiv = document.createElement("div");
    btnDiv.style.display = "flex";
    btnDiv.style.gap = "8px";
    btnDiv.style.marginBottom = "8px";
    container.appendChild(btnDiv);
    
    var selectBtn = document.createElement("button");
    selectBtn.textContent = "📁 选择目录";
    var uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📤 选择音频上传";
    var refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    var applyBtn = document.createElement("button");
    applyBtn.textContent = "✅ 应用排序";
    btnDiv.appendChild(selectBtn);
    btnDiv.appendChild(uploadBtn);
    btnDiv.appendChild(refreshBtn);
    btnDiv.appendChild(applyBtn);
    
    selectBtn.onclick=function(){
        FxAiFolderSelector("audio").then(result => {
            if(result!==undefined)
            {
                subdirWidget.value = result;
                updateList();
            }
        });
    }

    // 列表容器
    var listDiv = document.createElement("div");
    listDiv.style.display = "flex";
    listDiv.style.flexWrap = "wrap";
    listDiv.style.gap = "5px";
    listDiv.style.maxHeight = "290px";
    listDiv.style.overflowY = "auto";
    listDiv.style.padding = "4px";
    listDiv.style.border = "1px solid #666";
    container.appendChild(listDiv);

    // 点击选择文件上传
    uploadBtn.onclick = function() {
        var input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = "audio/*";
        input.onchange = function() {
            if (!input.files.length) return;
            var files = Array.prototype.slice.call(input.files);
            var originalText = uploadBtn.textContent;
            uploadBtn.textContent = "上传中...";
            uploadBtn.disabled = true;

            uploadFiles(files, subdirWidget.value, function(idx, prog) {
                uploadBtn.textContent = "上传中 " + (idx + 1) + "/" + files.length + " " + Math.round(prog * 100) + "%";
            }, function(err) {
                if (err) {
                    alert("上传失败: " + err.message);
                }
                uploadBtn.textContent = originalText;
                uploadBtn.disabled = false;
                if (!err) {
                    updateList();
                }
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
        var items = listDiv.querySelectorAll(".audio-item");
        var ordered = [];
        for (var m = 0; m < items.length; m++) {
            ordered.push(items[m].dataset.filename);
        }
        applyChanges(subdirWidget.value, ordered, function(err) {
            if (err) {
                alert("应用失败: " + err.message);
            } else {
                updateList();
            }
        });
    };

    // 更新列表核心方法
    function updateList() {
        fetchFileList(subdirWidget.value, function(err, files) {
            listDiv.innerHTML = "";
            if (err) return;

            // 原 for...of 改为标准 for 循环
            for (var fIdx = 0; fIdx < files.length; fIdx++) {
                var file = files[fIdx];
                var item = document.createElement("div");
                item.className = "audio-item";
                item.dataset.filename = file;
                item.style.position = "relative";
                item.style.width = "250px";
                item.style.height = "80px";
                item.style.margin = "4px";
                item.style.cursor = "grab";
                item.style.borderRadius = "6px";
                item.style.overflow = "hidden";
                item.style.display = "flex";
                item.style.flexDirection = "column";
                item.style.alignItems = "center";
                item.style.justifyContent = "center";

                // 音频标签
                var audio = document.createElement("audio");
                audio.controls = true;
                audio.style.width = "100%";
                audio.style.marginBottom = "4px";
                var audioSrc = api.apiURL("/fxai/audio/preview?subdir=" + encodeURIComponent(subdirWidget.value) + "&filename=" + encodeURIComponent(file));
                audio.src = audioSrc;

                // 文件名
                var nameSpan = document.createElement("div");
                nameSpan.textContent = file;
                nameSpan.style.color = "white";
                nameSpan.style.fontSize = "10px";
                nameSpan.style.textAlign = "center";
                nameSpan.style.padding = "2px";
                nameSpan.style.whiteSpace = "nowrap";
                nameSpan.style.overflow = "hidden";
                nameSpan.style.textOverflow = "ellipsis";
                nameSpan.style.width = "100%";

                // 删除按钮
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

                // ====================== 修复 BUG 核心 ======================
                (function(currentItem, currentFile){
                    delBtn.onclick = function(e) {
                        e.stopPropagation();
                        // 真正删除服务器文件
                        deleteAudio(subdirWidget.value, currentFile, function(err){
                            if(err){
                                alert("删除失败："+err.message);
                            }else{
                                // 只删除当前UI，不刷新整个列表
                                currentItem.remove();
                            }
                        });
                    };
                })(item, file);
                // ===========================================================

                item.appendChild(audio);
                item.appendChild(nameSpan);
                item.appendChild(delBtn);
                listDiv.appendChild(item);
            }

            // 动态加载 Sortable 并初始化
            if (!window.Sortable) {
                var script = document.createElement("script");
                script.src = "./Sortable.min.js";
                script.onload = function() {
                    initSortable();
                };
                document.head.appendChild(script);
            } else {
                initSortable();
            }
        });
    }

    // 初始化拖拽排序
    function initSortable() {
        if (sortable) {
            sortable.destroy();
        }
        sortable = new Sortable(listDiv, {
            animation: 150,
            handle: ".audio-item",
            ghostClass: "sortable-ghost"
        });
    }

    // 目录切换回调（完全还原原版 this 指向 + 调用逻辑）
    var origCallback = subdirWidget.callback;
    subdirWidget.callback = function(v) {
        if (origCallback) {
            origCallback.call(this, v);
        }
        updateList();
    };

    // 初始化渲染列表
    updateList();
}

// 注册扩展（保留原 import 语法 + 原生结构）
app.registerExtension({
    name: "FxAiAudioManager",
    nodeCreated: function(node) {
        if (node.comfyClass === TARGET_CLASS) {
            setTimeout(function() {
                addUI(node);
            }, 100);
        }
    }
});