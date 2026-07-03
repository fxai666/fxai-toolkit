import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

var TARGET_CLASS = "FxAiVideoManager";

var sortable = null;

// 获取视频文件列表
function fetchFileList(subdir, callback) {
    fetch(api.apiURL("/fxai/video/list?subdir=" + encodeURIComponent(subdir)))
        .then(function (resp) {
            if (!resp.ok) {
                callback(null, []);
                return;
            }
            return resp.json();
        })
        .then(function (data) {
            callback(null, data.files);
        })
        .catch(function (err) {
            callback(err, []);
        });
}

// 视频上传逻辑
function uploadFiles(files, subdir, onProgress, callback) {
    var index = 0;
    function uploadNext() {
        if (index >= files.length) {
            callback(null);
            return;
        }
        var file = files[index];
        var formData = new FormData();
        formData.append("video", file, file.name);
        formData.append("subdir", subdir);

        fetch(api.apiURL("/fxai/video/upload"), {
            method: "POST",
            body: formData
        })
        .then(function (response) {
            if (!response.ok) {
                throw new Error("上传失败: " + response.status);
            }
            if (onProgress) {
                onProgress(index, 1);
            }
            index++;
            uploadNext();
        })
        .catch(function (err) {
            callback(new Error("文件 " + file.name + " 上传失败: " + err.message));
        });
    }
    uploadNext();
}

// 应用排序/删除更改
function applyChanges(subdir, orderedFilenames, callback) {
    fetch(api.apiURL("/fxai/video/apply"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subdir: subdir, ordered_filenames: orderedFilenames })
    })
    .then(function (resp) {
        if (!resp.ok) {
            throw new Error("应用更改失败");
        }
        return resp.json();
    })
    .then(function (data) {
        callback(null, data.files);
    })
    .catch(function (err) {
        callback(err);
    });
}

// 全局阻止浏览器默认拖拽行为
function preventDefaultDragDrop() {
    document.addEventListener('dragover', function (e) { e.preventDefault(); });
    document.addEventListener('drop', function (e) { e.preventDefault(); });
    document.addEventListener('dragenter', function (e) { e.preventDefault(); });
    document.addEventListener('dragleave', function (e) { e.preventDefault(); });
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
    var domWidget = node.addDOMWidget("video_ui", "video_ui", container);
    domWidget.computeSize = function () { return [600, 545]; };

    var dropArea = document.createElement("div");
    dropArea.style.padding = "20px";
    dropArea.style.marginBottom = "8px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.textContent = "📥 拖拽视频到这里上传（支持多文件）";
    container.appendChild(dropArea);

    dropArea.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropArea.style.borderColor = "#fff";
    });
    dropArea.addEventListener("dragleave", function () {
        dropArea.style.borderColor = "#777";
    });
    dropArea.addEventListener("drop", function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.style.borderColor = "#777";
        var files = Array.prototype.slice.call(e.dataTransfer.files);
        if (!files.length) return;

        var videoFiles = [];
        for (var i = 0; i < files.length; i++) {
            if (files[i].type.startsWith('video/')) {
                videoFiles.push(files[i]);
            }
        }
        if (!videoFiles.length) {
            alert("请上传视频文件！");
            return;
        }

        var originalText = uploadBtn.textContent;
        uploadBtn.textContent = "上传中...";
        uploadBtn.disabled = true;

        uploadFiles(videoFiles, subdirWidget.value, function (idx, prog) {
            uploadBtn.textContent = "上传中 " + (idx + 1) + "/" + videoFiles.length + " " + Math.round(prog * 100) + "%";
        }, function (err) {
            if (err) {
                alert("上传失败: " + err.message);
                uploadBtn.textContent = originalText;
                uploadBtn.disabled = false;
            } else {
                updateList(function () {
                    uploadBtn.textContent = originalText;
                    uploadBtn.disabled = false;
                });
            }
        });
    });

    var btnDiv = document.createElement("div");
    btnDiv.style.display = "flex";
    btnDiv.style.gap = "8px";
    btnDiv.style.marginBottom = "8px";
    container.appendChild(btnDiv);
    
    var selectBtn = document.createElement("button");
    selectBtn.textContent = "📁 选择目录";
    var uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📤 选择视频上传";
    var refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    var applyBtn = document.createElement("button");
    applyBtn.textContent = "✅ 应用排序/删除";
    btnDiv.appendChild(selectBtn);
    btnDiv.appendChild(uploadBtn);
    btnDiv.appendChild(refreshBtn);
    btnDiv.appendChild(applyBtn);
    
    selectBtn.onclick=function(){
        FxAiFolderSelector("video").then(result => {
            if(result!==undefined)
            {
                subdirWidget.value = result;
                updateList();
            }
        });
    }

    var listDiv = document.createElement("div");
    listDiv.style.display = "flex";
    listDiv.style.flexWrap = "wrap";
    listDiv.style.gap = "8px";
    listDiv.style.maxHeight = "400px";
    listDiv.style.overflowY = "auto";
    listDiv.style.padding = "4px";
    listDiv.style.border = "1px solid #666";
    container.appendChild(listDiv);

    uploadBtn.onclick = function () {
        var input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = "video/*";
        input.onchange = function () {
            if (!input.files.length) return;
            var files = Array.prototype.slice.call(input.files);
            var videoFiles = [];
            for (var i = 0; i < files.length; i++) {
                if (files[i].type.startsWith('video/')) {
                    videoFiles.push(files[i]);
                }
            }
            if (!videoFiles.length) {
                alert("请选择视频文件！");
                return;
            }

            var originalText = uploadBtn.textContent;
            uploadBtn.textContent = "上传中...";
            uploadBtn.disabled = true;

            uploadFiles(videoFiles, subdirWidget.value, function (idx, prog) {
                uploadBtn.textContent = "上传中 " + (idx + 1) + "/" + videoFiles.length + " " + Math.round(prog * 100) + "%";
            }, function (err) {
                if (err) {
                    alert("上传失败: " + err.message);
                    uploadBtn.textContent = originalText;
                    uploadBtn.disabled = false;
                } else {
                    updateList(function () {
                        uploadBtn.textContent = originalText;
                        uploadBtn.disabled = false;
                    });
                }
            });
        };
        input.click();
    };

    refreshBtn.onclick = function () {
        updateList();
    };

    applyBtn.onclick = function () {
        var items = listDiv.querySelectorAll(".video-item");
        var ordered = [];
        for (var i = 0; i < items.length; i++) {
            ordered.push(items[i].dataset.filename);
        }
        applyChanges(subdirWidget.value, ordered, function (err) {
            if (err) {
                alert("应用失败: " + err.message);
            } else {
                updateList();
            }
        });
    };

    function updateList(callback) {
        fetchFileList(subdirWidget.value, function (err, files) {
            listDiv.innerHTML = "";

            files.forEach(function (file) {
                var item = document.createElement("div");
                item.className = "video-item";
                item.dataset.filename = file;
                item.style.position = "relative";
                item.style.width = "280px";
                item.style.height = "180px";
                item.style.margin = "4px";
                item.style.cursor = "grab";
                item.style.borderRadius = "6px";
                item.style.overflow = "hidden";
                item.style.display = "flex";
                item.style.flexDirection = "column";
                item.style.alignItems = "center";
                item.style.justifyContent = "center";
                item.style.backgroundColor = "#333";

                var video = document.createElement("video");
                video.controls = true;
                video.style.width = "100%";
                video.style.height = "140px";
                video.style.objectFit = "contain";
                video.src = api.apiURL("/fxai/video/loop/preview?subdir=" + encodeURIComponent(subdirWidget.value) + "&filename=" + encodeURIComponent(file) + "&t=" + Date.now());

                video.onerror = function () {
                    video.style.display = "none";
                    var fallback = document.createElement("div");
                    fallback.textContent = "无法预览视频";
                    fallback.style.color = "#fff";
                    fallback.style.padding = "10px";
                    item.insertBefore(fallback, item.firstChild);
                };

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

                var delBtn = document.createElement("button");
                delBtn.textContent = "✖";
                delBtn.style.position = "absolute";
                delBtn.style.top = "2px";
                delBtn.style.right = "2px";
                delBtn.style.backgroundColor = "rgba(255,0,0,0.7)";
                delBtn.style.color = "white";
                delBtn.style.border = "none";
                delBtn.style.borderRadius = "50%";
                delBtn.style.width = "20px";
                delBtn.style.height = "20px";
                delBtn.style.cursor = "pointer";

                (function (currentFile, currentItem) {
                    delBtn.onclick = function (e) {
                        e.stopPropagation();
                        fetch(api.apiURL("/fxai/video/delete?subdir=" + encodeURIComponent(subdirWidget.value) + "&filename=" + encodeURIComponent(currentFile)))
                        .then(function (resp) {
                            if (!resp.ok) {
                                throw new Error("删除请求失败");
                            }
                            return resp.json();
                        })
                        .then(function (data) {
                            if (data.success) {
                                currentItem.remove();
                            } else {
                                alert("删除失败: " + data.error);
                            }
                        })
                        .catch(function (err) {
                            alert("删除出错: " + err.message);
                        });
                    };
                })(file, item);

                item.appendChild(video);
                item.appendChild(nameSpan);
                item.appendChild(delBtn);
                listDiv.appendChild(item);
            });

            if (!window.Sortable) {
                var script = document.createElement("script");
                script.src = "./Sortable.min.js";
                script.onload = function () {
                    initSortable();
                };
                document.head.appendChild(script);
            } else {
                initSortable();
            }

            function initSortable() {
                if (sortable) sortable.destroy();
                sortable = new Sortable(listDiv, {
                    animation: 150,
                    handle: ".video-item",
                    ghostClass: "sortable-ghost",
                    chosenClass: "sortable-chosen"
                });
                if (callback) callback();
            }
        });
    }

    var origCallback = subdirWidget.callback;
    subdirWidget.callback = function (v) {
        if (origCallback) {
            origCallback.call(this, v);
        }
        updateList();
    };

    updateList();
}

app.registerExtension({
    name: "FxAiVideoManager",
    nodeCreated: function (node) {
        if (node.comfyClass === TARGET_CLASS) {
            setTimeout(function () {
                addUI(node);
            }, 100);
        }
    }
});