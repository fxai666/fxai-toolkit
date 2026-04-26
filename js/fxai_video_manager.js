import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiVideoManager";

let sortable = null;

// 获取视频文件列表
async function fetchFileList(subdir) {
    const resp = await fetch(api.apiURL(`/fxai/video/list?subdir=${encodeURIComponent(subdir)}`));
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.files;
}

// 视频上传逻辑
async function uploadFiles(files, subdir, onProgress) {
    for (let i = 0; i < files.length; i++) {
        const file = files[i];

        const formData = new FormData();
        formData.append("video", file, file.name); 
        formData.append("subdir", subdir);

        try {
            const response = await fetch(api.apiURL("/fxai/video/upload"), {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`上传失败: ${response.status}`);
            }
            
            onProgress?.(i, 1); // 进度完成
        } catch (err) {
            throw new Error(`文件 ${file.name} 上传失败: ${err.message}`);
        }
    }
}

// 应用排序/删除更改
async function applyChanges(subdir, orderedFilenames) {
    const resp = await fetch(api.apiURL("/fxai/video/apply"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subdir, ordered_filenames: orderedFilenames })
        });
    if (!resp.ok) throw new Error("应用更改失败");
    const data = await resp.json();
    return data.files;
}

// 全局阻止浏览器默认拖拽行为
function preventDefaultDragDrop() {
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
    document.addEventListener('dragenter', (e) => e.preventDefault());
    document.addEventListener('dragleave', (e) => e.preventDefault());
}

function addUI(node) {
    if (node._uiAdded) return;
    node._uiAdded = true;

    // 启用全局防拖拽打开
    preventDefaultDragDrop();

    const subdirWidget = node.widgets.find(w => w.name === "目录");
    if (!subdirWidget) {
        console.error("未找到子目录控件");
        return;
    }

    const container = document.createElement("div");
    container.style.padding = "8px";
    container.style.border = "1px solid #555";
    container.style.borderRadius = "4px";
    const domWidget = node.addDOMWidget("video_ui", "video_ui", container);
    domWidget.computeSize = () => [600, 545];

    // 拖拽上传区域
    const dropArea = document.createElement("div");
    dropArea.style.padding = "20px";
    dropArea.style.marginBottom = "8px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.textContent = "📥 拖拽视频到这里上传（支持多文件）";
    container.appendChild(dropArea);

    // 拖拽逻辑
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
        if (!files.length) return;

        // 过滤仅视频文件
        const videoFiles = files.filter(file => file.type.startsWith('video/'));
        if (!videoFiles.length) {
            alert("请上传视频文件！");
            return;
        }

        const originalText = uploadBtn.textContent;
        uploadBtn.textContent = "上传中...";
        uploadBtn.disabled = true;
        try {
            await uploadFiles(videoFiles, subdirWidget.value, (idx, prog) => {
                uploadBtn.textContent = `上传中 ${idx+1}/${videoFiles.length} ${Math.round(prog*100)}%`;
            });
            await updateList();
        } catch(err) {
            alert("上传失败: " + err.message);
        } finally {
            uploadBtn.textContent = originalText;
            uploadBtn.disabled = false;
        }
    });

    // 按钮区域
    const btnDiv = document.createElement("div");
    btnDiv.style.display = "flex";
    btnDiv.style.gap = "8px";
    btnDiv.style.marginBottom = "8px";
    container.appendChild(btnDiv);

    const uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📤 选择视频上传";
    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "✅ 应用排序/删除";
    btnDiv.appendChild(uploadBtn);
    btnDiv.appendChild(refreshBtn);
    btnDiv.appendChild(applyBtn);

    // 视频列表区域
    const listDiv = document.createElement("div");
    listDiv.style.display = "flex";
    listDiv.style.flexWrap = "wrap";
    listDiv.style.gap = "8px";
    listDiv.style.maxHeight = "400px";
    listDiv.style.overflowY = "auto";
    listDiv.style.padding = "4px";
    listDiv.style.border = "1px solid #666";
    container.appendChild(listDiv);

    // 点击上传按钮逻辑
    uploadBtn.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = "video/*"; // 仅接受视频文件
        input.onchange = async () => {
            if (!input.files.length) return;
            const files = Array.from(input.files);
            const videoFiles = files.filter(file => file.type.startsWith('video/'));
            if (!videoFiles.length) {
                alert("请选择视频文件！");
                return;
            }

            const originalText = uploadBtn.textContent;
            uploadBtn.textContent = "上传中...";
            uploadBtn.disabled = true;
            try {
                await uploadFiles(videoFiles, subdirWidget.value, (idx, prog) => {
                    uploadBtn.textContent = `上传中 ${idx+1}/${videoFiles.length} ${Math.round(prog*100)}%`;
                });
                await updateList();
            } catch(err) {
                alert("上传失败: " + err.message);
            } finally {
                uploadBtn.textContent = originalText;
                uploadBtn.disabled = false;
            }
        };
        input.click();
    };

    // 刷新按钮逻辑
    refreshBtn.onclick = async () => {
        await updateList();
    };

    // 应用更改按钮逻辑
    applyBtn.onclick = async () => {
        const items = listDiv.querySelectorAll(".video-item");
        const ordered = Array.from(items).map(item => item.dataset.filename);
        try {
            await applyChanges(subdirWidget.value, ordered);
            await updateList();
            alert("应用成功！");
        } catch(err) {
            alert("应用失败: " + err.message);
        }
    };

    // 更新视频列表
    async function updateList() {
        const files = await fetchFileList(subdirWidget.value);
        listDiv.innerHTML = "";

        for (const file of files) {
            const item = document.createElement("div");
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

        // 视频预览控件
            const video = document.createElement("video");
            video.controls = true;
            video.style.width = "100%";
            video.style.height = "140px";
            video.style.objectFit = "contain";
            video.src = api.apiURL(`/fxai/video/loop/preview?subdir=${encodeURIComponent(subdirWidget.value)}&filename=${encodeURIComponent(file)}`);
        // 视频加载失败降级
            video.onerror = () => {
                video.style.display = "none";
                const fallback = document.createElement("div");
                fallback.textContent = "无法预览视频";
                fallback.style.color = "#fff";
                fallback.style.padding = "10px";
                item.insertBefore(fallback, item.firstChild);
            };

        // 文件名显示
            const nameSpan = document.createElement("div");
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
            const delBtn = document.createElement("button");
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
            delBtn.onclick = (e) => {
                e.stopPropagation();
                if (confirm(`确定删除 ${file} 吗？`)) {
                    item.remove();
                }
            };

            item.appendChild(video);
            item.appendChild(nameSpan);
            item.appendChild(delBtn);
            listDiv.appendChild(item);
        }

        // 加载Sortable实现拖拽排序
        if (!window.Sortable) {
            await new Promise((resolve) => {
                const script = document.createElement("script");
                script.src = "./Sortable.min.js";
                script.onload = resolve;
                document.head.appendChild(script);
            });
        }
        if (sortable) sortable.destroy();
        sortable = new Sortable(listDiv, {
            animation: 150,
            handle: ".video-item",
            ghostClass: "sortable-ghost",
            chosenClass: "sortable-chosen"
        });
    }

    // 目录切换时刷新列表
    const origCallback = subdirWidget.callback;
    subdirWidget.callback = function(v) {
        origCallback?.call(this, v);
        updateList();
    };

    // 初始化列表
    updateList();
}

// 注册ComfyUI扩展
app.registerExtension({
    name: "FxAiVideoManager",
    async nodeCreated(node) {
        if (node.comfyClass === TARGET_CLASS) {
            setTimeout(() => addUI(node), 100);
        }
},
});