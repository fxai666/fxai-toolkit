import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiImageManager";

let sortable = null;

async function fetchFileList(subdir) {
    const resp = await fetch(api.apiURL(`/fxai/image/list?subdir=${encodeURIComponent(subdir)}`));
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.files;
}

async function getNextNumber(subdir) {
    const resp = await fetch(api.apiURL(`/fxai/image/next_number?subdir=${encodeURIComponent(subdir)}`));
    if (!resp.ok) throw new Error("获取序号失败");
    const data = await resp.json();
    return data.next_num;
}

// 修复：正确的上传文件逻辑
async function uploadFiles(files, subdir, onProgress) {
    let nextNum = await getNextNumber(subdir);
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const ext = file.name.split('.').pop();
        const newName = String(nextNum + i).padStart(3, '0') + '.' + ext;

        const formData = new FormData();
        // 关键修复：直接 append 原始 file，重命名在后端处理更稳定
        formData.append("image", file, newName); 
        formData.append("subdir", subdir);

        try {
            // 修复：使用标准 fetch 上传，替代有问题的 XMLHttpRequest
            const response = await fetch(api.apiURL("/fxai/image/upload"), {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`上传失败: ${response.status}`);
            }
            
            onProgress?.(i, 1); // 进度完成
        } catch (err) {
            throw new Error(`文件 ${newName} 上传失败: ${err.message}`);
        }
    }
}

async function applyChanges(subdir, orderedFilenames) {
    const resp = await fetch(api.apiURL("/fxai/image/apply"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subdir, ordered_filenames: orderedFilenames })
        });
    if (!resp.ok) throw new Error("应用更改失败");
    const data = await resp.json();
    return data.files;
}

// 全局阻止浏览器默认拖拽行为（修复：防止拖拽图片自动打开）
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
    container.style.minWidth = "300px";
    node.addDOMWidget("image_ui", "image_ui", container, { serialize: false });

    // 拖拽上传区域
    const dropArea = document.createElement("div");
    dropArea.style.padding = "12px";
    dropArea.style.marginBottom = "8px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.textContent = "📥 拖拽图片到这里上传（支持多图）";
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
        e.preventDefault(); // 强制阻止默认行为
        e.stopPropagation();
        dropArea.style.borderColor = "#777";
        const files = Array.from(e.dataTransfer.files);
        if (!files.length) return;

        const originalText = uploadBtn.textContent;
        uploadBtn.textContent = "上传中...";
        uploadBtn.disabled = true;
        try {
            await uploadFiles(files, subdirWidget.value, (idx, prog) => {
                uploadBtn.textContent = `上传中 ${idx+1}/${files.length} ${Math.round(prog*100)}%`;
            });
            await updateList();
        } catch(err) {
            alert("上传失败: " + err.message);
        } finally {
            uploadBtn.textContent = originalText;
            uploadBtn.disabled = false;
        }
    });

    const btnDiv = document.createElement("div");
    btnDiv.style.display = "flex";
    btnDiv.style.gap = "8px";
    btnDiv.style.marginBottom = "8px";
    container.appendChild(btnDiv);

    const uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📤 选择图片上传";
    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "✅ 确认操作";
    btnDiv.appendChild(uploadBtn);
    btnDiv.appendChild(refreshBtn);
    btnDiv.appendChild(applyBtn);

    const listDiv = document.createElement("div");
    listDiv.style.display = "flex";
    listDiv.style.flexWrap = "wrap";
    listDiv.style.gap = "5px";
    listDiv.style.maxHeight = "400px";
    listDiv.style.overflowY = "auto";
    listDiv.style.padding = "4px";
    listDiv.style.border = "1px solid #666";
    container.appendChild(listDiv);

    // 点击上传
    uploadBtn.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = "image/*";
        input.onchange = async () => {
            if (!input.files.length) return;
            const files = Array.from(input.files);
            const originalText = uploadBtn.textContent;
            uploadBtn.textContent = "上传中...";
            uploadBtn.disabled = true;
            try {
                await uploadFiles(files, subdirWidget.value, (idx, prog) => {
                    uploadBtn.textContent = `上传中 ${idx+1}/${files.length} ${Math.round(prog*100)}%`;
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

    refreshBtn.onclick = async () => {
        await updateList();
    };

    applyBtn.onclick = async () => {
        const items = listDiv.querySelectorAll(".image-item");
        const ordered = Array.from(items).map(item => item.dataset.filename);
        try {
            await applyChanges(subdirWidget.value, ordered);
            await updateList();
        } catch(err) {
            alert("应用失败: " + err.message);
        }
    };

    async function updateList() {
        const files = await fetchFileList(subdirWidget.value);
        listDiv.innerHTML = "";

        for (const file of files) {
            const item = document.createElement("div");
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

            const img = document.createElement("img");
            img.src = api.apiURL(`/fxai/image/preview?subdir=${encodeURIComponent(subdirWidget.value)}&filename=${encodeURIComponent(file)}`);
            img.style.width = "100%";
            img.style.height = "100%";
            img.style.objectFit = "cover";
            img.style.display = "block";

            const nameSpan = document.createElement("div");
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

            const delBtn = document.createElement("button");
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
            delBtn.onclick = (e) => {
                e.stopPropagation();
                item.remove();
            };

            item.appendChild(img);
            item.appendChild(nameSpan);
            item.appendChild(delBtn);
            listDiv.appendChild(item);
        }

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
            handle: ".image-item",
            ghostClass: "sortable-ghost"
        });
    }

    const origCallback = subdirWidget.callback;
    subdirWidget.callback = function(v) {
        origCallback?.call(this, v);
        updateList();
    };

    updateList();
}

app.registerExtension({
    name: "BatchImageLoader",
    async nodeCreated(node) {
        if (node.comfyClass === TARGET_CLASS) {
            setTimeout(() => addUI(node), 100);
        }
},
});