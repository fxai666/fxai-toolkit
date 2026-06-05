import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ==============================================
// 🔥 分类配置（不变）
// ==============================================
const CATEGORY_CONFIG = {
    "套装": "clothes",
    "首饰": "jewelry",
    "上衣": "tops",
    "胸罩": "bra",
    "裤子": "pants",
    "裙子": "skirts",
    "内裤": "underpants",
    "鞋袜": "shoessocks",
    "姿势": "pose",
    "装备": "equipment",
    "场景": "scene",
    "家具": "furniture",
    "座驾": "vehicle",
    "产品": "products",
    "其他": "other"
};

// ==============================================
// 工具：获取文件列表
// ==============================================
function fetchFileList(subdir) {
    return new Promise(resolve => {
        const url = api.apiURL("/fxai/image/v2/list?subdir=" + encodeURIComponent(subdir));
        fetch(url)
            .then(resp => resp.ok ? resp.json() : { files: [] })
            .then(data => resolve(data.files || []))
            .catch(() => resolve([]));
    });
}

// ==============================================
// 全局弹窗
// ==============================================
window.FxAiCharacterAssetsSelector = function(selectStr) {
    return new Promise(resolve => {
        // 选中数组全局常驻，切换标签不丢失
        let selected = (selectStr||"")
            .split(",")
            .map(item => item.trim())
            .filter(item => item !== "");

        // 遮罩
        const mask = document.createElement("div");
        mask.style.cssText = `
            position: fixed; inset: 0; z-index: 99999;
            background: rgba(0,0,0,0.85);
            display: flex; align-items: center; justify-content: center;
        `;
        document.body.appendChild(mask);

        // 弹窗
        const modal = document.createElement("div");
        modal.style.cssText = `
            width: 900px; max-width: 95vw;
            height: 750px; max-height: 90vh;
            background: #222; border-radius: 10px;
            padding: 20px; box-sizing: border-box;
            display: flex; flex-direction: column; gap: 16px;
        `;
        mask.appendChild(modal);

        // 标题
        const title = document.createElement("div");
        title.textContent = "🖼️ 选择图片";
        title.style.cssText = "font-size: 18px; color: #fff; font-weight: bold;";
        modal.appendChild(title);

        let currentSubdir = Object.values(CATEGORY_CONFIG)[0];

        // 标签栏
        const tabBar = document.createElement("div");
        tabBar.style.cssText = "display: flex; gap: 8px; flex-wrap: wrap; margin-left:5px";
        modal.appendChild(tabBar);

        function setActiveTab(tab) {
            tabBar.querySelectorAll("button").forEach(t => {
                t.style.background = "#333"; t.style.color = "#ccc";
            });
            tab.style.background = "#4a8fff"; tab.style.color = "#fff";
        }

        Object.entries(CATEGORY_CONFIG).forEach(([label, dir]) => {
            const btn = document.createElement("button");
            btn.textContent = label;
            btn.style.cssText = `
                padding: 3px 6px; border: none; border-radius: 4px;
                background: #333; color: #ccc; cursor: pointer;
            `;
            btn.onclick = () => {
                currentSubdir = dir;
                setActiveTab(btn);
                renderList();
            };
            tabBar.appendChild(btn);
            if (dir === currentSubdir) setActiveTab(btn);
        });

        // 图片展示区：弹性自动换行布局
        const listContainer = document.createElement("div");
        listContainer.style.cssText = `
            flex: 1; overflow-y: auto;
            display: flex; flex-wrap: wrap; gap: 5px;
            padding:4px; align-content:flex-start;
        `;
        modal.appendChild(listContainer);

        const selectedWrap = document.createElement("div");
        selectedWrap.style.cssText = `
            min-height:86px; max-height:120px; overflow-x:auto; overflow-y:hidden;
            background:#2b2b2b; border-radius:6px; padding:10px;
            display:flex; gap:8px; align-items:center;
        `;
        modal.appendChild(selectedWrap);

        // 刷新底部已选预览列表 + 自动编号
        function renderSelectedBar() {
            selectedWrap.innerHTML = "";
            if(selected.length === 0){
                selectedWrap.innerHTML = '<span style="color:#999;">暂无选中素材，点击上方图片添加</span>';
                return;
            }
            // 遍历生成带序号的选中项
            selected.forEach((path, index)=>{
                let splitArr = path.split("/");
                let sub = splitArr[0];
                let fname = splitArr[1];
                const previewUrl = api.apiURL(`/fxai/image/v2/preview?subdir=${encodeURIComponent(sub)}&filename=${encodeURIComponent(fname)}`);

                const item = document.createElement("div");
                item.style.cssText = `
                    width:70px; height:70px; position:relative; border-radius:4px; overflow:hidden;
                    flex-shrink:0; border:2px solid #4a8fff;
                `;
                // 缩略图
                const img = document.createElement("img");
                img.src = previewUrl;
                img.style.cssText = "width:100%;height:100%;object-fit:cover;";
                
                // 🔥 数字序号（左上角）
                const numTag = document.createElement("div");
                numTag.textContent = index + 1; // 1、2、3...
                numTag.style.cssText = `
                    position:absolute; top:0; left:0; width:22px; height:22px;
                    background:#4a8fff; color:#fff; font-size:12px; font-weight:bold;
                    text-align:center; line-height:22px; border-radius:0 0 4px 0;
                    z-index:2;
                `;

                // 删除按钮
                const delBtn = document.createElement("div");
                delBtn.textContent = "×";
                delBtn.style.cssText = `
                    position:absolute; top:0; right:0; width:18px;height:18px;
                    background:#f54242; color:#fff; text-align:center; line-height:18px;
                    font-size:14px; cursor:pointer; z-index:2;
                `;
                delBtn.onclick = (e)=>{
                    e.stopPropagation();
                    let idx = selected.indexOf(path);
                    if(idx>-1) selected.splice(idx,1);
                    renderSelectedBar();
                    renderList();
                };
                item.append(img, numTag, delBtn);
                selectedWrap.appendChild(item);
            })
        }

        // 底部按钮栏
        const bottomBar = document.createElement("div");
        bottomBar.style.cssText = "display: flex; justify-content: flex-end; gap: 10px;";
        modal.appendChild(bottomBar);

        const btnCancel = document.createElement("button");
        btnCancel.textContent = "取消";
        const btnConfirm = document.createElement("button");
        btnConfirm.textContent = "✅ 确认选择";
        btnConfirm.style.background = "#4a8fff"; btnConfirm.style.color = "#fff";
        bottomBar.append(btnCancel, btnConfirm);

        function close() {
            document.body.removeChild(mask);
        }
        btnCancel.onclick = () => { resolve(); close(); };
        btnConfirm.onclick = () => { resolve(selected.join(",")); close(); };
        mask.onclick = (e) => e.target === mask && close();

        // 渲染上方图片列表
        function renderList() {
            listContainer.innerHTML = "";
            fetchFileList(currentSubdir).then(files => {
                files.forEach(filename => {
                    const previewUrl = api.apiURL(`/fxai/image/v2/preview?subdir=${encodeURIComponent(currentSubdir)}&filename=${encodeURIComponent(filename)}`);
                    const realPath = `${currentSubdir}/${filename}`;
                    const isSelected = selected.includes(realPath);

                    const item = document.createElement("div");
                    item.style.cssText = `
                        width:128px;height:128px;position:relative; border-radius:6px;
                        overflow:hidden; cursor:pointer; flex-shrink:0;
                        border:3px solid ${isSelected ? "#4a8fff" : "transparent"}; background:#111;
                    `;
                    const img = document.createElement("img");
                    img.src = previewUrl;
                    img.style.cssText = `position:absolute; inset:0; width:100%; height:100%; object-fit:cover;`;
                    item.appendChild(img);
                    listContainer.appendChild(item);

                    item.onclick = () => {
                        if(!selected.includes(realPath)){
                            selected.push(realPath);
                            renderSelectedBar();
                            renderList();
                        }
                    };
                });
            });
        }

        // 初始渲染
        renderList();
        renderSelectedBar();
    });
};

// 注册扩展
app.registerExtension({
    name: "FxAiCharacterAssetsSelector"
});