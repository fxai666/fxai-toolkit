import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { CATEGORY_CONFIG } from "./fxai_category_config.js";

// ==============================================
// 工具：获取文件列表
// ==============================================
function fetchFileList(subdir) {
    return new Promise(function (resolve) {
        var url = api.apiURL("/fxai/image/v2/list?subdir=" + encodeURIComponent(subdir));
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

// ==============================================
// 全局弹窗
// ==============================================
window.FxAiCharacterAssetsSelector = function (selectStr) {
    return new Promise(function (resolve) {
        // 选中数组全局常驻，切换标签不丢失
        var selected = (selectStr || "")
            .split(",")
            .map(function (item) {
                return item.trim();
            })
            .filter(function (item) {
                return item !== "";
            });

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
            width: 970px; max-width: 95vw;
            height: 750px; max-height: 90vh;
            background: #222; border-radius: 10px;
            padding: 10px; box-sizing: border-box;
            display: flex; flex-direction: column; gap: 16px;
        `;
        mask.appendChild(modal);

        // 标题
        var title = document.createElement("div");
        title.textContent = "🖼️ 选择图片";
        title.style.cssText = "font-size: 18px; color: #fff; font-weight: bold;";
        modal.appendChild(title);

        var currentSubdir = Object.values(CATEGORY_CONFIG)[0];

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

        Object.entries(CATEGORY_CONFIG).forEach(function (entry) {
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

        // 图片展示区：弹性自动换行布局
        var listContainer = document.createElement("div");
        listContainer.style.cssText = `
            flex: 1; overflow-y: auto;
            display: flex; flex-wrap: wrap; gap: 5px;
            padding:4px; align-content:flex-start;
        `;
        modal.appendChild(listContainer);

        var selectedWrap = document.createElement("div");
        selectedWrap.style.cssText = `
            min-height:86px; max-height:120px; overflow-x:auto; overflow-y:hidden;
            background:#2b2b2b; border-radius:6px; padding:10px;
            display:flex; gap:8px; align-items:center;
        `;
        modal.appendChild(selectedWrap);

        // 刷新底部已选预览列表 + 自动编号
        function renderSelectedBar() {
            selectedWrap.innerHTML = "";
            if (selected.length === 0) {
                selectedWrap.innerHTML = '<span style="color:#999;">暂无选中素材，点击上方图片添加</span>';
                return;
            }
            selected.forEach(function (path, index) {
                var splitArr = path.split("/");
                var sub = splitArr[0];
                var fname = splitArr[1];
                var previewUrl = api.apiURL("/fxai/image/v2/preview?subdir=" + encodeURIComponent(sub) + "&filename=" + encodeURIComponent(fname));

                var item = document.createElement("div");
                item.style.cssText = `
                    width:70px; height:70px; position:relative; border-radius:4px; overflow:hidden;
                    flex-shrink:0; border:2px solid #4a8fff;
                `;
                var img = document.createElement("img");
                img.src = previewUrl;
                img.style.cssText = "width:100%;height:100%;object-fit:cover;";

                var numTag = document.createElement("div");
                numTag.textContent = index + 1;
                numTag.style.cssText = `
                    position:absolute; top:0; left:0; width:22px; height:22px;
                    background:#4a8fff; color:#fff; font-size:12px; font-weight:bold;
                    text-align:center; line-height:22px; border-radius:0 0 4px 0;
                    z-index:2;
                `;

                var delBtn = document.createElement("div");
                delBtn.textContent = "×";
                delBtn.style.cssText = `
                    position:absolute; top:0; right:0; width:18px;height:18px;
                    background:#f54242; color:#fff; text-align:center; line-height:18px;
                    font-size:14px; cursor:pointer; z-index:2;
                `;
                delBtn.onclick = function (e) {
                    e.stopPropagation();
                    var idx = selected.indexOf(path);
                    if (idx > -1) selected.splice(idx, 1);
                    renderSelectedBar();
                    refreshAllItemBorder();
                };
                item.append(img, numTag, delBtn);
                selectedWrap.appendChild(item);
            });
        }

        // 批量刷新所有图片选中边框（仅样式，不重载列表）
        function refreshAllItemBorder() {
            var allItems = listContainer.querySelectorAll(".asset-item");
            allItems.forEach(function (el) {
                var path = el.dataset.path;
                if (selected.includes(path)) {
                    el.style.border = "3px solid #4a8fff";
                } else {
                    el.style.border = "3px solid transparent";
                }
            });
        }

        // 底部按钮栏
        var bottomBar = document.createElement("div");
        bottomBar.style.cssText = "display: flex; justify-content: flex-end; gap: 10px;";
        modal.appendChild(bottomBar);

        var btnCancel = document.createElement("button");
        btnCancel.textContent = "取消";
        var btnConfirm = document.createElement("button");
        btnConfirm.textContent = "✅ 确认选择";
        btnConfirm.style.background = "#4a8fff"; btnConfirm.style.color = "#fff";
        bottomBar.append(btnCancel, btnConfirm);

        function close() {
            document.body.removeChild(mask);
        }
        btnCancel.onclick = function () { resolve(); close(); };
        btnConfirm.onclick = function () { resolve(selected.join(",")); close(); };
        mask.onclick = function (e) {
            if (e.target === mask) close();
        };

        // 渲染图库列表（仅切换分类标签时完整重载）
        function renderList() {
            listContainer.innerHTML = "";
            fetchFileList(currentSubdir).then(function (files) {
                files.forEach(function (filename) {
                    var previewUrl = api.apiURL("/fxai/image/v2/preview?subdir=" + encodeURIComponent(currentSubdir) + "&filename=" + encodeURIComponent(filename));
                    var realPath = currentSubdir + "/" + filename;
                    var isSelected = selected.includes(realPath);

                    var item = document.createElement("div");
                    item.className = "asset-item";
                    item.dataset.path = realPath;
                    item.style.cssText = `
                        width:128px;height:128px;position:relative; border-radius:6px;
                        overflow:hidden; cursor:pointer; flex-shrink:0;
                        border:3px solid ${isSelected ? "#4a8fff" : "transparent"}; background:#111;
                    `;
                    var img = document.createElement("img");
                    img.src = previewUrl;
                    img.style.cssText = `position:absolute; inset:0; width:100%; height:100%; object-fit:cover;`;
                    item.appendChild(img);
                    listContainer.appendChild(item);

                    // 核心：点击切换选中/取消选中，只改当前样式
                    item.onclick = function () {
                        var idx = selected.indexOf(realPath);
                        if (idx === -1) {
                            // 未选中：加入数组 + 高亮边框
                            selected.push(realPath);
                            item.style.border = "3px solid #4a8fff";
                        } else {
                            // 已选中：删除数组 + 取消边框
                            selected.splice(idx, 1);
                            item.style.border = "3px solid transparent";
                        }
                        // 同步刷新底部选中栏
                        renderSelectedBar();
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