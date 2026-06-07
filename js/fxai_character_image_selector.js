import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
(function () {

    const ext = {
        name: "FxAi.CharacterSelector.Preview.ES5",
        setup: function () {
            var style = document.createElement("style");
            style.textContent = `
                .fxai-previews { 
                    display: flex; 
                    gap: 8px;        
                    padding: 6px; 
                    flex-wrap: wrap; 
                    max-width: 100%;
                }
                .fxai-previews img { 
                    width:80px;      
                    height:80px; 
                    object-fit:cover; 
                    border-radius:6px; 
                    cursor:pointer;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                }
                /* 按钮固定大小，不会变形 */
                .fxai-fixed-btn {
                    width: 180px !important;
                    height: 32px !important;
                    padding: 0 !important;
                    line-height: 32px !important;
                    font-size: 14px !important;
                    flex-shrink: 0 !important;
                    white-space: nowrap !important;
                    cursor: pointer;
                }
            `;
            document.head.appendChild(style);
        },
        beforeRegisterNodeDef: function (nodeType, nodeData, app) {
            if (nodeData.name === "FxAiCharacterImageSelector") {
                var onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    var previewEl = document.createElement("div");
                    previewEl.className = "fxai-previews";
                    this.addDOMWidget("previews", "preview", previewEl);

                    var btn = document.createElement("button");
                    btn.textContent = "📁 选择资源图片";
                    btn.className = "fxai-fixed-btn";
                    this.addDOMWidget("select_btn", "button", btn);

                    var self = this;

                    // ==============================
                    // 核心：传入字符串，自动分割显示多张图
                    // ==============================
                    function renderPreviews(fileStr) {
                        previewEl.innerHTML = "";
                        
                        // 空值判断
                        if (!fileStr || fileStr.trim() === "") {
                            return;
                        }

                        // 字符串分割成数组（只用来显示，不修改存储）
                        var paths = fileStr.split(",").map(p => p.trim()).filter(Boolean);

                        paths.forEach(path => {
                            var img = new Image();
                            img.src = `/fxai/image/v2/preview?filename=${encodeURIComponent(path)}&t=${Date.now()}`;
                            previewEl.appendChild(img);
                        });
                    }

                    // ==============================
                    // 按钮选择：保存字符串！不存数组！
                    // ==============================
                    btn.onclick = function () {
                        FxAiCharacterAssetsSelector().then(result => {
                            if (result !== undefined) {
                                var w = self.widgets.find(x => x.name === "selected_files");
                                // ✅ 直接存字符串，不 split！
                                w.value = result;
                                app.graph.setDirtyCanvas(true);
                                // 用字符串去渲染预览
                                renderPreviews(w.value);
                            }
                        });
                    };

                    // ==============================
                    // 初始化加载（读取字符串）
                    // ==============================
                    setTimeout(() => {
                        const widget = self.widgets.find(x => x.name === "selected_files");
                        if (widget && widget.value) {
                            renderPreviews(widget.value);
                        }
                    }, 200);

                    return r;
                };
            }
        }
    };

    app.registerExtension(ext);

})();