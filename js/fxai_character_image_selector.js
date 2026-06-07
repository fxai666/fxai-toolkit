import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
(function () {

    // 注册扩展
    const ext = {
        name: "FxAi.CharacterSelector.Preview.ES5",
        setup: function () {
            // 注入预览样式
            var style = document.createElement("style");
            style.textContent = `
                .fxai-previews { display: flex; gap: 4px; padding: 4px; flex-wrap: wrap; }
                .fxai-previews img { width:48px; height:48px; object-fit:cover; border-radius:4px; }
            `;
            document.head.appendChild(style);
        },
        beforeRegisterNodeDef: function (nodeType, nodeData, app) {
            if (nodeData.name === "FxAiCharacterImageSelector") {
                var onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    // 预览区域
                    var previewEl = document.createElement("div");
                    previewEl.className = "fxai-previews";
                    this.addDOMWidget("previews", "preview", previewEl);

                    // 按钮
                    var btn = document.createElement("button");
                    btn.textContent = "📁 选择资源图片";
                    btn.style.padding = "4px";
                    this.addDOMWidget("select_btn", "button", btn);

                    var self = this;

                    // 点击事件 ES5
                    btn.onclick = function () {
                        FxAiCharacterAssetsSelector().then(result => {
                            if(result!==undefined)
                            {
                                var w = self.widgets.find(function (x) {
                                    return x.name === "selected_files";
                                });
                                var files=w.value = result.split(",");
                                app.graph.setDirtyCanvas(true);
                                
                                // 预览图
                                previewEl.innerHTML = "";
                                files.forEach(function (path) {
                                    var img = new Image();
                                    img.src = `/fxai/image/v2/preview?filename=${encodeURIComponent(path)}&t=${Date.now()}`;
                                    img.className = "fxai-preview-img";
                                    previewEl.appendChild(img);
                                });
                            }
                        });
                    };

                    return r;
                };
            }
        }
    };

    app.registerExtension(ext);

})();