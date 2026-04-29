// ==============================================
// ComfyUI 官方规范：注册自定义节点前端
// ==============================================
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "FxAiVideoPreview",

    // 节点创建时绑定自定义界面
    nodeCreated:function (node) {
        if (node.comfyClass !== "FxAiVideoPreview") return;

        node.resizable = true;

        // 创建视频播放器元素（纯JS控制）
        const video = document.createElement("video");
        video.controls = true;
        video.style.width = "100%";
        video.style.borderRadius = "8px";
        video.style.maxHeight = "60vh";

        // 添加到节点
        node.addDOMWidget("video_player","container",video);

        // 接收后端传来的路径 → 自动播放
        node.onExecuted = function (output) {
            if (!output.path) return;
            video.src = `/fxai/video/preview?path=${output.path.join("")}&t=${Date.now()}`;
        };
    },
});