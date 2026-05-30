import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "FxAiVideoPreview",

    nodeCreated: function (node) {
        if (node.comfyClass !== "FxAiVideoPreview") {
            return;
        }

        node.resizable = true;

        // 创建视频元素
        var video = document.createElement("video");
        video.controls = true;
        video.style.width = "100%";
        video.style.borderRadius = "8px";
        video.style.maxHeight = "60vh";

        node.addDOMWidget("video_player", "container", video);

        // 刷新视频（切换标签时重新加载）
        function updateVideo() {
            if (node.outputs && node.outputs[0] && node.outputs[0].value) {
                var path = node.outputs[0].value;
                if (path) {
                    video.src = "/fxai/video/preview?path=" + encodeURIComponent(path) + "&t=" + Date.now();
                }
            }
        }

        // 节点执行完成后更新
        node.onExecuted = function (output) {
            if (!output.path) return;
            var path = output.path[0];
            video.src = "/fxai/video/preview?path=" + encodeURIComponent(path) + "&t=" + Date.now();
        };

        // 切换标签时重新渲染
        node.onGraphConfigured = function () {
            updateVideo();
        };

        // 显示时刷新
        node.onResize = function () {
            updateVideo();
        };
    }
});