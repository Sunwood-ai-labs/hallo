import gradio as gr
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/app'))

from inference import inference_process

def run_inference(config, source_image, driving_audio, output, pose_weight, face_weight, lip_weight, face_expand_ratio, checkpoint):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=config)
    parser.add_argument("--source_image", type=str, default=source_image)
    parser.add_argument("--driving_audio", type=str, default=driving_audio)
    parser.add_argument("--output", type=str, default=output)
    parser.add_argument("--pose_weight", type=float, default=pose_weight)
    parser.add_argument("--face_weight", type=float, default=face_weight)
    parser.add_argument("--lip_weight", type=float, default=lip_weight)
    parser.add_argument("--face_expand_ratio", type=float, default=face_expand_ratio)
    parser.add_argument("--checkpoint", type=str, default=checkpoint)

    args = parser.parse_args()
    
    # 推論処理実行
    inference_process(args)

    return output

inputs = [
    gr.Textbox(label="設定ファイル", value="/app/configs/inference/default.yaml"),
    gr.Image(label="ソース画像", type="filepath"),  
    gr.Audio(label="ドライビングオーディオ", type="filepath"),
    gr.Textbox(label="出力ビデオ", value=".cache/output.mp4"),
    gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="ポーズの重み"),
    gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="顔の重み"),
    gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="リップの重み"),    
    gr.Slider(minimum=0.5, maximum=2, step=0.1, value=1.2, label="顔の拡張比"),
    gr.Textbox(label="チェックポイント", value=None)
]

output = gr.Video(label="出力ビデオ")


gr.Interface(fn=run_inference, inputs=inputs, outputs=output, title="Hallo App", theme='gradio/monochrome').launch(share=True, debug=True, server_name="0.0.0.0")