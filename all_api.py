import os
import sys
import io
import argparse
import numpy as np
import torch
import torchaudio
import random
import librosa
from scipy.io.wavfile import write

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
import shutil

max_val = 0.8
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"

prompt_sr, target_sr = 16000, 22050
default_data = np.zeros(target_sr)
from flask import Flask, request, Response, send_from_directory, jsonify
from flask_cors import CORS
from flask import make_response
import aliyun_oss

app = Flask(__name__)
CORS(app, cors_allowed_origins="*")
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz')
CORS(app, supports_credentials=True)


def save_name(name, opt_path):
    if not name or name == "":
        return False
    shutil.copyfile(f"{ROOT_DIR}/{opt_path}", f"{ROOT_DIR}/voices/{name}.pt")


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def delete_all_files_in_folder(folder_path):
    # 确保文件夹存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # 删除文件
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹及其内容
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除 {file_path} 失败: {e}")

        # 删除文件夹本身
        try:
            os.rmdir(folder_path)  # 删除空文件夹
            print(f"已删除空文件夹: {folder_path}")
        except Exception as e:
            print(f"删除文件夹 {folder_path} 失败: {e}")
    else:
        print(f"文件夹 {folder_path} 不存在或不是一个有效的文件夹")


def generate_audio(mode, tts_text, spk_id, spk_customize, prompt_text, prompt_speech_16k, seed=0, stream="False",
                   speed=1.0, user_id="", exp_name=""):
    global cosyvoice
    print(f"mode:{mode}, tts_text:{tts_text}, spk_id:{spk_id}, spk_customize:{spk_customize}, prompt_text:{prompt_text}, prompt_speech_16k:{prompt_speech_16k}, seed:{seed}, stream:{stream}, speed:{speed}, user_id:{user_id}, exp_name:{exp_name}")
    set_all_random_seed(seed)
    stream = stream == "True"
    if mode == '预训练音色':
        if spk_customize != "无" and not os.path.exists(f"{ROOT_DIR}/output/{user_id}/{spk_customize}.pt"):
            aliyun_oss.download_file(f"{user_id}/model/cosyvoice/{spk_customize}.pt",
                                     f"{ROOT_DIR}/output/{user_id}/{spk_customize}.pt")
        if stream:
            def generate():
                for i, j in enumerate(cosyvoice.inference_sft(tts_text, spk_id, stream, speed, spk_customize, user_id)):
                    tts_speeches = []
                    buffer = io.BytesIO()
                    tts_speeches.append(j['tts_speech'])
                    audio_data = torch.concat(tts_speeches, dim=1)
                    torchaudio.save(buffer, audio_data, target_sr, format="ogg")
                    buffer.seek(0)
                    yield buffer.read()
                delete_all_files_in_folder(f"{ROOT_DIR}/output/{user_id}")

            response = make_response(generate())
            response.headers['Content-Type'] = 'audio/ogg'
            response.headers['Content-Disposition'] = 'inline; filename=sound.ogg'
            print(response)
            return response
        else:
            buffer = io.BytesIO()
            tts_speeches = []
            for i, j in enumerate(cosyvoice.inference_sft(tts_text, spk_id, stream, speed, spk_customize)):
                tts_speeches.append(j['tts_speech'])
            audio_data = torch.concat(tts_speeches, dim=1)
            torchaudio.save(buffer, audio_data, target_sr, format="wav")
            buffer.seek(0)
            delete_all_files_in_folder(f"{ROOT_DIR}/output/{user_id}")
            return Response(buffer.read(), mimetype="audio/wav")
    elif mode == '3s极速复刻':
        audio_file = f"{ROOT_DIR}/参考音频/{prompt_speech_16k}"
        if not os.path.exists(audio_file):
            aliyun_oss.download_file(prompt_speech_16k, audio_file)
        prompt_speech_16k = audio_file
        prompt_speech_16k = postprocess(load_wav(prompt_speech_16k, prompt_sr))
        output_path = f"output/{user_id}/{exp_name}.pt"
        print("output_path" + output_path)
        file_path = f"output/{user_id}"
        print("create_path:" + file_path)
        if not os.path.exists(f"{ROOT_DIR}/{file_path}"):
            os.makedirs(f"{ROOT_DIR}/{file_path}")
        print(f"创建了：{ROOT_DIR}/{file_path}")
        if stream:
            def generate():
                for i, j in enumerate(
                        cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream, speed,
                                                      output_path)):
                    tts_speeches = []
                    buffer = io.BytesIO()
                    tts_speeches.append(j['tts_speech'])
                    audio_data = torch.concat(tts_speeches, dim=1)
                    torchaudio.save(buffer, audio_data, target_sr, format="ogg")
                    buffer.seek(0)
                    yield buffer.read()
                try:
                    aliyun_oss.upload_file(f"{user_id}/model/cosyvoice/{exp_name}.pt",
                                           f"{ROOT_DIR}/{file_path}/{exp_name}.pt")
                    delete_all_files_in_folder(f"{ROOT_DIR}/{file_path}")
                    last_index_ = audio_file.rfind('/')
                    del_file_ = audio_file[:last_index_]
                    delete_all_files_in_folder(del_file_)
                    print("模型生成并上传成功")
                except Exception as e:
                    print(e)

            response = make_response(generate())
            response.headers['Content-Type'] = 'audio/ogg'
            # response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
            response.headers['Content-Disposition'] = 'inline; filename=sound.ogg'
            return response
        else:
            buffer = io.BytesIO()
            tts_speeches = []
            for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream, speed,
                                                                output_path)):
                tts_speeches.append(j['tts_speech'])
            audio_data = torch.concat(tts_speeches, dim=1)
            torchaudio.save(buffer, audio_data, target_sr, format="wav")
            buffer.seek(0)
            aliyun_oss.upload_file(f"{user_id}/model/cosyvoice/{exp_name}.pt",
                                   f"{ROOT_DIR}/{file_path}/{exp_name}.pt")
            delete_all_files_in_folder(f"{ROOT_DIR}/{file_path}")
            last_index_ = audio_file.rfind('/')
            del_file_ = audio_file[:last_index_]
            delete_all_files_in_folder(del_file_)
            return Response(buffer.read(), mimetype="audio/wav")
    elif mode == '跨语种复刻':
        audio_file = f"{ROOT_DIR}/参考音频/{prompt_speech_16k}"
        if not os.path.exists(audio_file):
            aliyun_oss.download_file(prompt_speech_16k, audio_file)
            prompt_speech_16k = audio_file
        prompt_speech_16k = postprocess(load_wav(prompt_speech_16k, prompt_sr))
        if stream:
            def generate():
                for i, j in enumerate(cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream, speed)):
                    tts_speeches = []
                    buffer = io.BytesIO()
                    tts_speeches.append(j['tts_speech'])
                    audio_data = torch.concat(tts_speeches, dim=1)
                    torchaudio.save(buffer, audio_data, target_sr, format="ogg")
                    buffer.seek(0)
                    yield buffer.read()
                last_index_ = audio_file.rfind('/')
                del_file_ = audio_file[:last_index_]
                delete_all_files_in_folder(del_file_)

            response = make_response(generate())
            response.headers['Content-Type'] = 'audio/ogg'
            # response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
            response.headers['Content-Disposition'] = 'inline; filename=sound.ogg'
            return response
        else:
            buffer = io.BytesIO()
            tts_speeches = []
            for i, j in enumerate(cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream, speed)):
                tts_speeches.append(j['tts_speech'])
            audio_data = torch.concat(tts_speeches, dim=1)
            torchaudio.save(buffer, audio_data, target_sr, format="wav")
            buffer.seek(0)
            last_index_ = audio_file.rfind('/')
            del_file_ = audio_file[:last_index_]
            delete_all_files_in_folder(del_file_)
            return Response(buffer.read(), mimetype="audio/wav")
    elif mode == '自然语言控制':
        if not os.path.exists(f"{ROOT_DIR}/output/{user_id}/{exp_name}.pt"):
            aliyun_oss.download_file(f"{user_id}/model/cosyvoice/{exp_name}.pt",
                                     f"{ROOT_DIR}/output/{user_id}/{exp_name}.pt")
        if stream:
            def generate():
                for i, j in enumerate(
                        cosyvoice.inference_instruct(tts_text, spk_id, prompt_text, stream, speed, spk_customize,
                                                     user_id)):
                    tts_speeches = []
                    buffer = io.BytesIO()
                    tts_speeches.append(j['tts_speech'])
                    audio_data = torch.concat(tts_speeches, dim=1)
                    torchaudio.save(buffer, audio_data, target_sr, format="ogg")
                    buffer.seek(0)
                    yield buffer.read()
                delete_all_files_in_folder(f"{ROOT_DIR}/output/{user_id}")

            response = make_response(generate())
            response.headers['Content-Type'] = 'audio/ogg'
            # response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
            response.headers['Content-Disposition'] = 'inline; filename=sound.ogg'
            return response
        else:
            buffer = io.BytesIO()
            tts_speeches = []
            for i, j in enumerate(
                    cosyvoice.inference_instruct(tts_text, spk_id, prompt_text, stream, speed, spk_customize, user_id)):
                tts_speeches.append(j['tts_speech'])
            audio_data = torch.concat(tts_speeches, dim=1)
            torchaudio.save(buffer, audio_data, target_sr, format="wav")
            buffer.seek(0)
            delete_all_files_in_folder(f"{ROOT_DIR}/output/{user_id}")
            return Response(buffer.read(), mimetype="audio/wav")


@app.route("/generate_audio", methods=['POST'])
def generate_audio_api():
    request_data = request.get_json()
    print(request_data.get('mode'), request_data.get('tts_text'), request_data.get('spk_id'))
    return generate_audio(
        request_data.get('mode'),
        request_data.get('tts_text'),
        request_data.get('spk_id'),
        request_data.get('spk_customize', '无'),
        request_data.get('prompt_text'),
        request_data.get('prompt_speech_16k'),
        request_data.get('seed', 0),
        request_data.get('stream', "True"),
        request_data.get('speed', 1.0),
        request_data.get('user_id', ""),
        request_data.get('exp_name', "")
    )


@app.route("/change_instruct", methods=['GET'])
def change_instr_api():
    global cosyvoice
    cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
    return jsonify({"status": "success"})


@app.route("/generate_audio_get", methods=['GET'])
def generate_audio_get():
    return generate_audio(
        "预训练音色",
        "在那白日烈火所触不到的黑暗里，深渊一般的黑，似乎带走了所有生气，淤泥，昆虫，秽物横流。但，在我们连看都不想看的地方，却仍能将春一般的美好展现出来——苔！青苔！生不如夏花般绚烂，死没有秋草般静美。她就在那里，不动着，静静着，绽放得宛如牡丹，享受着自己的春，尽管没有光明。人生啊！何尝不是这样，堕入过最无边的黑暗，失去光明呵！失去前进的步伐呵！我们为什么不看看青苔，那是比莲更高洁的存在——“出淤泥而不染”。但他却比莲更平凡。",
        "中文女",
        "无",
        None,
        None,
        0,
        "True",
        1.0,
        "1321",
        ""
    )


@app.route("/save", methods=['GET'])
def save_model():
    path = "output/" + request.args.get('user_id') + '/' + request.args.get("id") + "/" + request.args.get("id") + ".pt"
    save_name(request.args.get('name'), path)
    return jsonify({"status": "success"})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=9883,
                        help='Port for the Flask app')
    args = parser.parse_args()
    sft_spk = cosyvoice.list_avaliable_spks()

    # Use the port from argparse
    app.run(host='0.0.0.0', port=args.port)
