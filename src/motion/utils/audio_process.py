import whisper
import json
import tqdm
import os
import re
import csv
import numpy as np
import copy
# from faster_whisper import WhisperModel


def get_asr_result(audio_path, file_path, asr_processed_path, save_file=False):
    print(file_path)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            result = json.load(file)
    else:
        # use asr to get sentence from speech
        model = whisper.load_model("large")
        result = model.transcribe(audio_path, word_timestamps=True)
        # model = WhisperModel("large")
        # result, _ = model.transcribe(audio_path, vad_filter=True, word_timestamps=True)
        new_entry = {'audio_path': audio_path}
        result = {**new_entry, **result}

        if save_file:
            print("save!")
            with open(file_path, "w") as file:
                json.dump(result, file, indent=4)


    # get the split sentence for llm
    # parts = re.split(r'(?<=[?!.])', result["text"])
    # res_sentence = [part.strip() for part in parts if part]


    parts = re.split(r'(\.{3}|[?!.])', result["text"])
    # Merge every two elements in the list (sentence and separator)
    merged_parts = [parts[i] + (parts[i+1] if i+1 < len(parts) else '') for i in range(0, len(parts), 2)]
    res_sentence = [part.strip() for part in merged_parts if part]

    # sentence = re.split(r'[.?!]', result["text"])
    # res_sentence = [item+'.' for item in sentence if item]

    # get the split word for vq insert

    fps = 60
    down_sample_rate = 8
    seg_data = result['segments']

    sentences_time_info = []
    sentence_info = []
    for seg_data_seq in seg_data:
        for word in seg_data_seq['words']:
            word_dict = {
                'word': word['word'].replace(" ", ""),
                'start_code': round(word['start']*fps/down_sample_rate),
                'end_code': round(word['end']*fps/down_sample_rate)
            }
            sentence_info.append(word_dict)
            if '.' in word['word'] or '?' in word['word'] or '!' in word['word']:
                sentence_dict = {
                    'sentence': sentence_info,
                    'sentence_start_code': sentence_info[0]['start_code'],
                    'sentence_end_code': sentence_info[-1]['end_code']
                }
                sentences_time_info.append(sentence_dict)
                sentence_info = []
    if sentence_info:
        sentence_dict = {
            'sentence': sentence_info,
            'sentence_start_code': sentence_info[0]['start_code'],
            'sentence_end_code': sentence_info[-1]['end_code']
        }
        sentences_time_info.append(sentence_dict)

    with open(asr_processed_path, "w") as file:
        json.dump(sentences_time_info, file, indent=4)

    if len(res_sentence) != len(sentences_time_info):
        print(audio_path)
        print(len(res_sentence), len(sentences_time_info))
        print(res_sentence)
    # assert(len(res_sentence) == len(sentences_time_info))

    return res_sentence, sentences_time_info


def speech_diarization_into_one_json(file_paths, save_merge_asr_path):
    asr_file_info = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            asr_res = json.load(file)
        asr_file_info.append(asr_res)

    actor_1_asr = asr_file_info[0]
    actor_2_asr = asr_file_info[1]

    actor_1_ptr = 0
    actor_2_ptr = 0
    timestamp_ptr = 0
    actor_1_sentence_num = len(actor_1_asr["segments"])
    actor_2_sentence_num = len(actor_2_asr["segments"])

    merge_asr_res = []
    while(actor_1_ptr < actor_1_sentence_num and actor_2_ptr < actor_2_sentence_num):
        actor_1_current_start_time = actor_1_asr["segments"][actor_1_ptr]["start"]
        actor_1_current_end_time = actor_1_asr["segments"][actor_1_ptr]["end"]
        actor_2_current_start_time = actor_2_asr["segments"][actor_2_ptr]["start"]
        actor_2_current_end_time = actor_2_asr["segments"][actor_2_ptr]["end"]

        current_ptr_sentence = {}

        if actor_1_current_start_time < actor_2_current_start_time:
            timestamp_ptr = actor_1_current_end_time
            current_ptr_sentence["actor"] = 'character_1'
            current_ptr_sentence["start_time"] = actor_1_current_start_time
            current_ptr_sentence["end_time"] = actor_1_current_end_time
            current_ptr_sentence["sentence"] = actor_1_asr["segments"][actor_1_ptr]["text"]
            current_ptr_sentence["words"] = actor_1_asr["segments"][actor_1_ptr]["words"]
            actor_1_ptr += 1

        else:
            timestamp_ptr = actor_2_current_end_time
            current_ptr_sentence["actor"] = 'character_2'
            current_ptr_sentence["start_time"] = actor_2_current_start_time
            current_ptr_sentence["end_time"] = actor_2_current_end_time
            current_ptr_sentence["sentence"] = actor_2_asr["segments"][actor_2_ptr]["text"]
            current_ptr_sentence["words"] = actor_2_asr["segments"][actor_2_ptr]["words"]
            actor_2_ptr += 1

        merge_asr_res.append(current_ptr_sentence)

    if actor_1_ptr == actor_1_sentence_num and actor_2_ptr < actor_2_sentence_num:
        for i in range(actor_2_ptr, actor_2_sentence_num):
            current_ptr_sentence = {}
            current_ptr_sentence["actor"] = 'character_2'
            current_ptr_sentence["start_time"] = actor_2_asr["segments"][i]["start"]
            current_ptr_sentence["end_time"] = actor_2_asr["segments"][i]["end"]
            current_ptr_sentence["sentence"] = actor_2_asr["segments"][i]["text"]
            current_ptr_sentence["words"] = actor_2_asr["segments"][i]["words"]
            merge_asr_res.append(current_ptr_sentence)
    elif actor_2_ptr == actor_2_sentence_num and actor_1_ptr < actor_1_sentence_num:
        for i in range(actor_1_ptr, actor_1_sentence_num):
            current_ptr_sentence = {}
            current_ptr_sentence["actor"] = 'character_1'
            current_ptr_sentence["start_time"] = actor_1_asr["segments"][i]["start"]
            current_ptr_sentence["end_time"] = actor_1_asr["segments"][i]["end"]
            current_ptr_sentence["sentence"] = actor_1_asr["segments"][i]["text"]
            current_ptr_sentence["words"] = actor_1_asr["segments"][i]["words"]
            merge_asr_res.append(current_ptr_sentence)

    with open(save_merge_asr_path, 'w', encoding='utf-8') as file:
        json.dump(merge_asr_res, file, ensure_ascii=False, indent=4)
    # assert(0)


def speech_diarization_into_multi_txt(merge_asr_res_path, final_save_path):
    with open(merge_asr_res_path, 'r', encoding='utf-8') as file:
        merge_asr_res = json.load(file)

    sentence_num = len(merge_asr_res)
    previous_actor = merge_asr_res[0]["actor"]

    new_merge_asr_res = []
    results_trasncripts_txt = ""
    for i in range(0, sentence_num):
        current_actor = merge_asr_res[i]["actor"]
        current_sentence = merge_asr_res[i]["sentence"]
        if i > 0 and current_actor == previous_actor:
            results_trasncripts_txt = results_trasncripts_txt.rstrip('\n')
            results_trasncripts_txt += current_sentence + '\n'
            new_merge_asr_res[-1]["sentence"] = new_merge_asr_res[-1]["sentence"].rstrip('\n') + merge_asr_res[i]["sentence"]
            new_merge_asr_res[-1]["actor"] = current_actor
            new_merge_asr_res[-1]["words"] = new_merge_asr_res[-1]["words"] + merge_asr_res[i]["words"]
            new_merge_asr_res[-1]["end_time"] = merge_asr_res[i]["end_time"]
        else:
            results_trasncripts_txt += current_actor + ": "+ current_sentence + '\n'
            new_merge_asr_res.append(merge_asr_res[i])
        previous_actor = current_actor

    with open(final_save_path, 'w') as file:
        file.write(results_trasncripts_txt)

    save_new_merge_asr_path = os.path.join(os.path.dirname(merge_asr_res_path), 'merge_asr_res_again.json')
    with open(save_new_merge_asr_path, 'w', encoding='utf-8') as file:
        json.dump(new_merge_asr_res, file, ensure_ascii=False, indent=4)
    return results_trasncripts_txt, save_new_merge_asr_path


def get_double_asr_result_in_one_json(file_paths, save_path):
    save_path = os.path.join(save_path, 'asr')
    os.makedirs(save_path, exist_ok=True)
    asr_processed_path_res = []
    for audio_path in file_paths:
        file_name = audio_path.split('/')[-1].split('.')[0]
        asr_file_path = os.path.join(save_path, file_name + '_asr.json')
        asr_processed_path = os.path.join(save_path, file_name + '_asr_processed.json')
        res_sentence, sentences_time_info = get_asr_result(audio_path, asr_file_path, asr_processed_path, 1)
        asr_processed_path_res.append(asr_file_path)

    save_merge_asr_path = os.path.join(save_path, 'merge_asr_res.json')
    final_save_path = os.path.join(save_path, 'merge_asr_res_only_txt.txt')
    speech_diarization_into_one_json(asr_processed_path_res, save_merge_asr_path)
    results_trasncripts_txt, save_new_merge_asr_path = speech_diarization_into_multi_txt(save_merge_asr_path, final_save_path)


    return save_new_merge_asr_path, final_save_path, results_trasncripts_txt


def asr_time_parser(merge_asr_res_path, start_time=5.0, interval=2.5, end_time=100.0, save_path=None):
    with open(merge_asr_res_path, 'r', encoding='utf-8') as file:
        merge_asr_res = json.load(file)


    all_segments = {}
    all_segments_words = {}
    all_segments_only_next_timespan = {}
    seq_current_time = start_time
    seq_end_time = start_time + interval


    save_index = 1
    while seq_current_time < end_time:

        current_segment = ''
        current_segment_only_next_timespan = ''
        current_segment_words = []
        for i, entry in enumerate(merge_asr_res, start=0):

            previous_word = ''
            after_word = ''
            remain_word = ''
            if entry["end_time"] < seq_end_time and entry["end_time"] > seq_current_time:
                for word in entry['words']:
                    if word['end'] < seq_current_time:
                        previous_word += word['word']
                    else:
                        remain_word += word['word']
                        current_segment_words.append(word)
                if previous_word:
                    current_segment += entry['actor'] + ": "+ previous_word + "{" + remain_word + "}" + '\n'
                else:
                    current_segment += entry['actor'] + ": {"+ remain_word + "}" + '\n'

                if remain_word:
                    current_segment_only_next_timespan += entry['actor'] + ": " + remain_word + '\n'
            elif entry["start_time"] < seq_end_time and entry["start_time"] > seq_current_time:
                for word in entry['words']:
                    if word['end'] < seq_end_time:
                        remain_word += word['word']
                        current_segment_words.append(word)
                    else:
                        after_word += word['word']
                if after_word:
                    current_segment += entry['actor'] + ": {"+ remain_word + "}"+ after_word + '\n'
                else:
                    current_segment += entry['actor'] + ": {"+ remain_word + "}" + '\n'
                if remain_word:
                    current_segment_only_next_timespan += entry['actor'] + ": " + remain_word + '\n'
            elif entry["start_time"] < seq_current_time and entry["end_time"] > seq_end_time:
                for word in entry['words']:
                    if word['end'] < seq_current_time:
                        previous_word += word['word']
                    elif word['end'] < seq_end_time:
                        remain_word += word['word']
                        current_segment_words.append(word)
                    else:
                        after_word += word['word']
                current_segment += entry['actor'] + ": "
                if previous_word:
                    current_segment += previous_word
                current_segment += " {"+ remain_word + "}"
                if after_word:
                    current_segment += after_word
                current_segment += '\n'

                if remain_word:
                    current_segment_only_next_timespan += entry['actor'] + ": " + remain_word + '\n'
            elif entry["start_time"] > seq_end_time:
                continue


        if len(current_segment) > 0:
            all_segments[save_index] = current_segment
            all_segments_words[save_index] = current_segment_words
            all_segments_only_next_timespan[save_index] = current_segment_only_next_timespan
        else:
            all_segments[save_index] = ' '
            all_segments_words[save_index] = []
            all_segments_only_next_timespan[save_index] = ' '
        save_index += 1


        seq_current_time += interval
        seq_end_time += interval

    save_path = os.path.join(save_path, 'asr')
    final_save_path = os.path.join(save_path, 'merge_asr_res_time_span.json')
    with open(final_save_path, 'w', encoding='utf-8') as file:
        json.dump(all_segments, file, ensure_ascii=False, indent=4)

    final_save_path_words = os.path.join(save_path, 'merge_asr_res_time_span_words.json')
    with open(final_save_path_words, 'w', encoding='utf-8') as file:
        json.dump(all_segments_words, file, ensure_ascii=False, indent=4)

    final_save_path_only_next_timespan = os.path.join(save_path, 'merge_asr_res_time_span_only_next_timespan.json')
    with open(final_save_path_only_next_timespan, 'w', encoding='utf-8') as file:
        json.dump(all_segments_only_next_timespan, file, ensure_ascii=False, indent=4)

    return all_segments, all_segments_only_next_timespan, all_segments_words


def speaker_recognition(file_paths, window_size=5, step_size=2.5, threshold=0.4):
    from pyAudioAnalysis import audioBasicIO as aIO
    from pyAudioAnalysis import audioSegmentation as aS
    import noisereduce as nr

    def process_single_file(wav_file):

        [Fs, x] = aIO.read_audio_file(wav_file)
        denoised_frames = nr.reduce_noise(y=x, sr=Fs)

        segments = aS.silence_removal(denoised_frames, Fs, 0.020, 0.020, smooth_window=0.5, weight=0.5, plot=False)

        total_seconds = len(x) / Fs
        window_start = 0
        window_end = window_size

        window_results = []

        while 1:
            window_duration = window_end - window_start
            speech_in_window = 0.0

            for segment in segments:
                if(segment[0] >= window_end):
                    continue
                if(segment[1] <= window_start):
                    continue
                overlap_start = max(segment[0], window_start)
                overlap_end = min(segment[1], window_end)
                overlap_duration = overlap_end - overlap_start
                speech_in_window += overlap_duration / window_duration

            if speech_in_window >= threshold:
                window_results.append(True)
            else:
                window_results.append(False)

            if(window_end >= total_seconds):
                break

            if(window_start == 0):
                window_start += step_size
            window_start += step_size
            window_end += step_size

        return window_results

    results = {}
    assert len(file_paths) == 2
    wav_file_1, wav_file_2 = file_paths
    results[wav_file_1] = process_single_file(wav_file_1)
    results[wav_file_2] = process_single_file(wav_file_2)

    return results


def emotion_recognition(file_paths):
    from funasr import AutoModel
    model = AutoModel(model="iic/emotion2vec_plus_large") # seed, base, large

    def process_result(res):
        labels = res[0]['labels']
        scores = res[0]['scores']
        sorted_scores = sorted(zip(scores, labels), reverse=True, key=lambda x: x[0])
        if sorted_scores[0][0] > 0.5:
            return [sorted_scores[0][1].split('/')[-1]]
        else:
            return [sorted_scores[0][1].split('/')[-1], sorted_scores[1][1].split('/')[-1]]

    assert len(file_paths) == 2
    wav_file_1, wav_file_2 = file_paths
    res_1 = model.generate(wav_file_1, granularity="utterance", extract_embedding=False)
    res_2 = model.generate(wav_file_2, granularity="utterance", extract_embedding=False)

    results = {}
    results[wav_file_1] = process_result(res_1)
    results[wav_file_2] = process_result(res_2)

    return results


if __name__ == "__main__":
    test_audios_paths = ['./data/test_audios/audio/20231119_001_052.wav', './data/test_audios/audio/20231119_002_052.wav']
    wav_save_dir = './data/test_audios/wav'
    asr_save_dir = './data/test_audios/asr'
    os.makedirs(wav_save_dir, exist_ok=True)
    os.makedirs(asr_save_dir, exist_ok=True)
    # extract_waveform(test_audios_paths, wav_save_dir, fps=30, sample_rate=16000)
    asr_res, _, _ = get_double_asr_result_in_one_json(test_audios_paths, asr_save_dir)
    asr_time_parser(asr_res, save_path=asr_save_dir)
    print(emotion_recognition(test_audios_paths))
    print(speaker_recognition(test_audios_paths))
