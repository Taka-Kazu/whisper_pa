import whisper
from pulsectl import Pulse
from pulsectl import _pulsectl as c_pulse
import threading
import numpy as np
from scipy.io import wavfile
import time
import os

PA_SAMPLE_FLOAT32LE = 5
PA_SAMPLE_S16_LE = 3

if __name__ == "__main__":
    lock = threading.Lock()
    cond = threading.Condition(lock=lock)

    source_name = os.environ.get("PULSE_SOURCE")
    pulse = Pulse(threading_lock=lock)
    samples, proplist = [0], c_pulse.pa.proplist_from_string("application.id=whisper")
    hz = 44100
    ss = c_pulse.PA_SAMPLE_SPEC(format=PA_SAMPLE_FLOAT32LE, channels=1, rate=hz)
    s = c_pulse.pa.stream_new_with_proplist(
        pulse._ctx, "whisper", c_pulse.byref(ss), None, proplist
    )
    c_pulse.pa.proplist_free(proplist)

    recorded = np.empty(0, dtype=np.float32)

    @c_pulse.PA_STREAM_REQUEST_CB_T
    def cb_read(s, nbytes, userdata):
        global recorded
        size_of_float = int(c_pulse.sizeof(c_pulse.c_float))
        buff, bs = c_pulse.c_void_p(), c_pulse.c_int(nbytes)
        if bs.value < size_of_float:
            return
        c_pulse.pa.stream_peek(s, buff, c_pulse.byref(bs))
        samples = c_pulse.cast(buff, c_pulse.POINTER(c_pulse.c_float))
        size = int(bs.value / size_of_float)
        c_float_array = c_pulse.c_float * size
        samples = c_float_array.from_address(c_pulse.addressof(samples.contents))
        samples = np.ctypeslib.as_array(samples).copy()
        recorded = np.hstack((recorded, samples))
        c_pulse.pa.stream_drop(s)

    c_pulse.pa.stream_set_read_callback(s, cb_read, None)
    try:
        c_pulse.pa.stream_connect_record(
            s,
            source_name,
            None,
            c_pulse.PA_STREAM_ADJUST_LATENCY,
        )
    except c_pulse.pa.CallError:
        c_pulse.pa.stream_unref(s)
        exit(-1)

    end_flag = False

    def cb_listen():
        global end_flag
        try:
            while True:
                with lock:
                    if end_flag:
                        break
                pulse._pulse_poll(0.1)
                with cond:
                    cond.notify_all()
        finally:
            try:
                c_pulse.pa.stream_disconnect(s)
                c_pulse.pa.stream_unref(s)
            except c_pulse.pa.CallError:
                pass

    listening_thread = threading.Thread(target=cb_listen)
    listening_thread.start()

    def transcribe():
        global cond, end_flag, recorded

        model_name = "medium"
        print(f"Loading {model_name} model...")
        model = whisper.load_model(model_name, download_root="./.cache")
        print("Load completed")

        while True:
            print("Recording...")
            with cond:
                while not end_flag and recorded.shape and recorded.shape[0] / hz < 5:
                    cond.wait()
            with lock:
                if end_flag:
                    return
            print("Start transcribing...")
            with lock:
                # At least 10s
                end_idx = min(hz * 10, recorded.shape[0])
                wavfile.write("/tmp/cache.wav", rate=hz, data=recorded[: end_idx])
                audio = whisper.load_audio("/tmp/cache.wav")
                recorded = np.delete(recorded, range(0, end_idx), axis=0)

            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")

            options = whisper.DecodingOptions(fp16=True)
            result = whisper.decode(model, mel, options)

            print(result.text)
            wavfile.write("test.wav", rate=hz, data=audio)

    transcribe_thread = threading.Thread(target=transcribe)
    transcribe_thread.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            with lock:
                end_flag = True
            with cond:
                cond.notify_all()
                break

    transcribe_thread.join()
    listening_thread.join()
    pulse.close()
